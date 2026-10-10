import pytest

from agent_harness.agent import Agent
from agent_harness.checkpoint import CheckpointError, CheckpointStore
from agent_harness.context import ContextManager, ContextPolicy
from agent_harness.hitl import AutoApprover
from agent_harness.providers import ResilientProvider, RetryPolicy, call_tool, say, scripted_run
from agent_harness.tools import default_tools
from agent_harness.tracing import init_tracing, load_spans
from agent_harness.types import AssistantMessage, CancellationToken, Message, RunLimits, ToolCall, Usage


class FlakyProvider:
    model = "test"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls < 3:
            raise TimeoutError("temporary transport failure")
        return AssistantMessage("ok", usage=Usage(input_tokens=1, output_tokens=1))


def test_resilient_provider_retries_only_transient_failures():
    provider = FlakyProvider()
    sleeps = []
    wrapped = ResilientProvider(
        provider,
        RetryPolicy(max_attempts=3, base_delay_s=0, jitter=0),
        sleep=sleeps.append,
    )

    assert wrapped.complete([], []).content == "ok"
    assert provider.calls == 3
    assert sleeps == [0, 0]


def test_context_manager_compacts_at_a_safe_assistant_boundary():
    provider = scripted_run(say("handoff"))
    manager = ContextManager(
        provider,
        ContextPolicy(max_chars=100, keep_recent_messages=2, max_summary_input_chars=500),
    )
    messages = [Message.system("rules"), Message.user("task")]
    for index in range(5):
        call = ToolCall.new("read", {})
        messages.append(Message(role="assistant", content=f"step {index}", tool_calls=[call]))
        messages.append(Message.tool_result(call, "result"))

    compacted = manager.maybe_compact(messages)

    assert compacted.compacted
    assert compacted.dropped_messages > 0
    assert compacted.messages[0].content.startswith("rules\n\n[Earlier conversation compacted")
    assert compacted.messages[1].content == "task"
    assert compacted.messages[2].role == "assistant"


def test_checkpoint_round_trip_is_private_and_atomic(tmp_path):
    path = tmp_path / "run.checkpoint.json"
    store = CheckpointStore(path)
    from agent_harness.checkpoint import CheckpointState

    store.save(CheckpointState("task", [Message.user("task")], 2))
    loaded = store.load()

    assert loaded.task == "task"
    assert loaded.next_step == 2
    assert path.stat().st_mode & 0o077 == 0


def test_resume_continues_only_from_a_completed_tool_boundary(tmp_path):
    class StatefulProvider:
        model = "stateful-test"

        def complete(self, messages, tools):
            if any(message.role == "tool" for message in messages):
                return AssistantMessage("finished", usage=Usage(1, 1))
            return AssistantMessage(
                "",
                [ToolCall.new("read_file", {"path": "notes.md"})],
                Usage(1, 1),
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("notes")
    checkpoint = CheckpointStore(tmp_path / "run.checkpoint.json")
    provider = StatefulProvider()

    first = Agent(
        provider,
        default_tools(workspace),
        approver=AutoApprover(),
        max_steps=1,
        checkpoint_store=checkpoint,
        tracer_provider=init_tracing(tmp_path / "first.jsonl"),
    ).run("read notes")
    assert first.stopped_reason == "max_steps"
    assert checkpoint.path.exists()

    second = Agent(
        provider,
        default_tools(workspace),
        approver=AutoApprover(),
        max_steps=2,
        checkpoint_store=checkpoint,
        resume=True,
        tracer_provider=init_tracing(tmp_path / "second.jsonl"),
    ).run("read notes")
    assert second.answer == "finished"
    assert second.stopped_reason == "completed"
    assert not checkpoint.path.exists()


def test_run_stops_on_a_hard_token_budget(tmp_path):
    result = Agent(
        scripted_run(say("this is more than one token")),
        [],
        tracer_provider=init_tracing(tmp_path / "budget.jsonl"),
        limits=RunLimits(max_steps=2, max_output_tokens=1),
    ).run("task")

    assert result.stopped_reason == "budget_exceeded"
    assert "output token limit" in result.answer


def test_pre_cancelled_run_does_not_call_provider(tmp_path):
    token = CancellationToken()
    token.cancel("operator requested stop")
    provider = scripted_run(say("must not run"))

    result = Agent(
        provider,
        [],
        tracer_provider=init_tracing(tmp_path / "cancel.jsonl"),
        cancellation=token,
    ).run("task")

    assert result.stopped_reason == "cancelled"
    assert provider.calls == 0


def test_tool_error_is_queryable_without_recording_tool_output(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    trace = tmp_path / "escape.jsonl"
    result = Agent(
        scripted_run(
            call_tool("run_shell", {"command": "cat ../outside.txt"}),
            say("blocked"),
        ),
        default_tools(workspace),
        approver=AutoApprover(),
        tracer_provider=init_tracing(trace),
    ).run("inspect parent")

    tool_span = next(span for span in load_spans(trace) if span["name"] == "tool.call")
    assert tool_span["attributes"]["tool.result_status"] == "error"
    assert "shell path escapes workspace root" in tool_span["attributes"]["tool.error_message"]
    assert "outside.txt" in tool_span["attributes"]["tool.error_message"]
    assert result.answer == "blocked"


def test_resume_refuses_an_inflight_side_effect(tmp_path):
    from agent_harness.checkpoint import CheckpointState

    checkpoint = CheckpointStore(tmp_path / "pending.json")
    checkpoint.save(
        CheckpointState(
            "task",
            [Message.user("task")],
            1,
            pending_tools=[ToolCall.new("write_file", {"path": "x", "content": "y"})],
            provider_model="stateful-test",
        )
    )
    with pytest.raises(CheckpointError, match="in-flight"):
        Agent(
            FlakyProvider(),
            [],
            checkpoint_store=checkpoint,
            resume=True,
        ).run("task")
