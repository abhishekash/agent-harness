import io

from agent_harness.agent import Agent
from agent_harness.providers import say, scripted_run
from agent_harness.summary import RollingSummary, clamp_summary
from agent_harness.tracing import init_tracing
from agent_harness.ui import MinimalTerminalUI


def test_clamp_summary_is_two_lines_and_removes_model_scaffolding():
    summary = clamp_summary("Summary:\n- read the notes\n- wrote the output\n- verified the result")
    assert len(summary.splitlines()) <= 2
    assert "Summary:" not in summary
    assert summary.endswith("…")


def test_progress_summary_uses_same_provider_without_consuming_agent_script(tmp_path):
    provider = scripted_run(say("done"))
    events = []
    progress = RollingSummary(provider)
    agent = Agent(
        provider,
        [],
        tracer_provider=init_tracing(tmp_path / "trace.jsonl"),
        progress_summarizer=progress,
        event_handler=lambda name, data: events.append((name, data)),
    )

    result = agent.run("finish the task")

    assert result.answer == "done"
    assert result.summary
    assert len(result.summary.splitlines()) <= 2
    assert provider.calls == 1  # the summary uses the provider's same-model hook
    assert [name for name, _ in events].count("summary") == 1
    assert result.usage.total > 0


def test_minimal_ui_keeps_summary_content_to_two_lines():
    stream = io.StringIO()
    ui = MinimalTerminalUI(stream, mode="always")
    ui.start("task", "scripted", "one\ntwo\nthree")
    ui.handle_event("summary", {"text": "first\nsecond\nthird"})
    ui.finish("first\nsecond\nthird")

    # Strip ANSI control sequences only for this small rendering invariant.
    import re

    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", stream.getvalue())
    assert "  first\n  second" in plain
