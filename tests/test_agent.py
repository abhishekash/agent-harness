"""End-to-end agent tests: scripted provider through the real loop."""
import pytest

from agent_harness.agent import Agent
from agent_harness.hitl import AutoApprover, CallbackApprover, Decision, DenyAllApprover
from agent_harness.providers import call_tool, say, scripted_run
from agent_harness.providers.base import ProviderError
from agent_harness.tools import default_tools
from agent_harness.tracing import init_tracing, load_spans


def make_agent(script, workspace, tmp_path, approver=None, **kwargs):
    trace_path = tmp_path / "traces.jsonl"
    tp = init_tracing(trace_path)
    agent = Agent(
        scripted_run(*script),
        default_tools(workspace),
        approver=approver or AutoApprover(),
        tracer_provider=tp,
        **kwargs,
    )
    return agent, tp, trace_path


def test_happy_path_traces_everything(workspace, tmp_path):
    agent, tp, trace_path = make_agent(
        [
            call_tool("read_file", {"path": "notes.md"}),
            call_tool("write_file", {"path": "SUMMARY.md", "content": "done"}),
            say("wrote SUMMARY.md"),
        ],
        workspace,
        tmp_path,
    )
    result = agent.run("summarize notes")
    assert result.stopped_reason == "completed"
    assert result.answer == "wrote SUMMARY.md"
    assert (workspace / "SUMMARY.md").read_text() == "done"
    assert len(result.approvals) == 1
    assert result.usage.total > 0

    tp.force_flush()
    spans = load_spans(trace_path)
    names = [s["name"] for s in spans]
    assert names.count("llm.complete") == 3
    assert names.count("tool.call") == 2
    hitl = [e for s in spans for e in s["events"] if e["name"] == "hitl.decision"]
    assert len(hitl) == 1
    assert hitl[0]["attributes"]["hitl.decision"] == "approve"
    root = next(s for s in spans if s["name"] == "agent.run")
    assert root["attributes"]["agent.stopped_reason"] == "completed"


def test_denied_tool_feeds_reason_back(workspace, tmp_path):
    agent, tp, trace_path = make_agent(
        [
            call_tool("write_file", {"path": "evil.sh", "content": "rm -rf /"}),
            say("Understood, I won't do that."),
        ],
        workspace,
        tmp_path,
        approver=DenyAllApprover(rationale="no destructive writes"),
    )
    result = agent.run("write evil.sh")
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert "no destructive writes" in tool_msgs[0].content
    assert "Do not retry" in tool_msgs[0].content
    assert not (workspace / "evil.sh").exists()

    tp.force_flush()
    spans = load_spans(trace_path)
    denied_span = next(s for s in spans if s["attributes"].get("tool.denied"))
    assert denied_span["name"] == "tool.call"


def test_edited_arguments_are_used(workspace, tmp_path):
    agent, _, _ = make_agent(
        [
            call_tool("write_file", {"path": "../escape.txt", "content": "x"}),
            say("done"),
        ],
        workspace,
        tmp_path,
        approver=CallbackApprover(lambda r: (Decision.EDIT, {"path": "safe.txt", "content": "x"})),
    )
    result = agent.run("escape")
    assert (workspace / "safe.txt").exists()
    assert result.approvals[0].decision is Decision.EDIT


def test_unknown_tool_is_reported_not_raised(workspace, tmp_path):
    agent, _, _ = make_agent(
        [call_tool("hack_the_planet", {}), say("ok")],
        workspace,
        tmp_path,
    )
    result = agent.run("hack")
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert "unknown tool" in tool_msgs[0].content


def test_max_steps_guard(workspace, tmp_path):
    agent, _, _ = make_agent(
        [call_tool("list_dir", {})] * 50,
        workspace,
        tmp_path,
        max_steps=3,
    )
    result = agent.run("loop")
    assert result.stopped_reason == "max_steps"
    assert result.steps == 3


def test_provider_error_surfaces_cleanly(workspace, tmp_path):
    class BrokenProvider:
        model = "broken"

        def complete(self, messages, tools):
            raise ProviderError("rate limited")

    trace_path = tmp_path / "t.jsonl"
    agent = Agent(
        BrokenProvider(),
        default_tools(workspace),
        tracer_provider=init_tracing(trace_path),
    )
    result = agent.run("anything")
    assert result.stopped_reason == "error"
    assert "rate limited" in result.answer


def test_bad_tool_arguments_return_error(workspace, tmp_path):
    agent, _, _ = make_agent(
        [call_tool("write_file", {"wrong": "args"}), say("recovered")],
        workspace,
        tmp_path,
    )
    result = agent.run("bad args")
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    assert "bad arguments" in tool_msgs[0].content


def test_skills_index_enters_system_prompt(workspace, tmp_path):
    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: demo\n---\nbody")

    seen = {}

    def capture(messages):
        seen["system"] = messages[0].content
        return say("done")

    from agent_harness.skills import discover_skills

    agent, _, _ = make_agent([capture], workspace, tmp_path, skills=discover_skills([tmp_path / "skills"]))
    agent.run("task")
    assert "demo-skill: demo" in seen["system"]
