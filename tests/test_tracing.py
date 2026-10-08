import json

from agent_harness.tracing import group_traces, init_tracing, get_tracer, load_spans, render_file


def _make_trace(path):
    provider = init_tracing(path)
    tr = get_tracer(provider)
    with tr.start_as_current_span("agent.run") as root:
        root.set_attribute("llm.model", "scripted")
        with tr.start_as_current_span("tool.call") as t:
            t.set_attribute("tool.name", "write_file")
            t.set_attribute("tool.risk", "write")
            t.add_event(
                "hitl.decision",
                {"hitl.decision": "deny", "hitl.approver": "cli", "hitl.rationale": "no"},
            )
    provider.force_flush()
    return f"{root.get_span_context().trace_id:032x}"


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    trace_id = _make_trace(path)
    spans = load_spans(path)
    assert len(spans) == 2
    assert {s["trace_id"] for s in spans} == {trace_id}
    child = next(s for s in spans if s["name"] == "tool.call")
    parent = next(s for s in spans if s["name"] == "agent.run")
    assert child["parent_span_id"] == parent["span_id"]


def test_render_marks_human_decisions(tmp_path):
    path = tmp_path / "t.jsonl"
    _make_trace(path)
    out = render_file(path)
    assert "🔐 human: deny by cli — no" in out
    assert "agent.run" in out


def test_render_unknown_trace_id_lists_known(tmp_path):
    path = tmp_path / "t.jsonl"
    _make_trace(path)
    out = render_file(path, trace_id="deadbeef")
    assert "unknown trace" in out


def test_group_traces_orders_by_start(tmp_path):
    path = tmp_path / "t.jsonl"
    _make_trace(path)
    traces = group_traces(load_spans(path))
    spans = next(iter(traces.values()))
    assert spans[0]["name"] == "agent.run"  # parent started first
