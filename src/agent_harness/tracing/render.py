"""Render JSONL traces back into a human timeline.

Pure functions over parsed span dicts — no OTel dependency here, so this
module is also reusable by `mcp-trace` and `agent-evals` via file parsing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_spans(path: str | Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def group_traces(spans: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    traces: dict[str, list[dict[str, Any]]] = {}
    for s in spans:
        traces.setdefault(s["trace_id"], []).append(s)
    for tid in traces:
        traces[tid].sort(key=lambda s: s["start_unix_nano"])
    return traces


def _duration_ms(span: dict[str, Any]) -> float:
    return (span["end_unix_nano"] - span["start_unix_nano"]) / 1e6


def _tree(spans: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for s in spans:
        by_parent.setdefault(s["parent_span_id"], []).append(s)
    for children in by_parent.values():
        children.sort(key=lambda s: s["start_unix_nano"])
    roots = by_parent.get(None, []) or [
        s for s in spans if s["parent_span_id"] not in {x["span_id"] for x in spans}
    ]
    out: list[tuple[dict[str, Any], int]] = []

    def walk(span: dict[str, Any], depth: int) -> None:
        out.append((span, depth))
        for child in by_parent.get(span["span_id"], []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return out


def render_trace(spans: list[dict[str, Any]]) -> str:
    """One trace → aligned text timeline. Human decisions are called out."""
    lines: list[str] = []
    total_ms = 0.0
    for span, depth in _tree(spans):
        dur = _duration_ms(span)
        if depth == 0:
            total_ms = dur
        indent = "  " * depth
        status = "" if span["status"] in ("UNSET", "OK") else f" ✗{span['status'].lower()}"
        extra = ""
        attrs = span["attributes"]
        if "llm.model" in attrs:
            tok_in, tok_out = attrs.get("llm.usage.input_tokens", "?"), attrs.get("llm.usage.output_tokens", "?")
            kind = attrs.get("llm.kind")
            label = f" · {kind}" if kind else ""
            extra = f"  [{attrs['llm.model']}{label} · {tok_in}→{tok_out} tok]"
        if "tool.name" in attrs:
            extra = f"  [{attrs['tool.name']} · risk={attrs.get('tool.risk', '?')}]"
        lines.append(f"{indent}{span['name']}  {dur:8.1f}ms{status}{extra}")
        for event in span["events"]:
            eindent = indent + "  "
            if event["name"] == "hitl.decision":
                a = event["attributes"]
                edited = " (edited args)" if a.get("hitl.edited") else ""
                lines.append(
                    f"{eindent}🔐 human: {a.get('hitl.decision')} by {a.get('hitl.approver')}{edited}"
                    + (f" — {a.get('hitl.rationale')}" if a.get("hitl.rationale") else "")
                )
            else:
                lines.append(f"{eindent}• {event['name']}")
    lines.append(f"\ntotal: {total_ms:.1f}ms across {len(spans)} spans")
    return "\n".join(lines)


def render_file(path: str | Path, trace_id: str | None = None) -> str:
    traces = group_traces(load_spans(path))
    if not traces:
        return "(no spans)"
    if trace_id is None:
        # latest trace by most recent span end
        trace_id = max(traces, key=lambda t: max(s["end_unix_nano"] for s in traces[t]))
    if trace_id not in traces:
        known = ", ".join(t[:8] for t in traces)
        return f"error: unknown trace {trace_id!r}; known: {known}"
    header = f"trace {trace_id}"
    return header + "\n" + "=" * len(header) + "\n" + render_trace(traces[trace_id])
