"""OpenTelemetry → JSONL.

One JSON object per span, one line per span. This file is the shared contract
of the whole ecosystem: `mcp-trace` reads it, `agent-evals` scores from it,
and `harness trace render` turns it back into a timeline.

Format (all ints are unix nanoseconds):
{
  "trace_id": "...", "span_id": "...", "parent_span_id": "..." | null,
  "name": "tool.call", "start_unix_nano": ..., "end_unix_nano": ...,
  "status": "OK" | "ERROR" | "UNSET",
  "attributes": {...},
  "events": [{"name": "hitl.decision", "time_unix_nano": ..., "attributes": {...}}]
}
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    parent_id = f"{span.parent.span_id:016x}" if span.parent else None
    return {
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_span_id": parent_id,
        "name": span.name,
        "start_unix_nano": span.start_time or 0,
        "end_unix_nano": span.end_time or 0,
        "status": span.status.status_code.name,
        "attributes": {k: _json_safe(v) for k, v in (span.attributes or {}).items()},
        "events": [
            {
                "name": e.name,
                "time_unix_nano": e.timestamp,
                "attributes": {k: _json_safe(v) for k, v in (e.attributes or {}).items()},
            }
            for e in span.events
        ],
    }


class JsonlSpanExporter(SpanExporter):
    """Appends finished spans to a JSONL file (fsync-free, but flushed per export)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                for span in spans:
                    f.write(json.dumps(span_to_dict(span), ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.chmod(self.path, 0o600)
            return SpanExportResult.SUCCESS
        except OSError:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:  # nothing buffered
        pass


def init_tracing(path: str | Path, service_name: str = "agent-harness") -> TracerProvider:
    """Create (but do not globally register) a provider writing spans to ``path``."""
    from opentelemetry.sdk.resources import Resource

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    return provider


def get_tracer(provider: TracerProvider | None, name: str = "agent_harness") -> trace.Tracer:
    if provider is None:
        return trace.NoOpTracerProvider().get_tracer(name)
    return provider.get_tracer(name)
