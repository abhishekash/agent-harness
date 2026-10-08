"""Run persistence: a JSONL log of completed runs.

Traces answer "what happened inside run X"; the run store answers
"what have I run lately, and what did it cost". Both are queryable via
the sibling project `mcp-trace`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_harness.agent import RunResult


def record_run(path: str | Path, result: RunResult) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trace_id": result.trace_id,
        "task": result.task,
        "answer_preview": result.answer[:300],
        "steps": result.steps,
        "stopped_reason": result.stopped_reason,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        },
        "cost_usd": round(result.cost_usd, 8),
        "approvals": [
            {
                "tool": a.request.tool_name,
                "decision": a.decision.value,
                "approver": a.approver,
                "rationale": a.rationale,
                "edited": a.edited_arguments is not None,
            }
            for a in result.approvals
        ],
        "finished_unix": time.time(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def list_runs(path: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    return runs[-limit:]
