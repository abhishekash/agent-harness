"""Crash-safe run checkpoints.

A checkpoint is written atomically and with user-only permissions. It is a
resume boundary, not a transaction around arbitrary side effects: an
in-flight tool call is intentionally treated as unsafe to replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from agent_harness.hitl.policy import ApprovalRecord, ApprovalRequest, Decision
from agent_harness.types import Message, Risk, ToolCall, Usage


CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """The run cannot be safely resumed from a checkpoint."""


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "name": message.name,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }


def _message_from_dict(value: dict[str, Any]) -> Message:
    return Message(
        role=value["role"],
        content=value.get("content", ""),
        tool_call_id=value.get("tool_call_id"),
        name=value.get("name"),
        tool_calls=[
            ToolCall(item["id"], item["name"], dict(item.get("arguments", {})))
            for item in value.get("tool_calls", [])
        ],
    )


def _approval_to_dict(record: ApprovalRecord) -> dict[str, Any]:
    return {
        "request": {
            "tool_name": record.request.tool_name,
            "risk": record.request.risk.value,
            "arguments": record.request.arguments,
            "summary": record.request.summary,
        },
        "decision": record.decision.value,
        "approver": record.approver,
        "rationale": record.rationale,
        "edited_arguments": record.edited_arguments,
        "decided_at": record.decided_at,
    }


def _approval_from_dict(value: dict[str, Any]) -> ApprovalRecord:
    request = value["request"]
    return ApprovalRecord(
        request=ApprovalRequest(
            tool_name=request["tool_name"],
            risk=Risk(request["risk"]),
            arguments=dict(request.get("arguments", {})),
            summary=request.get("summary", ""),
        ),
        decision=Decision(value["decision"]),
        approver=value.get("approver", "checkpoint"),
        rationale=value.get("rationale", ""),
        edited_arguments=value.get("edited_arguments"),
        decided_at=float(value.get("decided_at", time.time())),
    )


@dataclass
class CheckpointState:
    task: str
    messages: list[Message]
    next_step: int
    usage: Usage = field(default_factory=Usage)
    approvals: list[ApprovalRecord] = field(default_factory=list)
    pending_tools: list[ToolCall] | None = None
    provider_model: str = "unknown"
    summary: str = ""
    updated_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "task": self.task,
            "messages": [_message_to_dict(message) for message in self.messages],
            "next_step": self.next_step,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "approvals": [_approval_to_dict(record) for record in self.approvals],
            "pending_tools": [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in self.pending_tools
            ]
            if self.pending_tools is not None
            else None,
            "provider_model": self.provider_model,
            "summary": self.summary,
            "updated_unix": self.updated_unix,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CheckpointState":
        if value.get("version") != CHECKPOINT_VERSION:
            raise CheckpointError(
                f"unsupported checkpoint version {value.get('version')!r}; "
                f"expected {CHECKPOINT_VERSION}"
            )
        pending = value.get("pending_tools")
        return cls(
            task=value["task"],
            messages=[_message_from_dict(item) for item in value.get("messages", [])],
            next_step=int(value["next_step"]),
            usage=Usage(
                input_tokens=int(value.get("usage", {}).get("input_tokens", 0)),
                output_tokens=int(value.get("usage", {}).get("output_tokens", 0)),
            ),
            approvals=[_approval_from_dict(item) for item in value.get("approvals", [])],
            pending_tools=(
                [ToolCall(item["id"], item["name"], dict(item.get("arguments", {}))) for item in pending]
                if pending is not None
                else None
            ),
            provider_model=value.get("provider_model", "unknown"),
            summary=value.get("summary", ""),
            updated_unix=float(value.get("updated_unix", time.time())),
        )


class CheckpointStore:
    """Read/write one checkpoint with an atomic replace and mode 0600."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, state: CheckpointState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_unix = time.time()
        payload = json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def load(self) -> CheckpointState:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CheckpointError(f"checkpoint not found: {self.path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"checkpoint is unreadable: {self.path}: {exc}") from exc
        try:
            return CheckpointState.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"checkpoint is invalid: {self.path}: {exc}") from exc

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
