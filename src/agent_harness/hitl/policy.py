"""HITL core types + policy: which tool calls require a human decision.

Every approval decision is recorded (and emitted as a trace event by the
agent loop) so a run is fully auditable after the fact — including *why*
a human approved, denied, or edited a call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_harness.tools.base import Tool
from agent_harness.types import Risk


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    EDIT = "edit"  # approve, but with human-modified arguments


@dataclass(frozen=True)
class ApprovalRequest:
    tool_name: str
    risk: Risk
    arguments: dict[str, Any]
    summary: str  # one-line human-readable description of what the call does


@dataclass
class ApprovalRecord:
    request: ApprovalRequest
    decision: Decision
    approver: str  # e.g. "cli", "auto", "callback"
    rationale: str = ""
    edited_arguments: dict[str, Any] | None = None
    decided_at: float = field(default_factory=time.time)

    def trace_event_attributes(self) -> dict[str, Any]:
        attrs = {
            "hitl.decision": self.decision.value,
            "hitl.approver": self.approver,
            "hitl.rationale": self.rationale,
        }
        if self.edited_arguments is not None:
            attrs["hitl.edited"] = True
        return attrs


class ApprovalPolicy:
    """Maps tools to approval requirements.

    Default: READ runs free, WRITE and EXECUTE need a human.
    ``always_require`` / ``never_require`` override by tool name — e.g.
    never_require={"write_file"} for evals, always_require={"deploy"} in prod.
    """

    def __init__(
        self,
        require_for: frozenset[Risk] = frozenset({Risk.WRITE, Risk.EXECUTE}),
        always_require: frozenset[str] = frozenset(),
        never_require: frozenset[str] = frozenset(),
    ):
        self.require_for = require_for
        self.always_require = always_require
        self.never_require = never_require

    def requires_approval(self, tool: Tool) -> bool:
        if tool.name in self.always_require:
            return True
        if tool.name in self.never_require:
            return False
        return tool.risk in self.require_for

    @staticmethod
    def default() -> "ApprovalPolicy":
        return ApprovalPolicy()

    @staticmethod
    def permissive() -> "ApprovalPolicy":
        """Nothing requires approval. For unattended evals only — say so in the trace."""
        return ApprovalPolicy(require_for=frozenset())

    @staticmethod
    def paranoid() -> "ApprovalPolicy":
        """Everything, including reads, requires approval."""
        return ApprovalPolicy(require_for=frozenset(Risk))
