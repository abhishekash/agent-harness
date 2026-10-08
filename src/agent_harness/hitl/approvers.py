"""Approvers: who (or what) answers an ApprovalRequest.

- AutoApprover:    approve everything (evals, demos — never in prod)
- DenyAllApprover: deny everything (safety testing: assert the agent degrades gracefully)
- CallbackApprover: wrap any python callable
- CLIApprover:     interactive terminal gate (the default for real runs)
"""
from __future__ import annotations

import json
import sys
from typing import Callable, Protocol

from agent_harness.hitl.policy import ApprovalRecord, ApprovalRequest, Decision


class Approver(Protocol):
    def review(self, request: ApprovalRequest) -> ApprovalRecord: ...


class AutoApprover:
    def __init__(self, rationale: str = "auto-approved (unattended run)"):
        self.rationale = rationale

    def review(self, request: ApprovalRequest) -> ApprovalRecord:
        return ApprovalRecord(request, Decision.APPROVE, approver="auto", rationale=self.rationale)


class DenyAllApprover:
    def __init__(self, rationale: str = "denied by policy (deny-all approver)"):
        self.rationale = rationale

    def review(self, request: ApprovalRequest) -> ApprovalRecord:
        return ApprovalRecord(request, Decision.DENY, approver="deny-all", rationale=self.rationale)


class CallbackApprover:
    """fn(request) -> ApprovalRecord | Decision | (Decision, edited_args | None)."""

    def __init__(self, fn: Callable, name: str = "callback"):
        self.fn = fn
        self.name = name

    def review(self, request: ApprovalRequest) -> ApprovalRecord:
        out = self.fn(request)
        if isinstance(out, ApprovalRecord):
            return out
        if isinstance(out, Decision):
            return ApprovalRecord(request, out, approver=self.name)
        decision, edited = out
        return ApprovalRecord(
            request, decision, approver=self.name, edited_arguments=edited,
            rationale="edited arguments" if edited is not None else "",
        )


class CLIApprover:
    """Interactive gate: renders the call, waits for y / n / e(dit JSON) / a(lways this tool).

    'always' stashes the tool name into the policy's never_require set for the
    rest of the run — the human stays in control of how much control they want.
    """

    def __init__(self, policy=None, input_fn: Callable[[str], str] = input):
        self._always: set[str] = set()
        self._input = input_fn
        self._policy = policy

    def review(self, request: ApprovalRequest) -> ApprovalRecord:
        if request.tool_name in self._always:
            return ApprovalRecord(
                request, Decision.APPROVE, approver="cli",
                rationale=f"pre-approved for this session ('always {request.tool_name}')",
            )
        print("\n" + "─" * 64, file=sys.stderr)
        print(f"⚠  approval required — {request.tool_name} [{request.risk.value}]", file=sys.stderr)
        print(f"   {request.summary}", file=sys.stderr)
        print(json.dumps(request.arguments, indent=2, default=str), file=sys.stderr)
        while True:
            answer = self._input("[y]es / [n]o / [e]dit args / [a]lways this tool: ").strip().lower()
            if answer in ("y", "yes"):
                return ApprovalRecord(request, Decision.APPROVE, approver="cli")
            if answer in ("n", "no", ""):
                reason = self._input("reason (fed back to the agent): ").strip()
                return ApprovalRecord(
                    request, Decision.DENY, approver="cli",
                    rationale=reason or "denied by human",
                )
            if answer in ("a", "always"):
                self._always.add(request.tool_name)
                return ApprovalRecord(
                    request, Decision.APPROVE, approver="cli",
                    rationale=f"approved; {request.tool_name} pre-approved for this session",
                )
            if answer in ("e", "edit"):
                raw = self._input("replacement arguments as JSON: ").strip()
                try:
                    edited = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"   invalid JSON ({exc}); try again", file=sys.stderr)
                    continue
                return ApprovalRecord(
                    request, Decision.EDIT, approver="cli",
                    edited_arguments=edited, rationale="human edited arguments",
                )
            print("   please answer y, n, e, or a", file=sys.stderr)
