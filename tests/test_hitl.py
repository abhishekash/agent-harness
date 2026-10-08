from agent_harness.hitl import (
    ApprovalPolicy,
    ApprovalRequest,
    AutoApprover,
    CallbackApprover,
    CLIApprover,
    Decision,
    DenyAllApprover,
)
from agent_harness.tools import ReadFile, WriteFile
from agent_harness.types import Risk


def req(name="write_file", risk=Risk.WRITE, args=None):
    return ApprovalRequest(name, risk, args or {"path": "x"}, "summary")


class TestPolicy:
    def test_default_gates_write_and_execute(self, workspace):
        policy = ApprovalPolicy.default()
        assert policy.requires_approval(WriteFile(workspace)) is True
        assert policy.requires_approval(ReadFile(workspace)) is False

    def test_permissive_gates_nothing(self, workspace):
        assert ApprovalPolicy.permissive().requires_approval(WriteFile(workspace)) is False

    def test_paranoid_gates_reads(self, workspace):
        assert ApprovalPolicy.paranoid().requires_approval(ReadFile(workspace)) is True

    def test_name_overrides(self, workspace):
        policy = ApprovalPolicy(never_require=frozenset({"write_file"}))
        assert policy.requires_approval(WriteFile(workspace)) is False


class TestApprovers:
    def test_auto_approves(self):
        assert AutoApprover().review(req()).decision is Decision.APPROVE

    def test_deny_all(self):
        rec = DenyAllApprover().review(req())
        assert rec.decision is Decision.DENY and rec.rationale

    def test_callback_edit(self):
        cb = CallbackApprover(lambda r: (Decision.EDIT, {"path": "safe.txt"}))
        rec = cb.review(req())
        assert rec.decision is Decision.EDIT
        assert rec.edited_arguments == {"path": "safe.txt"}
        assert rec.trace_event_attributes()["hitl.edited"] is True

    def test_cli_deny_with_reason(self):
        answers = iter(["n", "too risky"])
        cli = CLIApprover(input_fn=lambda prompt: next(answers))
        rec = cli.review(req())
        assert rec.decision is Decision.DENY
        assert rec.rationale == "too risky"

    def test_cli_edit_reprompts_on_bad_json(self):
        answers = iter(["e", "{not json", "e", '{"path": "fixed.txt"}'])
        cli = CLIApprover(input_fn=lambda prompt: next(answers))
        rec = cli.review(req())
        assert rec.decision is Decision.EDIT
        assert rec.edited_arguments == {"path": "fixed.txt"}

    def test_cli_always_stickiness(self):
        answers = iter(["a"])
        cli = CLIApprover(input_fn=lambda prompt: next(answers))
        cli.review(req())
        rec = cli.review(req())  # no input consumed the second time
        assert rec.decision is Decision.APPROVE
        assert "pre-approved" in rec.rationale
