"""The agent loop.

    task → [llm → tool* → llm → …] → answer
                    ↑ HITL gate on gated tools, every decision traced

Everything the loop does lands in the trace:
- one root span `agent.run` per task
- one `agent.step` span per model turn
- one `llm.complete` span per provider call (tokens, cost as attributes)
- one `tool.call` span per tool (risk tier, hitl requirement as attributes)
- one `hitl.decision` event per human decision (approve/deny/edit + rationale)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent_harness.hitl import ApprovalPolicy, ApprovalRecord, ApprovalRequest, Approver, AutoApprover, Decision
from agent_harness.providers.base import Provider
from agent_harness.skills import Skill, render_index
from agent_harness.tools.base import Tool, tool_schema
from agent_harness.tracing import get_tracer
from agent_harness.types import Message, Usage

# USD per 1M tokens (input, output). Extend as needed.
COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "scripted": (0.0, 0.0),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are an agent operating inside a workspace via tools.\n"
    "- Prefer the smallest action that makes progress.\n"
    "- Tools with side effects may require human approval; if a call is denied, "
    "respect the stated reason and find another way or stop.\n"
    "- When you are done, answer plainly without calling tools."
)


@dataclass
class RunResult:
    task: str
    answer: str
    steps: int
    usage: Usage
    cost_usd: float
    trace_id: str
    approvals: list[ApprovalRecord] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    stopped_reason: str = "completed"  # completed | max_steps | error


class Agent:
    def __init__(
        self,
        provider: Provider,
        tools: Sequence[Tool],
        *,
        approver: Approver | None = None,
        policy: ApprovalPolicy | None = None,
        skills: Sequence[Skill] = (),
        tracer_provider: Any = None,
        max_steps: int = 12,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        cost_table: dict[str, tuple[float, float]] | None = None,
    ):
        self.provider = provider
        self.tools = {t.name: t for t in tools}
        self.approver = approver or AutoApprover()
        self.policy = policy or ApprovalPolicy.default()
        self.skills = list(skills)
        self.tracer = get_tracer(tracer_provider)
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.cost_table = cost_table or COST_TABLE

    # -- helpers ------------------------------------------------------------
    def _cost_usd(self, usage: Usage) -> float:
        model = getattr(self.provider, "model", "")
        in_rate, out_rate = self.cost_table.get(model, (0.0, 0.0))
        return (usage.input_tokens * in_rate + usage.output_tokens * out_rate) / 1_000_000

    @staticmethod
    def _summarize(tool: Tool, args: dict[str, Any]) -> str:
        if tool.name == "write_file":
            return f"write {len(str(args.get('content', '')))} chars to {args.get('path')}"
        if tool.name == "run_shell":
            return f"run shell: {args.get('command')}"
        return f"{tool.name}({', '.join(args)})"

    # -- the loop -------------------------------------------------------------
    def run(self, task: str) -> RunResult:
        approvals: list[ApprovalRecord] = []
        usage = Usage()
        skill_index = render_index(self.skills)
        system = self.system_prompt + (f"\n\n{skill_index}" if skill_index else "")
        messages: list[Message] = [Message.system(system), Message.user(task)]

        with self.tracer.start_as_current_span("agent.run") as root:
            root.set_attributes(
                {
                    "agent.task": task[:200],
                    "agent.max_steps": self.max_steps,
                    "agent.tools": sorted(self.tools),
                    "agent.skills": [s.name for s in self.skills],
                    "llm.model": getattr(self.provider, "model", "unknown"),
                }
            )
            trace_id = f"{root.get_span_context().trace_id:032x}"
            stopped = "completed"
            answer = ""
            steps = 0

            for steps in range(1, self.max_steps + 1):
                with self.tracer.start_as_current_span("agent.step") as step_span:
                    step_span.set_attribute("step.index", steps)

                    with self.tracer.start_as_current_span("llm.complete") as llm_span:
                        try:
                            msg = self.provider.complete(
                                messages, [tool_schema(t) for t in self.tools.values()]
                            )
                        except Exception as e:  # provider failure: surface, don't crash-loop
                            llm_span.record_exception(e)
                            stopped = "error"
                            answer = f"provider error on step {steps}: {e}"
                            break
                        usage = usage + msg.usage
                        llm_span.set_attributes(
                            {
                                "llm.model": getattr(self.provider, "model", "unknown"),
                                "llm.usage.input_tokens": msg.usage.input_tokens,
                                "llm.usage.output_tokens": msg.usage.output_tokens,
                                "llm.cost_usd": round(self._cost_usd(msg.usage), 8),
                            }
                        )
                    messages.append(Message.assistant(msg))

                    if not msg.tool_calls:
                        answer = msg.content
                        break

                    for call in msg.tool_calls:
                        result = self._run_tool(call, approvals)
                        messages.append(Message.tool_result(call, result))
            else:
                stopped = "max_steps"
                answer = msg.content or f"stopped: reached max_steps={self.max_steps}"

            if stopped == "error":
                root.set_attribute("agent.outcome", "error")
            root.set_attributes(
                {
                    "agent.steps": steps,
                    "agent.stopped_reason": stopped,
                    "agent.usage.input_tokens": usage.input_tokens,
                    "agent.usage.output_tokens": usage.output_tokens,
                    "agent.cost_usd": round(self._cost_usd(usage), 8),
                    "agent.approvals": len(approvals),
                }
            )

        return RunResult(
            task=task,
            answer=answer,
            steps=steps,
            usage=usage,
            cost_usd=self._cost_usd(usage),
            trace_id=trace_id,
            approvals=approvals,
            messages=messages,
            stopped_reason=stopped,
        )

    def _run_tool(self, call, approvals: list[ApprovalRecord]) -> str:
        tool = self.tools.get(call.name)
        with self.tracer.start_as_current_span("tool.call") as span:
            span.set_attributes(
                {
                    "tool.name": call.name,
                    "tool.risk": tool.risk.value if tool else "unknown",
                    "tool.call_id": call.id,
                }
            )
            if tool is None:
                span.set_attribute("tool.error", "unknown_tool")
                return f"error: unknown tool {call.name!r}; available: {sorted(self.tools)}"

            if self.policy.requires_approval(tool):
                span.set_attribute("hitl.required", True)
                request = ApprovalRequest(
                    tool_name=tool.name,
                    risk=tool.risk,
                    arguments=dict(call.arguments),
                    summary=self._summarize(tool, call.arguments),
                )
                record = self.approver.review(request)
                approvals.append(record)
                span.add_event("hitl.decision", record.trace_event_attributes())

                if record.decision is Decision.DENY:
                    span.set_attribute("tool.denied", True)
                    return (
                        f"denied by human ({record.rationale}). "
                        "Do not retry the same call; choose another approach or stop."
                    )
                if record.decision is Decision.EDIT and record.edited_arguments is not None:
                    call = type(call)(id=call.id, name=call.name, arguments=record.edited_arguments)

            started = time.perf_counter()
            try:
                result = tool.run(**call.arguments)
            except TypeError as e:  # bad args from the model
                span.set_attribute("tool.error", "bad_arguments")
                return f"error: bad arguments for {call.name}: {e}"
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("tool.error", type(e).__name__)
                return f"error: {call.name} raised {type(e).__name__}: {e}"
            finally:
                span.set_attribute("tool.duration_ms", (time.perf_counter() - started) * 1000)
            return result
