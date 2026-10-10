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
from typing import Any, Callable, Sequence

from agent_harness.checkpoint import CheckpointError, CheckpointState, CheckpointStore
from agent_harness.context import ContextError, ContextManager
from agent_harness.hitl import ApprovalPolicy, ApprovalRecord, ApprovalRequest, Approver, AutoApprover, Decision
from agent_harness.providers.base import Provider, ProviderError
from agent_harness.skills import Skill, render_index
from agent_harness.tools.base import Tool, tool_schema, validate_tool_arguments
from agent_harness.tracing import get_tracer
from agent_harness.types import AssistantMessage, CancellationToken, Message, RunLimits, Usage

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
    stopped_reason: str = "completed"  # completed | max_steps | error | cancelled | budget_exceeded | context_error | checkpoint_error
    summary: str = ""


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
        event_handler: Callable[[str, dict[str, Any]], None] | None = None,
        progress_summarizer: Any = None,
        limits: RunLimits | None = None,
        cancellation: CancellationToken | None = None,
        context_manager: ContextManager | None = None,
        checkpoint_store: CheckpointStore | None = None,
        resume: bool = False,
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
        self.event_handler = event_handler
        self.progress_summarizer = progress_summarizer
        self.limits = limits or RunLimits(max_steps=max_steps)
        self.max_steps = self.limits.max_steps
        self.cancellation = cancellation or CancellationToken()
        self.context_manager = context_manager
        self.checkpoint_store = checkpoint_store
        self.resume = resume

    # -- helpers ------------------------------------------------------------
    def _cost_usd(self, usage: Usage) -> float:
        model = getattr(self.provider, "model", "")
        in_rate, out_rate = self.cost_table.get(model, (0.0, 0.0))
        return (usage.input_tokens * in_rate + usage.output_tokens * out_rate) / 1_000_000

    @staticmethod
    def _summarize(tool: Tool, args: Any) -> str:
        if not isinstance(args, dict):
            return f"{tool.name}(invalid arguments)"
        if tool.name == "read_file":
            return f"read {args.get('path')}"
        if tool.name == "list_dir":
            return f"list {args.get('path', '.')}"
        if tool.name == "write_file":
            return f"write {len(str(args.get('content', '')))} chars to {args.get('path')}"
        if tool.name == "run_shell":
            return f"run shell: {args.get('command')}"
        return f"{tool.name}({', '.join(args)})"

    def _emit(self, event: str, **data: Any) -> None:
        """Notify optional UI observers without making them part of the loop."""
        if self.event_handler is None:
            return
        try:
            self.event_handler(event, data)
        except Exception:
            # A terminal renderer must never turn a successful tool run into a
            # failed agent run.
            return

    def _update_summary(self, task: str, fact: str, usage: Usage) -> tuple[Usage, str]:
        """Ask the same provider for a bounded progress rewrite."""
        if self.progress_summarizer is None:
            return usage, ""
        try:
            with self.tracer.start_as_current_span("llm.complete") as summary_span:
                try:
                    update = self.progress_summarizer.update(task, fact)
                    update_usage = getattr(update, "usage", Usage())
                    summary_span.set_attributes(
                        {
                            "llm.model": getattr(self.provider, "model", "unknown"),
                            "llm.kind": "progress_summary",
                            "llm.usage.input_tokens": update_usage.input_tokens,
                            "llm.usage.output_tokens": update_usage.output_tokens,
                            "llm.cost_usd": round(self._cost_usd(update_usage), 8),
                        }
                    )
                except Exception as exc:
                    summary_span.record_exception(exc)
                    raise
            usage = usage + update_usage
            text = str(getattr(update, "text", ""))
            self._emit("summary", text=text)
            return usage, text
        except Exception as exc:
            self._emit("summary_error", error=str(exc))
            return usage, str(getattr(self.progress_summarizer, "text", ""))

    def _limit_reason(self, started: float, usage: Usage) -> str | None:
        if self.cancellation.cancelled:
            return f"cancelled: {self.cancellation.reason}"
        limits = self.limits
        if limits.max_duration_s is not None and time.monotonic() - started >= limits.max_duration_s:
            return f"budget_exceeded: duration limit {limits.max_duration_s:.1f}s"
        if limits.max_input_tokens is not None and usage.input_tokens >= limits.max_input_tokens:
            return f"budget_exceeded: input token limit {limits.max_input_tokens}"
        if limits.max_output_tokens is not None and usage.output_tokens >= limits.max_output_tokens:
            return f"budget_exceeded: output token limit {limits.max_output_tokens}"
        if limits.max_cost_usd is not None and self._cost_usd(usage) >= limits.max_cost_usd:
            return f"budget_exceeded: cost limit ${limits.max_cost_usd:.4f}"
        return None

    def _save_checkpoint(
        self,
        task: str,
        messages: list[Message],
        next_step: int,
        usage: Usage,
        approvals: list[ApprovalRecord],
        pending_tools=None,
        summary: str = "",
    ) -> str | None:
        if self.checkpoint_store is None:
            return None
        try:
            self.checkpoint_store.save(
                CheckpointState(
                    task=task,
                    messages=list(messages),
                    next_step=next_step,
                    usage=usage,
                    approvals=list(approvals),
                    pending_tools=list(pending_tools) if pending_tools is not None else None,
                    provider_model=getattr(self.provider, "model", "unknown"),
                    summary=summary,
                )
            )
            return None
        except Exception as exc:
            message = f"checkpoint write failed: {exc}"
            self._emit("error", message=message)
            return message

    def _context_complete(self, messages: Sequence[Message]):
        with self.tracer.start_as_current_span("llm.complete") as span:
            try:
                summarize = getattr(self.provider, "summarize", None)
                response = summarize(messages) if summarize else self.provider.complete(messages, [])
                span.set_attributes(
                    {
                        "llm.model": getattr(self.provider, "model", "unknown"),
                        "llm.kind": "context_compaction",
                        "llm.usage.input_tokens": response.usage.input_tokens,
                        "llm.usage.output_tokens": response.usage.output_tokens,
                        "llm.cost_usd": round(self._cost_usd(response.usage), 8),
                    }
                )
                return response
            except Exception as exc:
                span.record_exception(exc)
                raise

    # -- the loop -------------------------------------------------------------
    def run(self, task: str) -> RunResult:
        approvals: list[ApprovalRecord] = []
        usage = Usage()
        skill_index = render_index(self.skills)
        system = self.system_prompt + (f"\n\n{skill_index}" if skill_index else "")
        messages: list[Message] = [Message.system(system), Message.user(task)]
        start_step = 1
        resumed = False
        resumed_summary = ""

        if self.resume:
            if self.checkpoint_store is None:
                raise CheckpointError("resume requested without a checkpoint store")
            state = self.checkpoint_store.load()
            if state.pending_tools:
                names = ", ".join(call.name for call in state.pending_tools)
                raise CheckpointError(
                    "checkpoint contains an in-flight tool call "
                    f"({names}); refusing to replay a possible side effect"
                )
            if state.task != task:
                raise CheckpointError("resume task does not match the checkpoint task")
            checkpoint_model = state.provider_model
            current_model = getattr(self.provider, "model", "unknown")
            if current_model == "scripted":
                raise CheckpointError(
                    "the scripted provider cannot safely resume; use a stateful live provider"
                )
            if checkpoint_model not in {"unknown", current_model}:
                raise CheckpointError(
                    f"checkpoint was created with model {checkpoint_model!r}, "
                    f"but this run uses {current_model!r}"
                )
            messages = state.messages
            usage = state.usage
            approvals = state.approvals
            start_step = state.next_step
            resumed_summary = state.summary
            resumed = True

        summary = resumed_summary
        if self.progress_summarizer is not None:
            try:
                if resumed and resumed_summary:
                    self.progress_summarizer.text = resumed_summary
                else:
                    summary = self.progress_summarizer.begin(task)
            except Exception as exc:
                self._emit("summary_error", error=str(exc))

        run_started = time.monotonic()
        initial_checkpoint_error = None
        if not resumed:
            initial_checkpoint_error = self._save_checkpoint(
                task, messages, start_step, usage, approvals, summary=summary
            )

        with self.tracer.start_as_current_span("agent.run") as root:
            root.set_attributes(
                {
                    "agent.task": task[:200],
                    "agent.max_steps": self.max_steps,
                    "agent.tools": sorted(self.tools),
                    "agent.skills": [s.name for s in self.skills],
                    "llm.model": getattr(self.provider, "model", "unknown"),
                    "agent.resumed": resumed,
                }
            )
            trace_id = f"{root.get_span_context().trace_id:032x}"
            stopped = "completed"
            answer = ""
            steps = start_step - 1
            self._emit(
                "run_started",
                task=task,
                model=getattr(self.provider, "model", "unknown"),
                resumed=resumed,
            )

            if initial_checkpoint_error:
                stopped = "checkpoint_error"
                answer = initial_checkpoint_error
            elif start_step > self.max_steps:
                stopped = "max_steps"
                answer = f"stopped: reached max_steps={self.max_steps}"
            else:
                for steps in range(start_step, self.max_steps + 1):
                    reason = self._limit_reason(run_started, usage)
                    if reason:
                        stopped = "cancelled" if reason.startswith("cancelled") else "budget_exceeded"
                        answer = reason
                        break

                    if self.context_manager is not None:
                        try:
                            context = self.context_manager.maybe_compact(
                                messages, completion=self._context_complete
                            )
                        except ContextError as exc:
                            stopped = "context_error"
                            answer = str(exc)
                            self._emit("error", message=answer)
                            break
                        if context.compacted:
                            messages = context.messages
                            usage = usage + context.usage
                            self._emit(
                                "context_compacted",
                                dropped_messages=context.dropped_messages,
                                compactions=self.context_manager.compactions,
                            )

                    checkpoint_error = self._save_checkpoint(
                        task, messages, steps, usage, approvals, summary=summary
                    )
                    if checkpoint_error:
                        stopped = "checkpoint_error"
                        answer = checkpoint_error
                        break

                    self._emit("step_started", step=steps)
                    with self.tracer.start_as_current_span("agent.step") as step_span:
                        step_span.set_attribute("step.index", steps)

                        with self.tracer.start_as_current_span("llm.complete") as llm_span:
                            try:
                                msg = self.provider.complete(
                                    messages, [tool_schema(t) for t in self.tools.values()]
                                )
                                if not isinstance(msg, AssistantMessage):
                                    raise ProviderError(
                                        f"provider returned {type(msg).__name__}, expected AssistantMessage"
                                    )
                            except KeyboardInterrupt:
                                self.cancellation.cancel("interrupted")
                                stopped = "cancelled"
                                answer = "cancelled by user"
                                break
                            except Exception as e:  # provider failure: surface, don't crash-loop
                                llm_span.record_exception(e)
                                stopped = "error"
                                answer = f"provider error on step {steps}: {e}"
                                self._emit("error", message=answer)
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

                        reason = self._limit_reason(run_started, usage)
                        if reason:
                            stopped = "cancelled" if reason.startswith("cancelled") else "budget_exceeded"
                            answer = reason
                            break

                        if not msg.tool_calls:
                            answer = msg.content
                            usage, summary = self._update_summary(
                                task, f"The agent finished with this answer: {answer[:500]}", usage
                            )
                            reason = self._limit_reason(run_started, usage)
                            if reason:
                                stopped = "cancelled" if reason.startswith("cancelled") else "budget_exceeded"
                                answer = reason
                            break

                        pending_error = self._save_checkpoint(
                            task, messages, steps, usage, approvals, msg.tool_calls, summary
                        )
                        if pending_error:
                            stopped = "checkpoint_error"
                            answer = pending_error
                            break

                        for call_index, call in enumerate(msg.tool_calls):
                            if self.cancellation.cancelled:
                                stopped = "cancelled"
                                answer = f"cancelled: {self.cancellation.reason}"
                                break
                            tool = self.tools.get(call.name)
                            action = (
                                self._summarize(tool, call.arguments)
                                if tool is not None
                                else call.name
                            )
                            self._emit(
                                "tool_started",
                                step=steps,
                                name=call.name,
                                summary=action,
                            )
                            approvals_before = len(approvals)
                            started = time.perf_counter()
                            try:
                                result = self._run_tool(call, approvals)
                            except KeyboardInterrupt:
                                self.cancellation.cancel("interrupted")
                                stopped = "cancelled"
                                answer = "cancelled by user"
                                break
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            record = approvals[-1] if len(approvals) > approvals_before else None
                            decision = record.decision.value if record else None
                            ok = not str(result).startswith(("error:", "denied by human"))
                            self._emit(
                                "tool_finished",
                                step=steps,
                                name=call.name,
                                ok=ok,
                                decision=decision,
                                duration_ms=round(elapsed_ms, 1),
                            )
                            messages.append(Message.tool_result(call, result))
                            outcome = "completed" if ok else "did not complete"
                            fact = f"{action} {outcome}."
                            if decision:
                                fact += f" Human decision: {decision}."
                            usage, summary = self._update_summary(task, fact, usage)
                            remaining_tools = msg.tool_calls[call_index + 1 :]
                            checkpoint_error = self._save_checkpoint(
                                task,
                                messages,
                                steps,
                                usage,
                                approvals,
                                remaining_tools or None,
                                summary,
                            )
                            if checkpoint_error:
                                stopped = "checkpoint_error"
                                answer = checkpoint_error
                                break
                            reason = self._limit_reason(run_started, usage)
                            if reason:
                                stopped = "cancelled" if reason.startswith("cancelled") else "budget_exceeded"
                                answer = reason
                                break

                        if stopped != "completed":
                            break
                        checkpoint_error = self._save_checkpoint(
                            task, messages, steps + 1, usage, approvals, summary=summary
                        )
                        if checkpoint_error:
                            stopped = "checkpoint_error"
                            answer = checkpoint_error
                            break
                else:
                    stopped = "max_steps"
                    answer = f"stopped: reached max_steps={self.max_steps}"
                    usage, summary = self._update_summary(
                        task,
                        f"The agent stopped after reaching the {self.max_steps}-step limit.",
                        usage,
                    )

            if stopped == "completed" and self.checkpoint_store is not None:
                self.checkpoint_store.clear()
            if stopped not in {"completed", "max_steps"}:
                root.set_attribute("agent.outcome", stopped)
            root.set_attributes(
                {
                    "agent.steps": steps,
                    "agent.stopped_reason": stopped,
                    "agent.usage.input_tokens": usage.input_tokens,
                    "agent.usage.output_tokens": usage.output_tokens,
                    "agent.cost_usd": round(self._cost_usd(usage), 8),
                    "agent.approvals": len(approvals),
                    "agent.summary": summary[:500],
                }
            )
            self._emit("run_finished", stopped_reason=stopped, summary=summary)

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
            summary=summary,
        )

    @staticmethod
    def _record_tool_result(span, result: str) -> None:
        text = str(result)
        if text.startswith("denied by human"):
            span.set_attribute("tool.result_status", "denied")
            span.set_attribute("tool.error_message", text[:500])
        elif text.startswith("error:"):
            span.set_attribute("tool.result_status", "error")
            span.set_attribute("tool.error_message", text[:500])
        else:
            span.set_attribute("tool.result_status", "ok")

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
                result = f"error: unknown tool {call.name!r}; available: {sorted(self.tools)}"
                span.set_attribute("tool.error", "unknown_tool")
                self._record_tool_result(span, result)
                return result

            validation_error = validate_tool_arguments(tool, call.arguments)
            if validation_error:
                span.set_attribute("tool.error", "bad_arguments")
                self._record_tool_result(span, validation_error)
                return validation_error

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
                    result = (
                        f"denied by human ({record.rationale}). "
                        "Do not retry the same call; choose another approach or stop."
                    )
                    self._record_tool_result(span, result)
                    return result
                if record.decision is Decision.EDIT and record.edited_arguments is not None:
                    edited_error = validate_tool_arguments(tool, record.edited_arguments)
                    if edited_error:
                        span.set_attribute("tool.error", "bad_edited_arguments")
                        self._record_tool_result(span, edited_error)
                        return edited_error
                    call = type(call)(id=call.id, name=call.name, arguments=record.edited_arguments)

            started = time.perf_counter()
            try:
                result = tool.run(**call.arguments)
            except TypeError as e:  # bad args from the model
                span.set_attribute("tool.error", "bad_arguments")
                result = f"error: bad arguments for {call.name}: {e}"
                self._record_tool_result(span, result)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_attribute("tool.error", type(e).__name__)
                result = f"error: {call.name} raised {type(e).__name__}: {e}"
                self._record_tool_result(span, result)
                return result
            finally:
                span.set_attribute("tool.duration_ms", (time.perf_counter() - started) * 1000)
            self._record_tool_result(span, result)
            return result
