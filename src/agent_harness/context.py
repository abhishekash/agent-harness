"""Bounded conversation context with explicit model-written compaction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from agent_harness.types import Message, Usage


CONTEXT_SYSTEM_PROMPT = """You are compacting an agent conversation for continued execution.
Preserve concrete facts needed to finish the task: files inspected or changed,
commands run, tool errors, approvals/denials, constraints, and unfinished work.
Do not invent facts. Return a concise factual context handoff, not commentary."""


class ContextError(RuntimeError):
    """The conversation could not be compacted safely."""


@dataclass(frozen=True)
class ContextPolicy:
    max_chars: int = 80_000
    keep_recent_messages: int = 8
    max_summary_input_chars: int = 24_000

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be positive")
        if self.keep_recent_messages < 2:
            raise ValueError("keep_recent_messages must be at least 2")
        if self.max_summary_input_chars < 1:
            raise ValueError("max_summary_input_chars must be positive")


@dataclass(frozen=True)
class ContextResult:
    messages: list[Message]
    usage: Usage
    compacted: bool
    dropped_messages: int = 0


def _message_size(message: Message) -> int:
    tool_size = sum(len(call.name) + len(str(call.arguments)) for call in message.tool_calls)
    return len(message.role) + len(message.content) + tool_size + 32


def _render_message(message: Message) -> str:
    tool_calls = ""
    if message.tool_calls:
        tool_calls = " " + ", ".join(
            f"{call.name}({call.arguments})" for call in message.tool_calls
        )
    return f"[{message.role}]{tool_calls}\n{message.content}\n"


class ContextManager:
    """Compact only at message boundaries, never in the middle of a tool turn."""

    def __init__(self, provider: Any, policy: ContextPolicy | None = None):
        self.provider = provider
        self.policy = policy or ContextPolicy()
        self.compactions = 0

    def maybe_compact(
        self,
        messages: Sequence[Message],
        completion: Callable[[Sequence[Message]], Any] | None = None,
    ) -> ContextResult:
        if sum(_message_size(message) for message in messages) <= self.policy.max_chars:
            return ContextResult(list(messages), Usage(), False)

        system = next((message for message in messages if message.role == "system"), None)
        task = next((message for message in messages if message.role == "user"), None)
        if system is None or task is None:
            raise ContextError("cannot compact a conversation without system and task messages")

        body_start = max(i for i, message in enumerate(messages) if message is task) + 1
        body = list(messages[body_start:])
        if len(body) <= self.policy.keep_recent_messages:
            raise ContextError(
                "context budget is smaller than the current tool turn; "
                "increase the context budget"
            )

        split = self._safe_split(body)
        if split <= 0 or split >= len(body):
            raise ContextError("no safe message boundary available for context compaction")
        dropped = body[:split]
        recent = body[split:]
        source = "".join(_render_message(message) for message in dropped)
        source = source[: self.policy.max_summary_input_chars]
        prompt = (
            "Summarize the earlier part of this agent run for the next model turn.\n\n"
            f"Earlier messages:\n{source}\n\n"
            "Include unfinished work and constraints."
        )
        try:
            summary_messages = [Message.system(CONTEXT_SYSTEM_PROMPT), Message.user(prompt)]
            if completion is not None:
                response = completion(summary_messages)
            else:
                summarize = getattr(self.provider, "summarize", None)
                response = (
                    summarize(summary_messages)
                    if summarize
                    else self.provider.complete(summary_messages, [])
                )
        except Exception as exc:
            raise ContextError(f"context compaction failed: {exc}") from exc

        handoff = response.content.strip()
        if not handoff:
            raise ContextError("context compaction returned an empty handoff")
        compacted_system = Message.system(
            system.content
            + "\n\n[Earlier conversation compacted; treat this as verified run context]\n"
            + handoff[: self.policy.max_summary_input_chars]
        )
        self.compactions += 1
        return ContextResult(
            [compacted_system, task, *recent],
            response.usage,
            True,
            dropped_messages=len(dropped),
        )

    def _safe_split(self, body: list[Message]) -> int:
        candidate = max(1, len(body) - self.policy.keep_recent_messages)
        # An assistant message starts a provider turn. Keeping from that point
        # preserves the assistant/tool-result pairing required by chat APIs.
        for index in range(candidate, len(body)):
            if body[index].role == "assistant":
                return index
        return candidate
