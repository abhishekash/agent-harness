"""A deterministic, scriptable provider.

Two uses:
1. Tests and evals: replay exact assistant turns without network or cost.
2. Demos: `harness run --provider scripted` exercises the full loop
   (tools, HITL gates, tracing) offline.

Script items are either AssistantMessage instances or callables that receive
the conversation so far and return an AssistantMessage — useful for asserting
that the loop fed the model the right context.
"""
from __future__ import annotations

from typing import Callable, Sequence, Union

from agent_harness.providers.base import ProviderError
from agent_harness.types import AssistantMessage, Message, ToolCall, Usage

ScriptItem = Union[AssistantMessage, Callable[[Sequence[Message]], AssistantMessage]]


def _estimate_tokens(text: str) -> int:
    # crude but stable: ~4 chars per token
    return max(1, len(text) // 4)


class ScriptedProvider:
    model = "scripted"

    def __init__(self, script: Sequence[ScriptItem]):
        self._script = list(script)
        self.calls: int = 0

    def complete(self, messages: Sequence[Message], tools) -> AssistantMessage:
        self.calls += 1
        if not self._script:
            raise ProviderError(
                f"script exhausted after {self.calls - 1} calls; "
                "the agent asked for one more completion than scripted"
            )
        item = self._script.pop(0)
        msg = item(messages) if callable(item) else item
        if msg.usage.total == 0:
            in_text = "\n".join(m.content for m in messages)
            msg.usage = Usage(
                input_tokens=_estimate_tokens(in_text),
                output_tokens=_estimate_tokens(msg.content) + 8 * len(msg.tool_calls),
            )
        return msg


def scripted_run(*turns: ScriptItem) -> ScriptedProvider:
    """Convenience: ``scripted_run(say("hi"), call_tool(...), say("done"))``."""
    return ScriptedProvider(list(turns))


def say(content: str) -> AssistantMessage:
    return AssistantMessage(content=content)


def call_tool(name: str, arguments: dict, content: str = "") -> AssistantMessage:
    return AssistantMessage(content=content, tool_calls=[ToolCall.new(name, arguments)])
