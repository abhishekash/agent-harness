"""Anthropic provider (optional: pip install agent-harness[anthropic]).

Kept deliberately thin: the interesting logic (loop, HITL, tracing) lives
provider-side-neutral in the harness. This file only translates types.
"""
from __future__ import annotations

from typing import Any, Sequence

from agent_harness.providers.base import ProviderError
from agent_harness.types import AssistantMessage, Message, ToolCall, Usage

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _split_system(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    return system, [m for m in messages if m.role != "system"]


def _to_api_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            out.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
        elif m.role in ("user",):
            out.append({"role": "user", "content": m.content})
    return out


def _to_api_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in tools
    ]


class AnthropicProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        temperature: float | None = None,
        **client_kwargs,
    ):
        if Anthropic is None:  # pragma: no cover
            raise ProviderError(
                "anthropic package not installed; run `pip install agent-harness[anthropic]`"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = Anthropic(**client_kwargs)

    def _complete(
        self,
        messages: Sequence[Message],
        tools,
        *,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        system, rest = _split_system(messages)
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = _to_api_tools(tools)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": _to_api_messages(rest),
            **kwargs,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        resp = self._client.messages.create(**request)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))
        return AssistantMessage(
            content="\n".join(text_parts),
            tool_calls=calls,
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
        )

    def complete(self, messages: Sequence[Message], tools) -> AssistantMessage:
        return self._complete(messages, tools)

    def summarize(self, messages: Sequence[Message]) -> AssistantMessage:
        """Rewrite a progress summary with this provider's configured model."""
        return self._complete(messages, [], max_tokens=min(self.max_tokens, 160))
