"""OpenAI-compatible chat-completions provider.

This adapter is intentionally endpoint-neutral. It supports OpenAI itself and
OpenAI-compatible services such as OpenCode Zen by changing ``base_url`` and
``api_key``. The harness loop remains provider-neutral; only message and tool
schema translation lives here.
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

from agent_harness.providers.base import ProviderError
from agent_harness.types import AssistantMessage, Message, ToolCall, Usage

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


def _to_api_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role in ("system", "user"):
            out.append({"role": message.role, "content": message.content})
        elif message.role == "assistant":
            item: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or None,
            }
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, separators=(",", ":")),
                        },
                    }
                    for call in message.tool_calls
                ]
            out.append(item)
        elif message.role == "tool":
            if not message.tool_call_id:
                raise ProviderError("tool message is missing tool_call_id")
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
    return out


def _to_api_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object"}),
            },
        }
        for tool in tools
    ]


def _response_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class OpenAIProvider:
    """Provider for OpenAI-compatible ``/chat/completions`` endpoints."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ):
        if client is None:
            if OpenAI is None:  # pragma: no cover
                raise ProviderError(
                    "openai package not installed; run `pip install agent-harness[opencode]`"
                )
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ProviderError("missing API key; set OPENAI_API_KEY or pass api_key")
            client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
        self._client = client

    def _complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": _to_api_messages(messages),
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            request["tools"] = _to_api_tools(tools)
        if self.temperature is not None:
            request["temperature"] = self.temperature

        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc

        choices = _response_value(response, "choices", []) or []
        if not choices:
            raise ProviderError("OpenAI-compatible response contained no choices")
        message = _response_value(choices[0], "message")
        if message is None:
            raise ProviderError("OpenAI-compatible response choice contained no message")

        content = _response_value(message, "content", "") or ""
        calls: list[ToolCall] = []
        for raw_call in _response_value(message, "tool_calls", []) or []:
            function = _response_value(raw_call, "function")
            if function is None:
                raise ProviderError("OpenAI-compatible tool call contained no function")
            raw_arguments = _response_value(function, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    f"model returned invalid JSON arguments for tool "
                    f"{_response_value(function, 'name', '<unknown>')!r}"
                ) from exc
            if not isinstance(arguments, dict):
                raise ProviderError("model returned non-object tool arguments")
            call_id = _response_value(raw_call, "id")
            name = _response_value(function, "name")
            if not call_id or not name:
                raise ProviderError("OpenAI-compatible tool call is missing id or name")
            calls.append(ToolCall(id=str(call_id), name=str(name), arguments=arguments))

        raw_usage = _response_value(response, "usage")
        return AssistantMessage(
            content=str(content),
            tool_calls=calls,
            usage=Usage(
                input_tokens=int(_response_value(raw_usage, "prompt_tokens", 0) or 0),
                output_tokens=int(_response_value(raw_usage, "completion_tokens", 0) or 0),
            ),
        )

    def complete(self, messages: Sequence[Message], tools) -> AssistantMessage:
        return self._complete(messages, tools)

    def summarize(self, messages: Sequence[Message]) -> AssistantMessage:
        return self._complete(messages, [], max_tokens=min(self.max_tokens, 160))
