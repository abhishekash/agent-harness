"""Provider protocol: anything that can complete a conversation."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

from agent_harness.types import AssistantMessage, Message


class Provider(Protocol):
    """An LLM backend.

    Implementations must be stateless across calls: the full conversation is
    passed in every time, and the returned AssistantMessage is appended by the
    agent loop. Keeping providers dumb is what makes the loop testable. A
    provider may also expose ``summarize(messages)`` for the optional live
    progress surface; it must use the same configured model and return an
    ``AssistantMessage`` without tool calls.
    """

    model: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantMessage:
        """Return the next assistant message.

        ``tools`` is a list of JSON-Schema-ish tool descriptions:
        ``{"name", "description", "parameters"}``. Providers that do not
        support tool use should simply return content-only messages.
        """
        ...


class ProviderError(RuntimeError):
    """Raised when a provider fails in a way the loop should surface."""
