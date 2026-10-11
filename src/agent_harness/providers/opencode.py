"""OpenCode Go provider using its OpenAI-compatible API."""
from __future__ import annotations

import os
from uuid import uuid4

from agent_harness import __version__
from agent_harness.providers.base import ProviderError
from agent_harness.providers.openai import OpenAIProvider

# Go's subscription route is distinct from Zen's pay-as-you-go/free-model
# route. The documented Go URL still contains "zen": https://opencode.ai/docs/go
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "mimo-v2.6-pro"


def _normalize_model(model: str) -> str:
    """Accept OpenCode CLI's ``provider/model`` spelling at the API boundary."""
    for prefix in ("opencode-go/", "opencode/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


class OpenCodeProvider(OpenAIProvider):
    """Call Go's chat-completions models with honest client/session identity.

    Each instance represents one conversation (a fresh instance per eval run).
    Reuse its session ID for model turns, summaries, compaction, and retries.
    Go models requiring Responses or Anthropic Messages are not supported by
    this adapter; choose a chat-completions model from the Go catalog.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        **kwargs,
    ):
        key = api_key or os.getenv("OPENCODE_API_KEY")
        if not key:
            raise ProviderError("missing API key; set OPENCODE_API_KEY or pass api_key")
        self.session_id = session_id or uuid4().hex
        super().__init__(
            model=_normalize_model(model),
            base_url=base_url or os.getenv("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
            api_key=key,
            request_headers={
                "User-Agent": f"agent-harness/{__version__}",
                "x-opencode-session": self.session_id,
            },
            **kwargs,
        )
