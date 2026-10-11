"""OpenCode Go provider using its OpenAI-compatible API."""
from __future__ import annotations

import os

from agent_harness.providers.base import ProviderError
from agent_harness.providers.openai import OpenAIProvider

# OpenCode Go is the subscription/API route. The older Zen endpoint has a
# separate free-tier policy and rejects external clients with HTTP 403.
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "mimo-v2.6-pro"


def _normalize_model(model: str) -> str:
    """Accept OpenCode CLI's ``provider/model`` spelling at the API boundary."""
    for prefix in ("opencode-go/", "opencode/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


class OpenCodeProvider(OpenAIProvider):
    """Call OpenCode Go through the standard chat-completions interface."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        key = api_key or os.getenv("OPENCODE_API_KEY")
        if not key:
            raise ProviderError("missing API key; set OPENCODE_API_KEY or pass api_key")
        super().__init__(
            model=_normalize_model(model),
            base_url=base_url or os.getenv("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
            api_key=key,
            **kwargs,
        )
