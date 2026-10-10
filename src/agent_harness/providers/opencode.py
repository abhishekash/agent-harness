"""OpenCode Zen provider using its OpenAI-compatible API."""
from __future__ import annotations

import os

from agent_harness.providers.base import ProviderError
from agent_harness.providers.openai import OpenAIProvider

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_MODEL = "big-pickle"


class OpenCodeProvider(OpenAIProvider):
    """Call OpenCode Zen through the standard chat-completions interface."""

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
            model=model,
            base_url=base_url or os.getenv("OPENCODE_BASE_URL", DEFAULT_BASE_URL),
            api_key=key,
            **kwargs,
        )
