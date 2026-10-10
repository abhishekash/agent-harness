"""Small provider reliability wrapper.

Retries are deliberately conservative: configuration, validation, and model
errors are surfaced immediately; only transport, timeout, rate-limit, and
server failures are retried.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Callable, Sequence

from agent_harness.providers.base import ProviderError
from agent_harness.types import AssistantMessage, CancellationToken, Message


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("retry delays must be non-negative")
        if self.jitter < 0:
            raise ValueError("jitter must be non-negative")


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, SystemExit, ProviderError)):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    return any(
        marker in name
        for marker in ("timeout", "ratelimit", "rate_limit", "connection", "serviceunavailable", "internalserver")
    )


class ResilientProvider:
    """Retry a provider's transient failures without changing its model."""

    def __init__(
        self,
        provider: Any,
        policy: RetryPolicy | None = None,
        cancellation: CancellationToken | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ):
        self.provider = provider
        self.model = getattr(provider, "model", "unknown")
        self.policy = policy or RetryPolicy()
        self.cancellation = cancellation
        self._sleep = sleep
        self._random = random_fn

    def complete(self, messages: Sequence[Message], tools) -> AssistantMessage:
        return self._call(lambda: self.provider.complete(messages, tools))

    def summarize(self, messages: Sequence[Message]) -> AssistantMessage:
        summarize = getattr(self.provider, "summarize", None)
        if summarize is None:
            return self._call(lambda: self.provider.complete(messages, []))
        return self._call(lambda: summarize(messages))

    def _call(self, fn: Callable[[], AssistantMessage]) -> AssistantMessage:
        last: BaseException | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            if self.cancellation and self.cancellation.cancelled:
                raise ProviderError(self.cancellation.reason)
            try:
                return fn()
            except Exception as exc:
                last = exc
                if attempt >= self.policy.max_attempts or not is_retryable(exc):
                    raise
                delay = min(
                    self.policy.max_delay_s,
                    self.policy.base_delay_s * (2 ** (attempt - 1)),
                )
                if self.policy.jitter:
                    delay += delay * self.policy.jitter * self._random()
                self._sleep_interruptibly(delay)
        assert last is not None
        raise last

    def _sleep_interruptibly(self, delay: float) -> None:
        if not self.cancellation:
            self._sleep(delay)
            return
        # Short slices make Ctrl-C/cancellation responsive without requiring a
        # second thread or a provider-specific async API.
        remaining = delay
        while remaining > 0 and not self.cancellation.cancelled:
            interval = min(0.1, remaining)
            self._sleep(interval)
            remaining -= interval
