"""Rolling progress summaries for the interactive harness UI.

The summary is deliberately a separate completion from the agent's main
conversation. It uses the same provider instance (and therefore the same
model), but its prompt is tiny and it never changes the agent's context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import textwrap
from typing import Any

from agent_harness.types import AssistantMessage, Message, Usage


SUMMARY_SYSTEM_PROMPT = """You maintain a progress summary for an AI agent run.
Rewrite the summary using only the task, the previous summary, and the new fact.
Capture what is complete and the current state, blocker, or next step.
Return plain text with at most two short lines. No heading, bullets, markdown,
speculation, or preamble. Never claim an action that is not in the new facts."""


@dataclass(frozen=True)
class SummaryUpdate:
    text: str
    usage: Usage = field(default_factory=Usage)


def _fit_line(line: str, width: int) -> str:
    line = " ".join(line.split())
    if len(line) <= width:
        return line
    if width <= 1:
        return line[:width]
    return line[: width - 1].rstrip() + "…"


def clamp_summary(text: str, *, width: int = 96, max_lines: int = 2) -> str:
    """Normalize model output to a small, terminal-safe number of lines."""
    width = max(20, width)
    max_lines = max(1, max_lines)
    cleaned: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("summary:"):
            line = line.split(":", 1)[1].strip()
        if line.startswith(("- ", "* ")):
            line = line[2:].strip()
        if line:
            cleaned.append(line)

    if not cleaned:
        return "No progress summary yet."

    wrapped: list[str] = []
    for line in cleaned:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [line]
        )

    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        # Make it clear that the second line is a bounded view rather than a
        # fabricated conclusion.
        wrapped[-1] = _fit_line(wrapped[-1].rstrip("…") + " …", width)
    return "\n".join(_fit_line(line, width) for line in wrapped)


@dataclass
class RollingSummary:
    """Keep a two-line summary by asking the current provider to rewrite it."""

    provider: Any
    max_width: int = 96
    text: str = "Starting the task."

    def begin(self, task: str) -> str:
        self.text = clamp_summary(f"Starting: {task}", width=self.max_width)
        return self.text

    def update(self, task: str, fact: str) -> SummaryUpdate:
        prompt = (
            f"Task:\n{task[:600]}\n\n"
            f"Previous summary:\n{self.text}\n\n"
            f"New fact:\n{fact[:600]}\n\n"
            "Rewrite the progress summary now."
        )
        messages = [
            Message.system(SUMMARY_SYSTEM_PROMPT),
            Message.user(prompt),
        ]
        summarize = getattr(self.provider, "summarize", None)
        result = summarize(messages) if summarize else self.provider.complete(messages, [])
        if isinstance(result, str):
            result = AssistantMessage(content=result)
        text = clamp_summary(result.content, width=self.max_width)
        if text:
            self.text = text
        return SummaryUpdate(self.text, result.usage)
