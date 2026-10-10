"""A small, dependency-free terminal surface for an agent run.

This is intentionally not a dashboard. The terminal shows one durable idea —
what has happened so far — plus a short activity trail. The progress summary
is owned by the model; this module only gives it a calm place to live.
"""
from __future__ import annotations

from collections import deque
import shutil
import sys
from typing import TextIO

from agent_harness.summary import clamp_summary


class MinimalTerminalUI:
    """Render a restrained live run view when attached to an interactive TTY."""

    def __init__(self, stream: TextIO | None = None, *, mode: str = "auto"):
        self.stream = stream or sys.stderr
        if mode not in {"auto", "always", "never"}:
            raise ValueError(f"unknown UI mode: {mode}")
        try:
            is_tty = bool(self.stream.isatty())
        except (AttributeError, OSError):
            is_tty = False
        self.enabled = mode == "always" or (mode == "auto" and is_tty)
        self._alt_screen = False
        self._started = False
        self._closed = False
        self._model = ""
        self._summary = "Starting the task."
        self._status = "waiting"
        self._activity: deque[str] = deque(maxlen=7)

    @property
    def summary(self) -> str:
        return self._summary

    def start(self, task: str, model: str, summary: str = "") -> None:
        if not self.enabled or self._started:
            return
        self._started = True
        self._model = model
        self._summary = clamp_summary(summary or f"Starting: {task}")
        # The alternate screen keeps a live run from polluting the user's
        # shell history. It also makes approval prompts safe to redraw over.
        self._write("\x1b[?1049h\x1b[?25l")
        self._alt_screen = True
        self._render()

    def handle_event(self, name: str, data: dict) -> None:
        if not self.enabled:
            return
        if name == "summary":
            self._summary = clamp_summary(str(data.get("text", "")), width=self._width() - 4)
        elif name == "run_started":
            self._status = "starting"
        elif name == "step_started":
            self._status = f"step {data.get('step', '?')} · thinking"
        elif name == "tool_started":
            action = str(data.get("summary") or data.get("name") or "tool")
            self._status = f"step {data.get('step', '?')} · {data.get('name', 'tool')}"
            self._activity.append(f"→ {action}")
        elif name == "tool_finished":
            tool = str(data.get("name", "tool"))
            decision = data.get("decision")
            if decision == "deny":
                marker = "×"
            elif decision == "edit":
                marker = "↺"
            else:
                marker = "✓" if data.get("ok", True) else "!"
            suffix = f" · {decision}" if decision and decision not in {"approve"} else ""
            self._activity.append(f"{marker} {tool}{suffix}")
            self._status = "working"
        elif name == "summary_error":
            self._activity.append("! progress summary unavailable")
        elif name == "error":
            self._activity.append(f"! {data.get('message', 'run error')}")
            self._status = "error"
        elif name == "run_finished":
            self._status = str(data.get("stopped_reason", "completed"))
        self._render()

    def finish(self, summary: str = "", status: str = "completed") -> None:
        if not self.enabled or not self._started or self._closed:
            return
        if summary:
            self._summary = clamp_summary(summary, width=self._width() - 4)
        self._status = status
        self._render()
        self.close()

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self._alt_screen:
            self._write("\x1b[?25h\x1b[?1049l")
            self._alt_screen = False
        # The live surface is intentionally temporary, but leave a compact
        # final card in normal scrollback so a fast scripted run is still
        # legible after the alternate screen closes.
        width = self._width()
        summary_lines = clamp_summary(self._summary, width=width - 4).splitlines()[:2]
        while len(summary_lines) < 2:
            summary_lines.append("")
        rule = "─" * max(12, min(width - 15, 64))
        self._write("\n  " + rule + " summary\n")
        self._write(f"  {summary_lines[0]}\n  {summary_lines[1]}\n")
        self._write(f"  {self._status}\n")
        for line in self._activity:
            self._write(f"  {line}\n")

    def _width(self) -> int:
        try:
            return max(60, shutil.get_terminal_size((88, 24)).columns)
        except OSError:
            return 88

    def _render(self) -> None:
        if not self.enabled or not self._started or self._closed:
            return
        width = self._width()
        summary_lines = clamp_summary(self._summary, width=width - 4).splitlines()[:2]
        while len(summary_lines) < 2:
            summary_lines.append("")
        rule = "─" * max(12, min(width - 15, 64))
        out = [
            "\x1b[2J\x1b[H",
            f"  \x1b[2m{self._model or 'harness'}\x1b[0m",
            f"  \x1b[2m{rule} summary\x1b[0m",
            f"  {self._accent(summary_lines[0], width - 4)}",
            f"  {self._accent(summary_lines[1], width - 4)}",
            "",
            f"  \x1b[2m{self._status}\x1b[0m",
        ]
        out.extend(f"  {self._dim(line, width - 4)}" for line in self._activity)
        self._write("\n".join(out) + "\n")

    @staticmethod
    def _truncate(value: str, width: int) -> str:
        value = " ".join(value.split())
        if len(value) <= width:
            return value
        return value[: max(1, width - 1)].rstrip() + "…"

    def _accent(self, value: str, width: int) -> str:
        value = self._truncate(value, width)
        return f"\x1b[38;5;252m{value}\x1b[0m"

    def _dim(self, value: str, width: int) -> str:
        value = self._truncate(value, width)
        return f"\x1b[38;5;245m{value}\x1b[0m"

    def _write(self, value: str) -> None:
        try:
            self.stream.write(value)
            self.stream.flush()
        except (BrokenPipeError, OSError):
            self.enabled = False
