"""Builtin tools: filesystem (sandboxed to a workspace root) + shell (allowlisted).

Design notes:
- File tools resolve every path against a workspace root and refuse escapes.
- Shell runs an allowlist of read-mostly commands with a hard timeout.
- Output is truncated so a single tool call can't blow up the context window.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any

from agent_harness.types import Risk

MAX_OUTPUT_CHARS = 8000


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = MAX_OUTPUT_CHARS // 2
    tail = MAX_OUTPUT_CHARS - head
    return f"{text[:head]}\n…[{len(text) - MAX_OUTPUT_CHARS} chars truncated]…\n{text[-tail:]}"


class _Rooted:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"path escapes workspace root: {path!r}")
        return p

    def _resolve_or_error(self, path: str) -> Path | str:
        """Resolve, returning an agent-readable error string instead of raising."""
        try:
            return self._resolve(path)
        except PermissionError as e:
            return f"error: {e}"


class ReadFile(_Rooted):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace. Returns its contents (truncated if large)."
    risk = Risk.READ
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to the workspace root"}},
        "required": ["path"],
    }

    def run(self, path: str, **_: Any) -> str:
        p = self._resolve_or_error(path)
        if isinstance(p, str):
            return p
        if not p.is_file():
            return f"error: no such file: {path}"
        try:
            return _truncate(p.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            return f"error: {e}"


class ListDir(_Rooted):
    name = "list_dir"
    description = "List files under a workspace directory (recursive, relative paths, capped at 500 entries)."
    risk = Risk.READ
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory relative to the workspace root", "default": "."}},
    }

    def run(self, path: str = ".", **_: Any) -> str:
        base = self._resolve_or_error(path)
        if isinstance(base, str):
            return base
        if not base.is_dir():
            return f"error: no such directory: {path}"
        entries: list[str] = []
        for p in sorted(base.rglob("*")):
            if len(entries) >= 500:
                entries.append("…[capped at 500 entries]")
                break
            if any(part.startswith(".") and part not in (".",) for part in p.relative_to(base).parts):
                continue  # skip dotfiles/dirs like .git
            rel = p.relative_to(self.root)
            entries.append(f"{rel}/" if p.is_dir() else str(rel))
        return "\n".join(entries) or "(empty)"


class WriteFile(_Rooted):
    name = "write_file"
    description = "Write text to a workspace file, creating parent directories. Overwrites existing content."
    risk = Risk.WRITE
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str, **_: Any) -> str:
        p = self._resolve_or_error(path)
        if isinstance(p, str):
            return p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {p.relative_to(self.root)}"


class RunShell(_Rooted):
    """Run an allowlisted, non-shell command with workspace path checks."""

    ALLOWLIST = frozenset(
        {"ls", "cat", "echo", "grep", "find", "head", "tail", "wc", "sort", "uniq", "pwd", "tree", "diff"}
    )
    BLOCKED_FIND_OPERATORS = frozenset({"-exec", "-execdir", "-delete", "-ok", "-okdir"})

    name = "run_shell"
    description = (
        "Run a shell command inside the workspace. Allowlisted binaries only "
        f"({', '.join(sorted(ALLOWLIST))}); 15s timeout; output truncated."
    )
    risk = Risk.EXECUTE
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Command line, e.g. 'grep -rn TODO src'"}},
        "required": ["command"],
    }

    def run(self, command: str, **_: Any) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"error: could not parse command: {e}"
        if not argv:
            return "error: empty command"
        if argv[0] not in self.ALLOWLIST:
            return f"error: {argv[0]!r} is not allowlisted ({sorted(self.ALLOWLIST)})"
        if any(token in self.BLOCKED_FIND_OPERATORS for token in argv):
            return "error: recursive find execution/deletion is not allowed"
        for token in argv[1:]:
            if token.startswith("/") or token == ".." or token.startswith("../") or "/../" in token:
                candidate = Path(token) if token.startswith("/") else self.root / token
                try:
                    candidate.resolve().relative_to(self.root)
                except ValueError:
                    return f"error: shell path escapes workspace root: {token!r}"
        try:
            proc = subprocess.run(
                argv,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "error: command timed out after 15s"
        except OSError as e:
            return f"error: could not run command: {e}"
        out = (proc.stdout + proc.stderr).strip()
        suffix = f"\n[exit code {proc.returncode}]" if proc.returncode else ""
        return _truncate(out) + suffix if out else f"(no output){suffix}"


def default_tools(root: str | Path) -> list:
    return [ReadFile(root), ListDir(root), WriteFile(root), RunShell(root)]
