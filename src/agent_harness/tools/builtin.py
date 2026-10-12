"""Builtin tools: workspace path checks and allowlisted read-only commands.

Design notes:
- File tools resolve every path against a workspace root and refuse escapes.
- Shell runs an allowlist of read-mostly commands with a hard timeout.
- Output is truncated so a single tool call can't blow up the context window.
"""
from __future__ import annotations

import os
import shlex
import shutil
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
    """Run a small read-only command grammar with workspace path checks."""

    ALLOWLIST = frozenset(
        {"ls", "cat", "echo", "grep", "find", "head", "tail", "wc", "sort", "uniq", "pwd", "tree", "diff"}
    )
    BLOCKED_FIND_OPERATORS = frozenset({"-exec", "-execdir", "-delete", "-ok", "-okdir"})
    SIMPLE_FLAGS = {
        "ls": frozenset("alh1"),
        "wc": frozenset("lwc"),
        "uniq": frozenset("cdui"),
    }
    GREP_FLAGS = frozenset("rnilcvwsFE")  # -R follows symlinks; -r does not.
    FIND_VALUE_PREDICATES = frozenset({"-name", "-iname", "-type", "-maxdepth", "-mindepth"})
    FIND_BARE_PREDICATES = frozenset({"-print"})

    name = "run_shell"
    description = (
        "Run a read-only command inside the workspace. Allowlisted binaries and options only "
        f"({', '.join(sorted(ALLOWLIST))}); 15s timeout; output truncated."
    )
    risk = Risk.EXECUTE
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Command line, e.g. 'grep -rn TODO src'"}},
        "required": ["command"],
    }

    def _check_paths(self, paths: list[str]) -> str | None:
        for path in paths:
            try:
                self._resolve(path)
            except PermissionError:
                return f"error: shell path escapes workspace root: {path!r}"
        return None

    @staticmethod
    def _flag_group(token: str, allowed: frozenset[str]) -> bool:
        return token.startswith("-") and not token.startswith("--") and bool(token[1:]) and all(
            flag in allowed for flag in token[1:]
        )

    def _validate_argv(self, argv: list[str]) -> str | None:
        command, args = argv[0], argv[1:]
        if command == "echo":
            return None  # No shell expansion or output redirection is involved.
        if command == "pwd":
            return "error: pwd does not accept arguments" if args else None

        if command == "find":
            if any(token in self.BLOCKED_FIND_OPERATORS for token in args):
                return "error: recursive find execution/deletion is not allowed"
            paths: list[str] = []
            index = 0
            while index < len(args) and not args[index].startswith("-"):
                paths.append(args[index])
                index += 1
            path_error = self._check_paths(paths or ["."])
            if path_error:
                return path_error
            while index < len(args):
                predicate = args[index]
                if predicate in self.FIND_BARE_PREDICATES:
                    index += 1
                elif predicate in self.FIND_VALUE_PREDICATES and index + 1 < len(args):
                    value = args[index + 1]
                    if predicate in {"-maxdepth", "-mindepth"} and not value.isdecimal():
                        return f"error: invalid find depth: {value!r}"
                    if predicate == "-type" and value not in {"f", "d", "l"}:
                        return f"error: unsupported find type: {value!r}"
                    index += 2
                else:
                    return f"error: unsupported find argument: {predicate!r}"
            return None

        if command == "grep":
            index = 0
            while index < len(args) and self._flag_group(args[index], self.GREP_FLAGS):
                index += 1
            if index >= len(args) or args[index].startswith("-"):
                return "error: grep needs a pattern (only read-only short flags are supported)"
            paths = args[index + 1 :]
            if any(path.startswith("-") for path in paths):
                return "error: unsupported grep argument"
            return self._check_paths(paths)

        if command in {"head", "tail"}:
            if len(args) >= 2 and args[0] in {"-n", "-c"}:
                if not args[1].isdecimal():
                    return f"error: invalid {command} count: {args[1]!r}"
                args = args[2:]
            if any(arg.startswith("-") for arg in args):
                return f"error: unsupported {command} argument"
            return self._check_paths(args)

        if command in self.SIMPLE_FLAGS:
            index = 0
            while index < len(args) and self._flag_group(args[index], self.SIMPLE_FLAGS[command]):
                index += 1
            args = args[index:]
        if any(arg.startswith("-") for arg in args):
            return f"error: unsupported {command} argument"
        if command == "diff" and len(args) != 2:
            return "error: diff requires two paths"
        return self._check_paths(args)

    def run(self, command: str, **_: Any) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"error: could not parse command: {e}"
        if not argv:
            return "error: empty command"
        if argv[0] not in self.ALLOWLIST:
            return f"error: {argv[0]!r} is not allowlisted ({sorted(self.ALLOWLIST)})"
        argument_error = self._validate_argv(argv)
        if argument_error:
            return argument_error
        # Do not resolve binaries through a caller-controlled PATH that might include
        # the workspace (or a writable directory before the system binaries).
        executable = shutil.which(argv[0], path=os.defpath)
        if executable is None:
            return f"error: allowlisted command is unavailable: {argv[0]}"
        argv[0] = executable
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
