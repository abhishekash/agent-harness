"""Minimal stdio MCP client — no SDK dependency.

Speaks newline-delimited JSON-RPC 2.0 with an MCP server subprocess:
initialize handshake → tools/list → tools/call. This is enough to mount any
stdio MCP server as harness tools, and small enough to audit line by line.

Spec: https://modelcontextprotocol.io  (stdio transport, protocol 2024-11-05)
"""
from __future__ import annotations

import json
import select
import shlex
import subprocess
import threading
from typing import Any

from agent_harness.types import Risk

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "agent-harness", "version": "0.1.0"}


class MCPError(RuntimeError):
    pass


class StdioMCPClient:
    def __init__(self, argv: list[str] | str, timeout: float = 30.0):
        self.argv = shlex.split(argv) if isinstance(argv, str) else list(argv)
        if not self.argv:
            raise ValueError("MCP server command cannot be empty")
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        self.timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        self._id = 0
        self._lock = threading.Lock()
        self.server_info: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> dict[str, Any]:
        if self._proc is not None:
            raise MCPError("MCP server is already started")
        try:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            result = self._rpc(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
            )
            self.server_info = result.get("serverInfo", {})
            self._notify("notifications/initialized", {})
            return result
        except (OSError, MCPError) as exc:
            self.close()
            if isinstance(exc, MCPError):
                raise
            raise MCPError(f"could not start MCP server {' '.join(self.argv)}: {exc}") from exc

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    def __enter__(self) -> "StdioMCPClient":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- protocol ----------------------------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        return self._rpc("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts = [
            c.get("text", "")
            for c in result.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p)
        if result.get("isError"):
            raise MCPError(text or f"tool {name!r} failed")
        return text or "(no content)"

    # -- plumbing ----------------------------------------------------------
    def _send(self, payload: dict[str, Any]) -> None:
        if not self._proc or self._proc.poll() is not None or not self._proc.stdin:
            raise MCPError(f"MCP server is not running: {' '.join(self.argv)}")
        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except OSError as exc:
            raise MCPError(f"MCP server pipe failed: {exc}") from exc

    def _read_line(self) -> str:
        assert self._proc and self._proc.stdout
        fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], self.timeout)
        if not ready:
            raise MCPError(f"server timed out after {self.timeout}s ({' '.join(self.argv)})")
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.poll()
            raise MCPError(
                f"server closed the pipe ({' '.join(self.argv)}; returncode={code})"
            )
        return line

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._id += 1
            req_id = self._id
            self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
            while True:
                try:
                    msg = json.loads(self._read_line())
                except json.JSONDecodeError as exc:
                    raise MCPError(f"invalid JSON from MCP server: {exc}") from exc
                if msg.get("id") != req_id:
                    continue  # notification or unrelated message
                if "error" in msg:
                    err = msg["error"]
                    raise MCPError(f"{method}: {err.get('code')} {err.get('message')}")
                return msg.get("result", {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})


class MCPTool:
    """Adapts a remote MCP tool to the harness Tool protocol.

    Unknown remote side effects default to Risk.EXECUTE (gate by default);
    pass ``risk_overrides`` to mark known-safe tools (e.g. read-only queries).
    """

    def __init__(
        self,
        client: StdioMCPClient,
        spec: dict[str, Any],
        server_label: str,
        risk_overrides: dict[str, Risk] | None = None,
    ):
        self._client = client
        self._remote_name = spec["name"]
        self.name = f"{server_label}__{spec['name']}"
        self.description = spec.get("description", f"MCP tool {spec['name']}")
        self.parameters = spec.get("inputSchema", {"type": "object", "properties": {}})
        self.risk = (risk_overrides or {}).get(self._remote_name, Risk.EXECUTE)

    def run(self, **kwargs: Any) -> str:
        return self._client.call_tool(self._remote_name, kwargs)


def mount_server(
    argv: list[str] | str,
    label: str | None = None,
    risk_overrides: dict[str, Risk] | None = None,
    timeout: float = 30.0,
) -> tuple[StdioMCPClient, list[MCPTool]]:
    """Start a server, return (client, tools). Caller owns the client lifecycle."""
    client = StdioMCPClient(argv, timeout=timeout)
    client.start()
    label = label or client.server_info.get("name") or "mcp"
    tools = [MCPTool(client, spec, label, risk_overrides) for spec in client.list_tools()]
    return client, tools
