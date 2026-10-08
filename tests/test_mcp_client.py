import sys
from pathlib import Path

import pytest

from agent_harness.mcp import MCPError, StdioMCPClient, mount_server
from agent_harness.types import Risk

FIXTURE = str(Path(__file__).parent / "fixtures" / "echo_mcp_server.py")


@pytest.fixture
def client():
    with StdioMCPClient([sys.executable, FIXTURE]) as c:
        yield c


def test_handshake(client):
    assert client.server_info["name"] == "echo-fixture"


def test_list_tools(client):
    names = [t["name"] for t in client.list_tools()]
    assert names == ["echo", "add"]


def test_call_tool(client):
    assert client.call_tool("echo", {"text": "hi"}) == "hi"
    assert client.call_tool("add", {"a": 1, "b": 2}) == "3"


def test_call_unknown_tool_raises(client):
    with pytest.raises(MCPError):
        client.call_tool("nope", {})


def test_mount_server_namespaces_tools():
    client, tools = mount_server([sys.executable, FIXTURE], risk_overrides={"echo": Risk.READ})
    try:
        by_name = {t.name: t for t in tools}
        assert "echo-fixture__echo" in by_name
        assert by_name["echo-fixture__echo"].risk is Risk.READ
        assert by_name["echo-fixture__add"].risk is Risk.EXECUTE  # gated by default
        assert by_name["echo-fixture__add"].run(a=40, b=2) == "42"
    finally:
        client.close()


def test_timeout_on_dead_server():
    c = StdioMCPClient([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.5)
    with pytest.raises(MCPError, match="timed out"):
        c.start()
    c.close()
