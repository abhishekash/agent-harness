"""Hand-rolled MCP server fixture (NDJSON over stdio, protocol 2024-11-05).

Zero dependencies so the harness test-suite stays self-contained.
Implements: initialize, notifications/initialized, tools/list, tools/call.
Tools: echo(text), add(a, b).
"""
import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo back the input text",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
]


def respond(req_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        if method == "initialize":
            respond(msg["id"], {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo-fixture", "version": "0.0.1"},
            })
        elif method == "notifications/initialized":
            pass  # no response for notifications
        elif method == "tools/list":
            respond(msg["id"], {"tools": TOOLS})
        elif method == "tools/call":
            params = msg.get("params", {})
            name, args = params.get("name"), params.get("arguments", {})
            if name == "echo":
                respond(msg["id"], {"content": [{"type": "text", "text": str(args.get("text", ""))}]})
            elif name == "add":
                respond(msg["id"], {"content": [{"type": "text", "text": str(args.get("a", 0) + args.get("b", 0))}]})
            else:
                respond(msg["id"], {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True})


if __name__ == "__main__":
    main()
