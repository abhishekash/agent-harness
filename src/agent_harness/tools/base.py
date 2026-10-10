"""Tool protocol: anything the agent may call.

Tools declare a Risk tier; the HITL policy uses it to decide which calls
need a human's approval. JSON-schema ``parameters`` are passed through to
providers unchanged.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_harness.types import Risk


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    risk: Risk
    parameters: dict[str, Any]  # JSON Schema object for the arguments

    def run(self, **kwargs: Any) -> str:
        """Execute and return a string result (fed back to the model as-is)."""
        ...


def tool_schema(tool: Tool) -> dict[str, Any]:
    """Provider-facing description of a tool."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }


def validate_tool_arguments(tool: Tool, arguments: Any) -> str | None:
    """Validate the small JSON-Schema subset used by providers before approval."""
    if not isinstance(arguments, dict):
        return f"error: bad arguments for {tool.name}: expected an object"
    schema = tool.parameters or {}
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        return f"error: bad arguments for {tool.name}: missing {', '.join(missing)}"
    properties = schema.get("properties", {})
    for name, value in arguments.items():
        spec = properties.get(name)
        if not spec or "type" not in spec:
            continue
        expected = spec["type"]
        valid = {
            "string": isinstance(value, str),
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "boolean": isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
        }.get(expected, True)
        if not valid:
            return f"error: bad arguments for {tool.name}: {name} must be {expected}"
    return None
