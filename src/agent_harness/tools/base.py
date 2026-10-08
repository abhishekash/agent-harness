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
