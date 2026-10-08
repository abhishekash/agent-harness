"""Core data types shared across the harness.

Everything here is deliberately dependency-free so that providers, tools,
HITL policies, and tracing all speak the same small language.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Risk(str, Enum):
    """Side-effect classification for tools. Drives HITL approval policy."""

    READ = "read"  # no side effects (read_file, list_dir)
    WRITE = "write"  # mutates local state (write_file)
    EXECUTE = "execute"  # runs code, network, or remote side effects


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def new(name: str, arguments: dict[str, Any]) -> "ToolCall":
        return ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    # set on role == "tool" results:
    tool_call_id: str | None = None
    name: str | None = None

    @staticmethod
    def system(content: str) -> "Message":
        return Message(role="system", content=content)

    @staticmethod
    def user(content: str) -> "Message":
        return Message(role="user", content=content)

    @staticmethod
    def assistant(msg: "AssistantMessage") -> "Message":
        return Message(role="assistant", content=msg.content, tool_calls=list(msg.tool_calls))

    @staticmethod
    def tool_result(call: ToolCall, content: str) -> "Message":
        return Message(role="tool", content=content, tool_call_id=call.id, name=call.name)


@dataclass
class AssistantMessage:
    """What a provider returns from one completion."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
