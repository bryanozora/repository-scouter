"""Small dataclasses shared between the LLM provider and the (future) agent loop.

Kept intentionally thin: these mirror the OpenAI-compatible chat-completions
wire format closely enough that a provider only has to translate to/from
them, without baking in agent-loop concerns like validation or retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """One tool call requested by the model."""

    id: str
    name: str
    # Raw JSON string, exactly as the API returns it. Parsing and schema
    # validation happen where the call is consumed (the agent loop), not
    # here -- callers like the smoke test need to observe invalid JSON
    # rather than have it raise underneath them.
    arguments: str


@dataclass
class Message:
    """One entry in a chat-completions conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # set on role="tool" replies

    def to_api_dict(self) -> dict[str, Any]:
        """Render this message as the OpenAI-compatible JSON shape."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return data


@dataclass
class LLMResponse:
    """The model's reply to one chat() call."""

    content: str | None
    tool_calls: list[ToolCall]
