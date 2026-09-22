"""LLM provider abstraction.

The agent loop must never import a vendor SDK or call a specific LLM's API
directly -- it only talks to an LLMProvider. That keeps the provider/model
swappable via .env (LLM_PROVIDER, LLM_MODEL) without touching agent code.

Default provider: Ollama, called through its OpenAI-compatible
/v1/chat/completions endpoint with raw httpx (no OpenAI SDK dependency).
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from .config import Settings, get_settings
from .models import LLMResponse, Message, ToolCall

# First call to a model may include Ollama loading it into memory, which can
# take a while on a small machine -- matches the smoke test's timeout.
DEFAULT_TIMEOUT = 300.0
DEFAULT_TEMPERATURE = 0.2


class LLMProvider(Protocol):
    """Anything the agent loop can call an LLM through."""

    def chat(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        """Send a conversation (+ optional tool schemas) and get one reply."""
        ...


class OllamaProvider:
    """Talks to a local Ollama server via its OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def chat(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]

        tool_calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in (message.get("tool_calls") or [])
        ]
        return LLMResponse(content=message.get("content"), tool_calls=tool_calls)


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Pick an LLMProvider based on LLM_PROVIDER (from settings or .env)."""
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.llm_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


if __name__ == "__main__":
    import sys

    settings = get_settings()
    provider = get_provider(settings)
    print(f"[llm] provider={settings.llm_provider} model={settings.llm_model} base_url={settings.ollama_base_url}")

    user_message = " ".join(sys.argv[1:]) or "Hello!"
    response = provider.chat([Message(role="user", content=user_message)])
    print(response.content)
