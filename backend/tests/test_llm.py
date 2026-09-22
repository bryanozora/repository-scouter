"""Tests for the LLM provider abstraction (app/llm.py), no real network calls.

All HTTP traffic is intercepted with respx; these tests only check that
OllamaProvider builds the right request and parses the OpenAI-compatible
response shape into our own dataclasses.
"""

import json

import httpx
import pytest
import respx

from app.config import Settings
from app.llm import OllamaProvider, get_provider
from app.models import Message

BASE_URL = "http://localhost:11434/v1"


def make_settings(**overrides) -> Settings:
    defaults = dict(
        llm_provider="ollama",
        llm_model="qwen2.5:7b-instruct",
        ollama_base_url=BASE_URL,
        github_token=None,
        max_steps=12,
        max_file_bytes=100_000,
        max_tool_result_chars=6000,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@respx.mock
def test_chat_returns_plain_text_content_when_no_tool_call():
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hello there"}}]},
        )
    )
    provider = OllamaProvider(base_url=BASE_URL, model="qwen2.5:7b-instruct")

    response = provider.chat([Message(role="user", content="hi")])

    assert response.content == "hello there"
    assert response.tool_calls == []


@respx.mock
def test_chat_parses_a_tool_call_into_a_toolcall():
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "list_directory",
                                        "arguments": '{"path": "src"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )
    )
    provider = OllamaProvider(base_url=BASE_URL, model="qwen2.5:7b-instruct")

    response = provider.chat(
        [Message(role="user", content="list src")],
        tools=[{"type": "function", "function": {"name": "list_directory"}}],
    )

    assert response.content is None
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.name == "list_directory"
    # arguments stay a raw JSON string (matches the OpenAI-compatible wire
    # format) -- parsing/validation is the caller's job.
    assert json.loads(call.arguments) == {"path": "src"}


@respx.mock
def test_chat_sends_model_messages_and_tools_in_the_request_body():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    provider = OllamaProvider(base_url=BASE_URL, model="qwen2.5:7b-instruct")
    tools = [{"type": "function", "function": {"name": "list_directory"}}]

    provider.chat(
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
        tools=tools,
    )

    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "qwen2.5:7b-instruct"
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert sent["tools"] == tools


@respx.mock
def test_chat_omits_tools_key_when_no_tools_given():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    )
    provider = OllamaProvider(base_url=BASE_URL, model="qwen2.5:7b-instruct")

    provider.chat([Message(role="user", content="hi")])

    sent = json.loads(route.calls.last.request.content)
    assert "tools" not in sent


@respx.mock
def test_chat_raises_on_http_error_status():
    respx.post(f"{BASE_URL}/chat/completions").mock(return_value=httpx.Response(500))
    provider = OllamaProvider(base_url=BASE_URL, model="qwen2.5:7b-instruct")

    with pytest.raises(httpx.HTTPStatusError):
        provider.chat([Message(role="user", content="hi")])


def test_get_provider_returns_ollama_provider_for_ollama_setting():
    provider = get_provider(make_settings())

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5:7b-instruct"
    assert provider.base_url == BASE_URL


def test_get_provider_raises_for_unknown_provider():
    with pytest.raises(ValueError):
        get_provider(make_settings(llm_provider="openai"))
