"""Tests for the agent loop (app/agent/loop.py).

No real LLM or GitHub calls: the provider is a small scripted fake and
repo_tools is a minimal fake satisfying the same list_directory/read_file
interface RepoTools exposes (already covered by test_tools.py). These
tests are about loop.py's own job: building valid OpenAI-compatible
message history, dispatching tool calls, enforcing the step and time
budgets, and never crashing on a malformed or slow tool call.
"""

from __future__ import annotations

import time

import pytest

from app.agent import loop, prompts
from app.config import Settings
from app.github_client import RepoRef
from app.models import LLMResponse, Message, ToolCall

REF = RepoRef(owner="octocat", repo="Hello-World")


def make_settings(**overrides) -> Settings:
    defaults = dict(
        llm_provider="ollama",
        llm_model="qwen2.5:7b-instruct",
        ollama_base_url="http://localhost:11434/v1",
        github_token=None,
        max_steps=5,
        max_file_bytes=100_000,
        max_tool_result_chars=6000,
        tool_timeout_seconds=5.0,
        scan_timeout_seconds=60.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class ScriptedProvider:
    """An LLMProvider that returns a scripted sequence of responses."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], list[dict] | None]] = []

    def chat(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of scripted responses")
        return self._responses.pop(0)


class FakeRepoTools:
    """A minimal stand-in for RepoTools, recording calls."""

    def __init__(
        self,
        list_directory_result="dir  src\nfile README.md (10B)",
        read_file_results=None,
        delay=0.0,
        dependencies_result="<file_content>\nrequirements.txt:\nhttpx>=0.27\n</file_content>",
        search_code_result="<file_content>\nNo matches.\n</file_content>",
    ):
        self.list_directory_calls: list[str] = []
        self.read_file_calls: list[tuple[str, object, object]] = []
        self.get_dependencies_calls: int = 0
        self.search_code_calls: list[str] = []
        self._list_result = list_directory_result
        self._read_results = read_file_results or {}
        self._delay = delay
        self._dependencies_result = dependencies_result
        self._search_code_result = search_code_result

    def list_directory(self, path):
        self.list_directory_calls.append(path)
        return self._list_result

    def read_file(self, path, start=None, end=None):
        if self._delay:
            time.sleep(self._delay)
        self.read_file_calls.append((path, start, end))
        return self._read_results.get(path, f"<file_content>\ncontent of {path}\n</file_content>")

    def get_dependencies(self):
        self.get_dependencies_calls += 1
        return self._dependencies_result

    def search_code(self, query):
        self.search_code_calls.append(query)
        return self._search_code_result


def make_fake_clock(*values):
    """A callable returning `values` in order, then repeating the last one."""
    remaining = list(values)

    def clock():
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return clock


# ---------------------------------------------------------------------------
# basic loop behavior
# ---------------------------------------------------------------------------


def test_run_scan_returns_model_text_when_no_tool_calls():
    provider = ScriptedProvider([LLMResponse(content="This is a Flask app.", tool_calls=[])])
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert result == "This is a Flask app."
    assert len(provider.calls) == 1


def test_run_scan_seeds_conversation_with_system_prompt_and_root_listing():
    provider = ScriptedProvider([LLMResponse(content="done", tool_calls=[])])
    repo_tools = FakeRepoTools(list_directory_result="dir  src\nfile README.md (10B)")

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    first_messages = provider.calls[0][0]
    assert first_messages[0].role == "system"
    assert first_messages[0].content == prompts.SYSTEM_PROMPT
    assert first_messages[1].role == "user"
    assert "README.md" in first_messages[1].content
    assert repo_tools.list_directory_calls == [""]


def test_run_scan_executes_tool_call_and_continues():
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": "src"}')],
            ),
            LLMResponse(content="Final summary.", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert result == "Final summary."
    assert "src" in repo_tools.list_directory_calls
    second_call_messages = provider.calls[1][0]
    tool_messages = [m for m in second_call_messages if m.role == "tool"]
    assert any("dir  src" in m.content for m in tool_messages)


def test_run_scan_dispatches_get_dependencies_tool_call():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name="get_dependencies", arguments="{}")]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert result == "done"
    assert repo_tools.get_dependencies_calls == 1


def test_run_scan_dispatches_search_code_tool_call():
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="search_code", arguments='{"query": "password"}')],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert result == "done"
    assert repo_tools.search_code_calls == ["password"]


# ---------------------------------------------------------------------------
# OpenAI-compatible message ordering
# ---------------------------------------------------------------------------


def test_assistant_and_tool_messages_are_ordered_correctly_for_multiple_tool_calls():
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="list_directory", arguments='{"path": ""}'),
                    ToolCall(id="call_2", name="read_file", arguments='{"path": "README.md"}'),
                ],
            ),
            LLMResponse(content="Done.", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    second_call_messages = provider.calls[1][0]
    assistant_idx = next(
        i for i, m in enumerate(second_call_messages) if m.role == "assistant" and m.tool_calls
    )
    assistant_msg = second_call_messages[assistant_idx]
    assert [c.id for c in assistant_msg.tool_calls] == ["call_1", "call_2"]

    tool_msg_1 = second_call_messages[assistant_idx + 1]
    tool_msg_2 = second_call_messages[assistant_idx + 2]
    assert tool_msg_1.role == "tool" and tool_msg_1.tool_call_id == "call_1"
    assert tool_msg_2.role == "tool" and tool_msg_2.tool_call_id == "call_2"


# ---------------------------------------------------------------------------
# malformed / unknown tool calls never crash the loop
# ---------------------------------------------------------------------------


def test_run_scan_handles_unknown_tool_name_without_crashing():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name="delete_repo", arguments="{}")]),
            LLMResponse(content="ok", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert result == "ok"
    tool_messages = [m for m in provider.calls[1][0] if m.role == "tool"]
    assert "unknown tool" in tool_messages[-1].content.lower()


def test_run_scan_handles_malformed_json_arguments_without_crashing():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments="not json")]),
            LLMResponse(content="ok", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert result == "ok"
    # Only the initial root-listing seed call happened -- the malformed
    # tool call itself was never dispatched, caught by the JSON check first.
    assert repo_tools.list_directory_calls == [""]


# ---------------------------------------------------------------------------
# step limit and scan timeout both fall back to a forced summary
# ---------------------------------------------------------------------------


def test_run_scan_stops_at_max_steps_and_forces_summary():
    settings = make_settings(max_steps=3)
    tool_call_response = LLMResponse(
        content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": ""}')]
    )
    provider = ScriptedProvider([tool_call_response] * 3 + [LLMResponse(content="Fallback summary.", tool_calls=[])])
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert "Fallback summary." in result
    assert "step limit" in result.lower()
    assert len(provider.calls) == 4  # 3 steps + 1 forced fallback call


def test_run_scan_stops_on_scan_timeout_with_fallback():
    settings = make_settings(max_steps=10, scan_timeout_seconds=60.0)
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": ""}')]
            ),
            LLMResponse(content="Fallback after timeout.", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()
    clock = make_fake_clock(0.0, 0.0, 500.0)  # start=0, step1 check=0 (ok), step2 check=500 (over budget)

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings, now=clock)

    assert "Fallback after timeout." in result
    assert "timeout" in result.lower()
    assert len(provider.calls) == 2  # 1 normal step + 1 forced fallback call


# ---------------------------------------------------------------------------
# per-tool-call timeout
# ---------------------------------------------------------------------------


def test_run_scan_tool_call_timeout_returns_error_and_continues():
    settings = make_settings(tool_timeout_seconds=0.05)
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": ""}')]
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools(delay=0.0)

    def slow_list_directory(path):
        time.sleep(0.2)
        return "dir  src"

    repo_tools.list_directory = slow_list_directory  # type: ignore[assignment]

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert result == "done"
    tool_messages = [m for m in provider.calls[1][0] if m.role == "tool"]
    assert "timed out" in tool_messages[-1].content.lower()


# ---------------------------------------------------------------------------
# prompt-injection defense wording
# ---------------------------------------------------------------------------


def test_system_prompt_warns_file_content_is_untrusted():
    assert "<file_content>" in prompts.SYSTEM_PROMPT
    assert "untrusted" in prompts.SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# discouraging a repeat root listing (prompt tuning)
# ---------------------------------------------------------------------------


def test_system_prompt_discourages_relisting_root():
    lowered = prompts.SYSTEM_PROMPT.lower()
    assert "list_directory" in lowered
    assert "already" in lowered
    assert "readme" in lowered
    assert "manifest" in lowered


def test_initial_user_message_reminds_not_to_relist_root():
    msg = prompts.initial_user_message("octocat", "Hello-World", "main", "Python", "file README.md (10B)")

    lowered = msg.lower()
    assert "do not call list_directory" in lowered
    assert "readme" in lowered


def test_system_prompt_describes_get_dependencies_and_search_code():
    lowered = prompts.SYSTEM_PROMPT.lower()
    assert "get_dependencies" in lowered
    assert "search_code" in lowered


def test_system_prompt_says_to_confirm_search_matches_with_read_file():
    lowered = prompts.SYSTEM_PROMPT.lower()
    assert "confirm" in lowered
    assert "candidate" in lowered


# ---------------------------------------------------------------------------
# loop-level exact-repeat-call detection
# ---------------------------------------------------------------------------


def test_run_scan_adds_corrective_hint_on_exact_repeated_tool_call():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": "src"}')]),
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_2", name="list_directory", arguments='{"path": "src"}')]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()  # dumb stand-in: any hint here must come from loop.py, not RepoTools' own dedup

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    third_call_messages = provider.calls[2][0]
    tool_messages = [m for m in third_call_messages if m.role == "tool"]
    assert len(tool_messages) == 2
    assert "already called" not in tool_messages[0].content.lower()
    assert "already called" in tool_messages[1].content.lower()


def test_run_scan_does_not_flag_repeats_with_different_arguments():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": "src"}')]),
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_2", name="list_directory", arguments='{"path": "docs"}')]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    third_call_messages = provider.calls[2][0]
    tool_messages = [m for m in third_call_messages if m.role == "tool"]
    assert all("already called" not in m.content.lower() for m in tool_messages)


def test_run_scan_flags_repeat_despite_json_whitespace_differences():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name="list_directory", arguments='{"path": "src"}')]),
            LLMResponse(content=None, tool_calls=[ToolCall(id="call_2", name="list_directory", arguments='{"path":"src"}')]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    repo_tools = FakeRepoTools()

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    third_call_messages = provider.calls[2][0]
    tool_messages = [m for m in third_call_messages if m.role == "tool"]
    assert "already called" in tool_messages[1].content.lower()
