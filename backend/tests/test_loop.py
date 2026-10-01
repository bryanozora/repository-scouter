"""Tests for the agent loop (app/agent/loop.py).

No real LLM or GitHub calls: the provider is a small scripted fake and
repo_tools is a minimal fake satisfying the same list_directory/read_file
interface RepoTools exposes (already covered by test_tools.py). These
tests are about loop.py's own job: building valid OpenAI-compatible
message history, dispatching tool calls, enforcing the step and time
budgets, and never crashing on a malformed or slow tool call.
"""

from __future__ import annotations

import json
import time

import pytest

from app.agent import loop, prompts
from app.agent import tools as tools_module
from app.config import Settings
from app.github_client import RepoRef
from app.models import LLMResponse, Message, TokenUsage, ToolCall

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
        # Off by default here so the many older tests that end with a bare
        # "done" aren't nudged or followed by a forced-report call; the tests
        # for those features turn them on explicitly.
        agent_nudges=False,
        forced_report=False,
        candidate_searches=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


class ScriptedProvider:
    """An LLMProvider that returns a scripted sequence of responses."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], list[dict] | None]] = []
        self.response_formats: list[dict | None] = []

    def chat(self, messages, tools=None, response_format=None):
        self.calls.append((list(messages), tools))
        self.response_formats.append(response_format)
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
        self.report_finding_calls: list[dict] = []
        self.report_finding_sources: list[str] = []
        self.report_finding_result = lambda args: "Recorded finding #1 (verified: fake)."
        self.forget_shown_calls: list[str] = []
        self.forget_search_lines_calls = 0
        self.candidate_search_calls: list[tuple] = []
        self.candidate_search_result = ""
        self.findings: list = []
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

    def forget_shown(self, path):
        self.forget_shown_calls.append(path)

    def forget_search_lines(self):
        self.forget_search_lines_calls += 1

    def candidate_search(self, terms):
        self.candidate_search_calls.append(terms)
        return self.candidate_search_result

    def report_finding(self, args, source="tool_call"):
        self.report_finding_calls.append(args)
        self.report_finding_sources.append(source)
        return self.report_finding_result(args)


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

    assert result.summary == "This is a Flask app."
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

    assert result.summary == "Final summary."
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

    assert result.summary == "done"
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

    assert result.summary == "done"
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

    assert result.summary == "ok"
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

    assert result.summary == "ok"
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

    assert "Fallback summary." in result.summary
    assert "step limit" in result.summary.lower()
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

    assert "Fallback after timeout." in result.summary
    assert "timeout" in result.summary.lower()
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

    assert result.summary == "done"
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


# ---------------------------------------------------------------------------
# report_finding dispatch and ScanResult
# ---------------------------------------------------------------------------


def _call(name: str, args: str = "{}", call_id: str = "call_1", usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=args)], usage=usage)


def _done(text: str = "done", usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], usage=usage)


def test_run_scan_dispatches_report_finding_with_the_full_argument_dict():
    args = {"severity": "high", "file": "db.py", "line_start": 4, "line_end": 4}
    provider = ScriptedProvider([_call("report_finding", json.dumps(args)), _done()])
    repo_tools = FakeRepoTools()

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert repo_tools.report_finding_calls == [args]


def test_run_scan_sends_all_five_tool_schemas():
    provider = ScriptedProvider([_done()])

    loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=make_settings())

    names = [t["function"]["name"] for t in provider.calls[0][1]]
    assert names == ["list_directory", "read_file", "get_dependencies", "search_code", "report_finding"]


def test_run_scan_returns_findings_and_stats_for_a_completed_scan():
    provider = ScriptedProvider([_call("get_dependencies"), _done("A Flask app.")])
    repo_tools = FakeRepoTools()
    repo_tools.findings = ["finding-a", "finding-b"]  # opaque to the loop

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert isinstance(result, loop.ScanResult)
    assert result.summary == "A Flask app."
    assert result.findings == ["finding-a", "finding-b"]
    assert result.stats.steps == 2
    assert result.stats.finish_reason == "completed"
    assert result.stats.duration_seconds >= 0


def test_run_scan_stats_record_step_limit_as_finish_reason():
    settings = make_settings(max_steps=2)
    provider = ScriptedProvider([_call("get_dependencies"), _call("search_code", '{"query": "x"}', "c2"), _done("fb")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=settings)

    assert result.stats.finish_reason.startswith("step limit")
    assert result.stats.steps == 2


# ---------------------------------------------------------------------------
# token accounting
# ---------------------------------------------------------------------------


def test_run_scan_sums_reported_token_usage_across_calls():
    provider = ScriptedProvider(
        [_call("get_dependencies", usage=TokenUsage(1000, 20)), _done(usage=TokenUsage(1300, 50))]
    )

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=make_settings())

    assert result.stats.prompt_tokens == 2300
    assert result.stats.completion_tokens == 70
    assert result.stats.max_prompt_tokens_seen == 1300
    assert result.stats.tokens_estimated is False


def test_run_scan_estimates_tokens_when_the_provider_reports_none():
    provider = ScriptedProvider([_done("some answer")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=make_settings())

    assert result.stats.tokens_estimated is True
    assert result.stats.prompt_tokens > 0
    assert result.stats.completion_tokens > 0


def test_run_scan_counts_the_fallback_call_too():
    settings = make_settings(max_steps=1)
    provider = ScriptedProvider(
        [_call("get_dependencies", usage=TokenUsage(1000, 10)), _done("fallback", usage=TokenUsage(1200, 40))]
    )

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=settings)

    assert result.stats.prompt_tokens == 2200
    assert result.stats.completion_tokens == 50


def test_run_scan_stops_with_fallback_when_total_token_budget_is_spent():
    settings = make_settings(max_steps=10, max_total_tokens=1500, max_prompt_tokens=100_000)
    provider = ScriptedProvider(
        [
            _call("get_dependencies", usage=TokenUsage(1000, 10)),
            _call("search_code", '{"query": "x"}', "c2", usage=TokenUsage(1100, 10)),
            _done("Fallback after budget."),
        ]
    )

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=settings)

    assert "Fallback after budget." in result.summary
    assert result.stats.finish_reason.startswith("token budget")
    assert len(provider.calls) == 3  # 2 steps, then over budget -> fallback


# ---------------------------------------------------------------------------
# context-window budget: compact old tool results, fall back only if needed
# ---------------------------------------------------------------------------

# The loop estimates 1 token per CHARS_PER_TOKEN_ESTIMATE characters.
CPT = loop.CHARS_PER_TOKEN_ESTIMATE


def test_old_tool_results_are_compacted_to_keep_the_prompt_under_budget():
    settings = make_settings(max_steps=10, max_prompt_tokens=3000)
    old = "a" * (1000 * CPT)  # ~1000 tokens
    new = "b" * (500 * CPT)  # ~500 tokens
    provider = ScriptedProvider(
        [
            _call("read_file", '{"path": "a.py"}', "c1", usage=TokenUsage(1500, 20)),  # then +1000 -> 2520, fits
            _call("read_file", '{"path": "b.py"}', "c2", usage=TokenUsage(2600, 20)),  # then +500 -> 3120, over
            _done("done"),
        ]
    )
    repo_tools = FakeRepoTools(read_file_results={"a.py": old, "b.py": new})

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert result.summary == "done"  # no fallback: compaction was enough
    assert result.stats.finish_reason == "completed"
    assert result.stats.compactions == 1
    third_call_tool_msgs = [m for m in provider.calls[2][0] if m.role == "tool"]
    assert old not in third_call_tool_msgs[0].content
    assert "removed" in third_call_tool_msgs[0].content
    assert third_call_tool_msgs[1].content == new  # newest result untouched
    # Compacted read_file lines no longer count as seen.
    assert repo_tools.forget_shown_calls == ["a.py"]


def test_results_from_the_latest_step_are_never_compacted_mid_scan():
    # Cap raised out of the way: this test is about compaction, not the cap.
    settings = make_settings(max_steps=10, max_prompt_tokens=3000, max_tool_result_chars=1_000_000)
    big = "x" * (10_000 * CPT)
    provider = ScriptedProvider([_call("read_file", '{"path": "big.py"}', usage=TokenUsage(1500, 20)), _done("fb")])
    repo_tools = FakeRepoTools(read_file_results={"big.py": big})

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    # Nothing older to compact, still over budget -> context fallback.
    assert result.stats.finish_reason.startswith("context budget")
    assert len(provider.calls) == 2  # step 1, then the fallback -- never a full step 2


def test_context_fallback_compacts_everything_it_needs_to_and_sends_no_tools():
    # Cap raised out of the way: this test is about compaction, not the cap.
    settings = make_settings(max_steps=10, max_prompt_tokens=3000, max_tool_result_chars=1_000_000)
    big = "x" * (10_000 * CPT)
    provider = ScriptedProvider([_call("read_file", '{"path": "big.py"}', usage=TokenUsage(1500, 20)), _done("fb")])
    repo_tools = FakeRepoTools(read_file_results={"big.py": big})

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert "fb" in result.summary
    fallback_messages, fallback_tools = provider.calls[-1]
    assert fallback_tools is None
    tool_msgs = [m for m in fallback_messages if m.role == "tool"]
    assert big not in tool_msgs[0].content
    assert "removed" in tool_msgs[0].content
    # Message structure stays valid for the OpenAI-compatible API.
    assert fallback_messages[0].role == "system"
    assert tool_msgs[0].tool_call_id == "call_1"


def test_step_limit_fallback_does_not_compact_when_everything_fits():
    settings = make_settings(max_steps=1)
    provider = ScriptedProvider([_call("get_dependencies"), _done("fb")])
    repo_tools = FakeRepoTools(dependencies_result="<file_content>\nflask\n</file_content>")

    loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    tool_msgs = [m for m in provider.calls[-1][0] if m.role == "tool"]
    assert "flask" in tool_msgs[0].content


# ---------------------------------------------------------------------------
# prompt content for M2b
# ---------------------------------------------------------------------------


def test_system_prompt_describes_both_v1_issue_types_and_the_read_first_rule():
    lowered = prompts.SYSTEM_PROMPT.lower()
    assert "report_finding" in lowered
    assert "hardcoded_secret" in lowered
    assert "sql_injection" in lowered
    assert "parameterized" in lowered
    assert "environment" in lowered
    assert "read the exact lines" in lowered


def test_fallback_prompt_says_recorded_findings_are_kept():
    lowered = prompts.fallback_message("step limit (3 steps) reached").lower()
    assert "do not call" in lowered
    assert "already recorded" in lowered


def test_oversized_tool_results_are_capped_by_the_loop():
    settings = make_settings(max_tool_result_chars=1000)
    huge = "m" * 50_000
    provider = ScriptedProvider([_call("search_code", '{"query": "x"}'), _done()])
    repo_tools = FakeRepoTools(search_code_result=huge)

    loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    tool_msg = [m for m in provider.calls[1][0] if m.role == "tool"][0]
    assert len(tool_msg.content) <= 1000 + loop.TOOL_RESULT_HEADROOM_CHARS + 200
    assert "truncated" in tool_msg.content


def test_tool_results_within_the_headroom_are_not_cut():
    # read_file already cuts its own body at whole lines; the loop must not
    # re-cut it (that would make RepoTools' shown-lines record wrong).
    settings = make_settings(max_tool_result_chars=1000)
    almost = "r" * (1000 + loop.TOOL_RESULT_HEADROOM_CHARS)
    provider = ScriptedProvider([_call("read_file", '{"path": "a.py"}'), _done()])
    repo_tools = FakeRepoTools(read_file_results={"a.py": almost})

    loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    tool_msg = [m for m in provider.calls[1][0] if m.role == "tool"][0]
    assert tool_msg.content == almost


def test_scan_result_to_dict_splits_and_sorts_findings_by_severity():
    from app.agent.findings import Finding

    def f(sev: str, verified: bool, line: int) -> Finding:
        return Finding(
            severity=sev, category="security", issue_type="sql_injection", file="a.py", line_start=line,
            line_end=line, description="d", suggestion="s", confidence=0.5, verified=verified,
        )

    result = loop.ScanResult(
        summary="sum",
        findings=[f("low", True, 1), f("high", True, 2), f("medium", False, 3), f("info", True, 4)],
        stats=loop.ScanStats(steps=3, finish_reason="completed"),
    )

    data = result.to_dict()

    assert data["summary"] == "sum"
    assert [x["severity"] for x in data["findings"]] == ["high", "low", "info"]
    assert [x["line_start"] for x in data["unverified_findings"]] == [3]
    assert data["stats"]["steps"] == 3
    assert json.loads(json.dumps(data)) == data  # plain JSON-serializable


# ---------------------------------------------------------------------------
# Nudge 1: a tool call written as text gets one corrective retry
# ---------------------------------------------------------------------------


def nudge_settings(**overrides) -> Settings:
    return make_settings(agent_nudges=True, **overrides)


def _thorough_tools() -> FakeRepoTools:
    """Fake tools whose history will satisfy the early-stop check, so tests of
    the text-tool-call nudge aren't also tripping the early-stop nudge."""
    return FakeRepoTools(read_file_results={"app.py": "File: app.py (lines 1-1 of 1)\n<file_content>\n1: x\n</file_content>"})


def _thorough_steps() -> list[LLMResponse]:
    return [
        _call("read_file", '{"path": "app.py"}', "t1"),
        _call("search_code", '{"query": "password"}', "t2"),
        _call("search_code", '{"query": "SELECT"}', "t3"),
    ]


TEXT_READ_CALL = 'Let me read it.\n```json\n{"name": "read_file", "arguments": {"path": "db.py"}}\n```'
TEXT_FINDING = (
    'Found one:\n```json\n{"severity": "high", "category": "security", "issue_type": "sql_injection", '
    '"file": "db.py", "line_start": 24, "line_end": 25, "description": "d", "suggestion": "s"}\n```'
)


def _user_nudges(messages: list[Message]) -> list[str]:
    return [m.content for m in messages if m.role == "user"][1:]  # skip the initial user message


def test_text_tool_call_gets_a_corrective_retry_instead_of_ending_the_scan():
    provider = ScriptedProvider(_thorough_steps() + [_done(TEXT_READ_CALL), _done("Real summary.")])

    result = loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    assert result.summary == "Real summary."
    assert result.stats.text_tool_call_nudges == 1
    last_messages = provider.calls[-1][0]
    assert last_messages[-2].role == "assistant" and last_messages[-2].content == TEXT_READ_CALL
    nudge = last_messages[-1]
    assert nudge.role == "user"
    assert "read_file" in nudge.content
    assert "as text" in nudge.content.lower()


def test_finding_written_as_text_is_executed_not_nudged():
    # Since approach 1, a finding written as text goes through report_finding
    # (source="parsed_text") instead of costing a nudge.
    provider = ScriptedProvider(_thorough_steps() + [_done(TEXT_FINDING)])
    repo_tools = _thorough_tools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=nudge_settings())

    assert result.stats.text_tool_call_nudges == 0
    assert repo_tools.report_finding_sources == ["parsed_text"]
    assert repo_tools.report_finding_calls[0]["line_start"] == 24


def test_qwen_style_tool_call_tags_in_text_are_detected():
    text = '<tool_call>\n{"name": "search_code", "arguments": {"query": "SELECT"}}\n</tool_call>'
    provider = ScriptedProvider(_thorough_steps() + [_done(text), _done("Real summary.")])

    result = loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    assert result.stats.text_tool_call_nudges == 1
    assert "search_code" in provider.calls[-1][0][-1].content


def test_text_tool_call_nudge_happens_only_once_per_scan():
    provider = ScriptedProvider(_thorough_steps() + [_done(TEXT_READ_CALL), _done(TEXT_FINDING)])

    result = loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    assert result.summary == TEXT_FINDING  # second time it's accepted as the final answer
    assert result.stats.text_tool_call_nudges == 1


@pytest.mark.parametrize(
    "text",
    [
        "This is a Flask app with three modules.",
        'Config looks like {"debug": true} in settings.',  # JSON, but not a tool call or a finding
        "The read_file tool showed db.py builds queries safely.",
    ],
)
def test_plain_answers_are_not_mistaken_for_text_tool_calls(text):
    provider = ScriptedProvider(_thorough_steps() + [_done(text)])

    result = loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    assert result.summary == text
    assert result.stats.text_tool_call_nudges == 0


def test_no_text_tool_call_nudge_when_nudges_are_disabled():
    provider = ScriptedProvider([_done(TEXT_READ_CALL)])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=make_settings(agent_nudges=False))

    assert result.summary == TEXT_READ_CALL
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Nudge 2: one push back on an early stop
# ---------------------------------------------------------------------------


def test_early_stop_without_reading_source_or_searching_is_pushed_back_once():
    provider = ScriptedProvider([_done("Looks like a Flask app."), _done("Final.")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=nudge_settings())

    assert result.summary == "Final."
    assert result.stats.early_stop_nudges == 1
    nudge = provider.calls[-1][0][-1].content.lower()
    assert "source file" in nudge
    assert "hardcoded secret" in nudge
    assert "sql injection" in nudge


def test_early_stop_nudge_happens_only_once_per_scan():
    provider = ScriptedProvider([_done("First."), _done("Second.")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=nudge_settings())

    assert result.summary == "Second."
    assert result.stats.early_stop_nudges == 1
    assert len(provider.calls) == 2


def test_no_push_back_when_source_was_read_and_both_issue_types_searched():
    provider = ScriptedProvider(_thorough_steps() + [_done("Final.")])

    result = loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    assert result.summary == "Final."
    assert result.stats.early_stop_nudges == 0


def test_push_back_names_only_the_missing_issue_type():
    provider = ScriptedProvider(
        [
            _call("read_file", '{"path": "app.py"}', "t1"),
            _call("search_code", '{"query": "api_key"}', "t2"),
            _done("Final."),
            _done("Really final."),
        ]
    )

    loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    nudge = provider.calls[-1][0][-1].content.lower()
    assert "sql injection" in nudge
    assert "hardcoded secret" not in nudge
    assert "source file" not in nudge


def test_a_report_finding_call_counts_as_covering_its_issue_type():
    finding = json.dumps({"issue_type": "SQL injection", "file": "app.py", "line_start": 1, "line_end": 1})
    provider = ScriptedProvider(
        [
            _call("read_file", '{"path": "app.py"}', "t1"),
            _call("search_code", '{"query": "password"}', "t2"),
            _call("report_finding", finding, "t3"),
            _done("Final."),
        ]
    )

    result = loop.run_scan(REF, "main", provider, _thorough_tools(), settings=nudge_settings())

    assert result.stats.early_stop_nudges == 0


def test_reading_only_docs_or_manifests_does_not_count_as_reading_source():
    provider = ScriptedProvider(
        [
            _call("read_file", '{"path": "README.md"}', "t1"),
            _call("read_file", '{"path": "requirements.txt"}', "t2"),
            _call("search_code", '{"query": "password"}', "t3"),
            _call("search_code", '{"query": "execute("}', "t4"),
            _done("Final."),
            _done("Really final."),
        ]
    )

    # max_steps=10: the final answer lands on step 5, and the last step is never nudged.
    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=nudge_settings(max_steps=10))

    assert result.stats.early_stop_nudges == 1
    assert "source file" in provider.calls[-1][0][-1].content.lower()


def test_a_failed_read_does_not_count_as_reading_source():
    provider = ScriptedProvider(
        [
            _call("read_file", '{"path": "missing.py"}', "t1"),
            _call("search_code", '{"query": "password"}', "t2"),
            _call("search_code", '{"query": "SELECT"}', "t3"),
            _done("Final."),
            _done("Really final."),
        ]
    )
    repo_tools = FakeRepoTools(read_file_results={"missing.py": "Error: path not found: 'missing.py'"})

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=nudge_settings())

    assert result.stats.early_stop_nudges == 1


def test_no_push_back_on_the_last_step():
    settings = nudge_settings(max_steps=1)
    provider = ScriptedProvider([_done("Final.")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=settings)

    assert result.summary == "Final."
    assert result.stats.early_stop_nudges == 0


def test_text_tool_call_nudge_takes_priority_over_early_stop_nudge():
    provider = ScriptedProvider([_done(TEXT_READ_CALL), _done("Final."), _done("Really final.")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=nudge_settings())

    assert result.stats.text_tool_call_nudges == 1
    assert result.stats.early_stop_nudges == 1
    nudges = _user_nudges(provider.calls[-1][0])
    assert "as text" in nudges[0].lower()
    assert "source file" in nudges[1].lower()


# ---------------------------------------------------------------------------
# Approach 1 in the loop: findings written as text in any reply
# ---------------------------------------------------------------------------


def test_text_findings_alongside_a_tool_call_are_executed_too():
    reply = LLMResponse(
        content='Recording: {"issue_type": "sql_injection", "file": "a.py", "line_start": 2, "line_end": 2}',
        tool_calls=[ToolCall("c1", "get_dependencies", "{}")],
    )
    provider = ScriptedProvider([reply, _done("done")])
    repo_tools = FakeRepoTools()

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    assert repo_tools.report_finding_sources == ["parsed_text"]


def test_every_finding_attempt_is_logged_with_its_source_and_outcome():
    finding = json.dumps({"issue_type": "sql_injection", "file": "a.py", "line_start": 1, "line_end": 1})
    provider = ScriptedProvider(
        [
            _call("report_finding", finding),
            _done('Also {"issue_type": "hardcoded_secret", "file": "b.py", "line_start": 3, "line_end": 3}'),
        ]
    )
    repo_tools = FakeRepoTools()
    outcomes = iter(["Recorded finding #1 (verified: x).", "Error: report_finding rejected, nothing was recorded."])
    repo_tools.report_finding_result = lambda args: next(outcomes)

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings())

    attempts = result.stats.finding_attempts
    assert [a["source"] for a in attempts] == ["tool_call", "parsed_text"]
    assert attempts[0]["file"] == "a.py" and attempts[0]["outcome"].startswith("Recorded")
    assert attempts[1]["file"] == "b.py" and attempts[1]["outcome"].startswith("Error")


# ---------------------------------------------------------------------------
# Approach 2: forced JSON reporting step after the scan
# ---------------------------------------------------------------------------


def forced_settings(**overrides) -> Settings:
    return make_settings(forced_report=True, **overrides)


def _forced(findings: list[dict] | str) -> LLMResponse:
    content = findings if isinstance(findings, str) else json.dumps({"findings": findings})
    return LLMResponse(content=content, tool_calls=[])


F1 = {"severity": "high", "category": "security", "issue_type": "sql_injection", "file": "db.py",
      "line_start": 24, "line_end": 25, "description": "d", "suggestion": "s"}
F2 = {**F1, "issue_type": "hardcoded_secret", "file": "config.py", "line_start": 17, "line_end": 17}


def test_forced_report_runs_after_the_final_answer_with_a_json_schema_and_no_tools():
    provider = ScriptedProvider([_done("Summary."), _forced([F1, F2])])
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=forced_settings())

    assert result.summary == "Summary."  # the forced step never replaces the summary
    messages, tools = provider.calls[-1]
    assert tools is None
    fmt = provider.response_formats[-1]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["properties"]["findings"]["type"] == "array"
    assert messages[-2].role == "assistant" and messages[-2].content == "Summary."
    assert messages[-1].role == "user" and "json" in messages[-1].content.lower()
    assert repo_tools.report_finding_calls == [F1, F2]
    assert repo_tools.report_finding_sources == ["forced_json", "forced_json"]
    assert result.stats.forced_report_rounds == 1


def test_forced_report_schema_offers_only_v1_enum_values():
    schema = loop.FORCED_REPORT_FORMAT["json_schema"]["schema"]["properties"]["findings"]["items"]

    assert schema["properties"]["issue_type"]["enum"] == ["hardcoded_secret", "sql_injection"]
    assert schema["properties"]["category"]["enum"] == ["security"]


def test_forced_report_empty_list_records_nothing():
    provider = ScriptedProvider([_done("Summary."), _forced([])])
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=forced_settings())

    assert repo_tools.report_finding_calls == []
    assert result.stats.forced_report_rounds == 1


def test_one_correction_round_when_forced_findings_are_rejected():
    provider = ScriptedProvider([_done("Summary."), _forced([F1, F2]), _forced([{**F1, "line_end": 24}])])
    repo_tools = FakeRepoTools()
    outcomes = iter(
        [
            "Error: report_finding rejected, nothing was recorded. Fix these and call it again:\n- line_start: you have only seen lines 24 of 'db.py'",
            "Recorded finding #1 (verified: x).",
            "Recorded finding #2 (verified: y).",
        ]
    )
    repo_tools.report_finding_result = lambda args: next(outcomes)

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=forced_settings())

    assert result.stats.forced_report_rounds == 2
    correction = provider.calls[-1][0][-1].content
    assert "rejected" in correction.lower()
    assert "you have only seen lines 24" in correction
    assert "cannot call tools" in correction.lower()
    assert provider.response_formats[-1] == loop.FORCED_REPORT_FORMAT
    assert repo_tools.report_finding_sources == ["forced_json"] * 3


def test_correction_round_happens_at_most_once():
    provider = ScriptedProvider([_done("Summary."), _forced([F1]), _forced([F1])])
    repo_tools = FakeRepoTools()
    repo_tools.report_finding_result = lambda args: "Error: report_finding rejected, nothing was recorded."

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=forced_settings())

    assert result.stats.forced_report_rounds == 2
    assert len(provider.calls) == 3


def test_no_correction_round_for_duplicates_or_flagged_findings():
    provider = ScriptedProvider([_done("Summary."), _forced([F1, F2])])
    repo_tools = FakeRepoTools()
    outcomes = iter(["Note: this was already recorded as finding #1", "Recorded finding #2, but flagged UNVERIFIED: x"])
    repo_tools.report_finding_result = lambda args: next(outcomes)

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=forced_settings())

    assert result.stats.forced_report_rounds == 1


def test_forced_report_tolerates_invalid_json():
    provider = ScriptedProvider([_done("Summary."), _forced("not json at all")])
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=forced_settings())

    assert result.summary == "Summary."
    assert repo_tools.report_finding_calls == []
    assert "no valid json" in result.stats.forced_report_note.lower()


def test_forced_report_survives_a_provider_error():
    class FailingSecondCall(ScriptedProvider):
        def chat(self, messages, tools=None, response_format=None):
            if response_format is not None:
                raise RuntimeError("ollama went away")
            return super().chat(messages, tools, response_format)

    provider = FailingSecondCall([_done("Summary.")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=forced_settings())

    assert result.summary == "Summary."
    assert "ollama went away" in result.stats.forced_report_note


def test_forced_report_also_runs_after_a_fallback():
    settings = forced_settings(max_steps=1)
    provider = ScriptedProvider([_call("get_dependencies"), _done("Fallback summary."), _forced([F1])])
    repo_tools = FakeRepoTools()

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert "Fallback summary." in result.summary
    messages = provider.calls[-1][0]
    assert messages[-2].content == "Fallback summary."  # the fallback reply is in the conversation
    assert repo_tools.report_finding_sources == ["forced_json"]


def test_forced_report_tokens_are_counted():
    provider = ScriptedProvider([_done("Summary.", usage=TokenUsage(1000, 10)), LLMResponse(
        content=json.dumps({"findings": []}), tool_calls=[], usage=TokenUsage(1100, 5))])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=forced_settings())

    assert result.stats.prompt_tokens == 2100


def test_forced_report_can_be_disabled():
    provider = ScriptedProvider([_done("Summary.")])

    result = loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=make_settings(forced_report=False))

    assert len(provider.calls) == 1
    assert result.stats.forced_report_rounds == 0


def test_compacting_a_search_result_forgets_search_seen_lines():
    settings = make_settings(max_steps=10, max_prompt_tokens=3000)
    old = "s" * (1000 * CPT)
    new = "b" * (500 * CPT)
    provider = ScriptedProvider(
        [
            _call("search_code", '{"query": "SELECT"}', "c1", usage=TokenUsage(1500, 20)),
            _call("read_file", '{"path": "b.py"}', "c2", usage=TokenUsage(2600, 20)),
            _done("done"),
        ]
    )
    repo_tools = FakeRepoTools(search_code_result=old, read_file_results={"b.py": new})

    loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert repo_tools.forget_search_lines_calls == 1
    assert repo_tools.forget_shown_calls == []


def test_final_step_compaction_does_not_forget_seen_lines():
    # Round-3 bug: the forced step compacted the conversation and that cleared
    # seen lines, so every forced_json finding was rejected as "not seen".
    # Lines seen earlier in the scan stay seen for the fallback/forced steps.
    big = "x" * (10_000 * CPT)
    settings = make_settings(
        max_steps=10, max_prompt_tokens=3000, max_tool_result_chars=1_000_000, forced_report=True
    )
    provider = ScriptedProvider(
        [
            _call("read_file", '{"path": "big.py"}', usage=TokenUsage(1500, 20)),
            _done("fallback summary"),
            _forced([]),
        ]
    )
    repo_tools = FakeRepoTools(read_file_results={"big.py": big})

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=settings)

    assert result.stats.compactions >= 1  # it did compact to fit the final calls...
    assert repo_tools.forget_shown_calls == []  # ...without forgetting what was seen
    assert repo_tools.forget_search_lines_calls == 0


# ---------------------------------------------------------------------------
# Approach 3: fixed candidate searches before the first turn
# ---------------------------------------------------------------------------

CANDIDATE_BLOCK = '<file_content>\nconfig.py:17: SMTP_PASSWORD = "Tr0ub4dor&3-prod"\n</file_content>'


def test_candidate_search_results_go_into_the_first_message():
    provider = ScriptedProvider([_done("Summary.")])
    repo_tools = FakeRepoTools()
    repo_tools.candidate_search_result = CANDIDATE_BLOCK

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings(candidate_searches=True))

    assert repo_tools.candidate_search_calls == [tools_module.CANDIDATE_TERMS]
    first_user = provider.calls[0][0][1].content
    assert CANDIDATE_BLOCK in first_user
    assert "not findings" in first_user


def test_candidate_searches_are_not_a_tool_or_a_step():
    provider = ScriptedProvider([_done("Summary.")])
    repo_tools = FakeRepoTools()
    repo_tools.candidate_search_result = CANDIDATE_BLOCK

    result = loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings(candidate_searches=True))

    assert result.stats.steps == 1
    assert "candidate_search" not in [t["function"]["name"] for t in provider.calls[0][1]]


def test_no_candidate_block_when_nothing_matched():
    provider = ScriptedProvider([_done("Summary.")])

    loop.run_scan(REF, "main", provider, FakeRepoTools(), settings=make_settings(candidate_searches=True))

    assert "automatic search" not in provider.calls[0][0][1].content


def test_candidate_searches_can_be_disabled():
    provider = ScriptedProvider([_done("Summary.")])
    repo_tools = FakeRepoTools()

    loop.run_scan(REF, "main", provider, repo_tools, settings=make_settings(candidate_searches=False))

    assert repo_tools.candidate_search_calls == []


def test_candidate_searches_count_as_checking_both_issue_types_for_the_early_stop_nudge():
    provider = ScriptedProvider([_done("Too early."), _done("Final.")])

    result = loop.run_scan(
        REF, "main", provider, FakeRepoTools(), settings=nudge_settings(candidate_searches=True)
    )

    nudge = provider.calls[-1][0][-1].content.lower()
    assert result.stats.early_stop_nudges == 1
    assert "source file" in nudge
    assert "hardcoded secret" not in nudge and "sql injection" not in nudge
