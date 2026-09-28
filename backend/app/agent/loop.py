"""The basic M1 agent loop: wires an LLMProvider and a repo-tools object
together into a bounded explore-then-explain conversation.

Deliberately dependency-injected (provider, repo_tools, even the clock) so
it can be tested with scripted fakes -- no real LLM or GitHub calls, see
tests/test_loop.py.

Two budgets bound a scan, per PLAN.md's "Timeouts per tool call and per
scan" guardrail:
  - max_steps: how many provider round-trips the loop may make.
  - scan_timeout: wall-clock budget for the whole scan, checked once per
    step (best-effort -- a single very slow provider call can still run
    past it by up to one step; there's no way to preempt a blocking call
    mid-flight without an async rewrite, which is out of scope for M1).
Each individual tool call additionally gets its own tool_timeout, enforced
with a worker thread since signal-based timeouts aren't portable to
Windows.

If either budget runs out before the model finishes on its own, one final
tool-less call asks it to summarize whatever it has learned so far, so a
scan always produces *something* rather than stopping cold.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Callable, Protocol

from . import prompts
from .tools import TOOLS
from ..config import Settings, get_settings
from ..github_client import RepoRef
from ..llm import LLMProvider
from ..models import LLMResponse, Message, ToolCall


class ToolExecutor(Protocol):
    """What RepoTools provides and all run_scan actually needs from it."""

    def list_directory(self, path: str) -> str: ...
    def read_file(self, path: str, start: object = None, end: object = None) -> str: ...


def run_scan(
    ref: RepoRef,
    branch: str,
    provider: LLMProvider,
    repo_tools: ToolExecutor,
    language: str | None = None,
    max_steps: int | None = None,
    tool_timeout: float | None = None,
    scan_timeout: float | None = None,
    settings: Settings | None = None,
    on_step: Callable[[int, LLMResponse, dict[str, str]], None] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> str:
    """Run one explore-then-explain scan and return the final summary text."""
    settings = settings or get_settings()
    max_steps = max_steps if max_steps is not None else settings.max_steps
    tool_timeout = tool_timeout if tool_timeout is not None else settings.tool_timeout_seconds
    scan_timeout = scan_timeout if scan_timeout is not None else settings.scan_timeout_seconds

    root_listing = repo_tools.list_directory("")
    messages: list[Message] = [
        Message(role="system", content=prompts.SYSTEM_PROMPT),
        Message(
            role="user",
            content=prompts.initial_user_message(ref.owner, ref.repo, branch, language, root_listing),
        ),
    ]

    seen_calls: set[tuple[str, str]] = set()
    start_time = now()
    for step in range(1, max_steps + 1):
        if now() - start_time > scan_timeout:
            return _finish_with_fallback(provider, messages, reason=f"scan timeout ({scan_timeout}s) reached")

        response = provider.chat(messages, tools=TOOLS)

        if not response.tool_calls:
            if on_step:
                on_step(step, response, {})
            return response.content or ""

        messages.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))
        tool_results: dict[str, str] = {}
        for call in response.tool_calls:
            result = _execute_tool_call(repo_tools, call, tool_timeout)

            call_key = (call.name, _normalize_args_for_repeat_check(call.arguments))
            if call_key in seen_calls:
                result += (
                    "\n\nNote: you already called this exact tool with these exact arguments "
                    "earlier in this scan. Try something different -- read a specific file, or "
                    "if you have enough information, give your final answer now without calling "
                    "another tool."
                )
            else:
                seen_calls.add(call_key)

            tool_results[call.id] = result
            messages.append(Message(role="tool", content=result, tool_call_id=call.id))

        if on_step:
            on_step(step, response, tool_results)

    return _finish_with_fallback(provider, messages, reason=f"step limit ({max_steps} steps) reached")


def _finish_with_fallback(provider: LLMProvider, messages: list[Message], reason: str) -> str:
    """Force one last tool-less reply summarizing whatever's been learned so far."""
    fallback_messages = messages + [
        Message(
            role="user",
            content=(
                f"You've reached the {reason}. Based on everything you've explored so far, write "
                "your best architecture summary now, in plain text. Do not call any more tools."
            ),
        )
    ]
    try:
        response = provider.chat(fallback_messages)
        summary = response.content or "(the model did not return a summary)"
    except Exception as exc:  # best-effort: still return *something* if even this call fails
        summary = f"(could not produce a final summary: {exc})"
    return f"[Scan ended: {reason}.]\n\n{summary}"


def _execute_tool_call(repo_tools: ToolExecutor, call: ToolCall, timeout: float) -> str:
    args, err = _parse_tool_arguments(call.arguments)
    if err:
        return f"Error: {err}"
    return _run_with_timeout(lambda: _dispatch_tool(repo_tools, call.name, args), timeout, call.name)


def _normalize_args_for_repeat_check(raw: str) -> str:
    """Canonical form of a tool call's arguments, so whitespace/key-order
    differences in otherwise-identical JSON don't defeat repeat detection."""
    try:
        parsed = json.loads(raw) if raw else {}
        return json.dumps(parsed, sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        return raw or ""


def _parse_tool_arguments(raw: str) -> tuple[dict, str | None]:
    if not raw:
        return {}, None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}, f"arguments were not valid JSON: {raw!r}"
    if not isinstance(parsed, dict):
        return {}, f"arguments must be a JSON object, got: {raw!r}"
    return parsed, None


def _dispatch_tool(repo_tools: ToolExecutor, name: str, args: dict) -> str:
    if name == "list_directory":
        return repo_tools.list_directory(args.get("path", ""))
    if name == "read_file":
        return repo_tools.read_file(args.get("path", ""), start=args.get("start"), end=args.get("end"))
    return f"Error: unknown tool {name!r}"


def _run_with_timeout(fn: Callable[[], str], timeout: float, tool_name: str) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return f"Error: {tool_name} timed out after {timeout}s"
        except Exception as exc:  # tools shouldn't raise (see tools.py), but don't trust that blindly
            return f"Error: unexpected failure calling {tool_name}: {exc}"
