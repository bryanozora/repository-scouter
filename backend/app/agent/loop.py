"""The agent loop: wires an LLMProvider and a repo-tools object together
into a bounded explore-then-explain conversation that also records
structured findings (via the report_finding tool).

Deliberately dependency-injected (provider, repo_tools, even the clock) so
it can be tested with scripted fakes -- no real LLM or GitHub calls, see
tests/test_loop.py.

Budgets that bound a scan (PLAN.md "Mandatory guardrails"):
  - max_steps: how many provider round-trips the loop may make.
  - scan_timeout: wall-clock budget for the whole scan, checked once per
    step (best-effort -- a single very slow provider call can still run
    past it by up to one step; there's no way to preempt a blocking call
    mid-flight without an async rewrite).
  - max_total_tokens: prompt + completion tokens summed over every call.
    Checked before each step, so the last step can overshoot it by one call.
  - max_prompt_tokens: the size of any *single* prompt. This one matters
    most: Ollama silently drops the start of a prompt longer than its
    context window (4096 tokens by default, measured -- see docs/NOTES.md),
    and the start is the system prompt, prompt-injection defense included.
    Before each call the loop projects the next prompt's size; if it's over
    budget it first replaces the oldest tool results with a short marker
    ("compaction"), and only if that isn't enough does it stop.
Each individual tool call additionally gets its own tool_timeout, enforced
with a worker thread since signal-based timeouts aren't portable to
Windows.

If any budget runs out before the model finishes on its own, one final
tool-less call asks it to summarize whatever it has learned so far, so a
scan always produces *something* rather than stopping cold. Findings live
in repo_tools, not in the conversation, so they survive both compaction
and the fallback.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import PurePosixPath
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from . import prompts
from .findings import SEVERITIES, extract_text_findings, findings_in_json
from .tools import CANDIDATE_TERMS, REPORT_FINDING_SCHEMA, SOURCE_EXTENSIONS, TOOLS
from ..config import Settings, get_settings
from ..github_client import RepoRef
from ..llm import LLMProvider
from ..models import LLMResponse, Message, ToolCall

# Rough chars-per-token for projecting prompt size before a call, and for
# estimating usage when the provider doesn't report it. Deliberately
# pessimistic: code tokenizes denser than English prose (~4 chars/token).
CHARS_PER_TOKEN_ESTIMATE = 3

# Any tool result longer than max_tool_result_chars + this is cut by the loop.
# The headroom exists because read_file already caps its own *body* at
# max_tool_result_chars (cutting at whole lines) and then adds a header and
# a note -- the loop must never re-cut that, or RepoTools' record of which
# lines the model has seen would be wrong.
TOOL_RESULT_HEADROOM_CHARS = 500

COMPACTED_TOOL_RESULT = (
    "(This older tool result was removed to fit the model's context window. "
    "Call the tool again if you still need it.)"
)

# The forced final reporting step: Ollama's OpenAI-compatible endpoint
# enforces this schema via constrained decoding (verified directly against
# Ollama 0.10.1 -- see docs/NOTES.md), so the reply is always well-formed
# JSON in this shape. Its *values* can still be wrong, which is why every
# item goes through report_finding's validation, read-gating and evidence check.
FORCED_REPORT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "findings_report",
        "schema": {
            "type": "object",
            "properties": {
                "findings": {"type": "array", "items": REPORT_FINDING_SCHEMA["function"]["parameters"]},
            },
            "required": ["findings"],
        },
    },
}
# The forced step gets one correction round if some findings were rejected.
MAX_FORCED_REPORT_ROUNDS = 2


class ToolExecutor(Protocol):
    """What RepoTools provides and all run_scan actually needs from it."""

    findings: list

    def list_directory(self, path: str) -> str: ...
    def read_file(self, path: str, start: object = None, end: object = None) -> str: ...
    def get_dependencies(self) -> str: ...
    def search_code(self, query: str) -> str: ...
    def report_finding(self, args: dict, source: str = "tool_call") -> str: ...
    def forget_shown(self, path: str) -> None: ...
    def forget_search_lines(self) -> None: ...
    def candidate_search(self, terms: tuple[str, ...]) -> str: ...


@dataclass
class ScanStats:
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # True if any call's usage had to be estimated (provider didn't report it).
    tokens_estimated: bool = False
    max_prompt_tokens_seen: int = 0
    compactions: int = 0
    # Corrective nudges sent (each at most once per scan; see _ReviewCoverage).
    text_tool_call_nudges: int = 0
    early_stop_nudges: int = 0
    # Every report_finding attempt, whichever path it came through:
    # {"source", "issue_type", "file", "line_start", "line_end", "outcome"}.
    finding_attempts: list = field(default_factory=list)
    # The forced JSON reporting step: calls made (1, or 2 with a correction
    # round) and a short note on what happened.
    forced_report_rounds: int = 0
    forced_report_note: str = ""
    duration_seconds: float = 0.0
    # "completed", or the budget that ended the scan, e.g. "step limit (12 steps)".
    finish_reason: str = ""


@dataclass
class ScanResult:
    summary: str
    findings: list
    stats: ScanStats = field(default_factory=ScanStats)

    def to_dict(self) -> dict:
        """The report as plain JSON: verified findings sorted by severity (PLAN.md
        section 5), unverified ones kept separately rather than silently dropped."""
        rank = {sev: i for i, sev in enumerate(SEVERITIES)}

        def ordered(findings: list) -> list[dict]:
            return [f.to_dict() for f in sorted(findings, key=lambda f: (rank.get(f.severity, len(rank)), f.file, f.line_start))]

        return {
            "summary": self.summary,
            "findings": ordered([f for f in self.findings if f.verified]),
            "unverified_findings": ordered([f for f in self.findings if not f.verified]),
            "stats": asdict(self.stats),
        }


class _TokenMeter:
    """Tracks token usage, and projects the size of the next prompt.

    The projection is the last call's real prompt + completion size (what the
    conversation weighed at that point), plus an estimate for whatever has
    been added since (tool results), minus what compaction removed.
    """

    def __init__(self, stats: ScanStats) -> None:
        self.stats = stats
        self._baseline_tokens = 0
        self._pending_chars = 0

    def add_chars(self, n: int) -> None:
        self._pending_chars += n

    def projected_prompt_tokens(self) -> int:
        return self._baseline_tokens + max(self._pending_chars, 0) // CHARS_PER_TOKEN_ESTIMATE

    def record(self, response: LLMResponse) -> None:
        if response.usage is not None:
            prompt, completion = response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            self.stats.tokens_estimated = True
            prompt = self.projected_prompt_tokens()
            produced = len(response.content or "") + sum(len(c.name) + len(c.arguments) for c in response.tool_calls)
            completion = max(1, produced // CHARS_PER_TOKEN_ESTIMATE)
        self.stats.prompt_tokens += prompt
        self.stats.completion_tokens += completion
        self.stats.max_prompt_tokens_seen = max(self.stats.max_prompt_tokens_seen, prompt)
        self._baseline_tokens = prompt + completion
        self._pending_chars = 0

    @property
    def total(self) -> int:
        return self.stats.prompt_tokens + self.stats.completion_tokens


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
    on_nudge: Callable[[int, str], None] | None = None,
    now: Callable[[], float] = time.monotonic,
) -> ScanResult:
    """Run one explore-then-explain scan; return the summary, findings, and stats."""
    settings = settings or get_settings()
    max_steps = max_steps if max_steps is not None else settings.max_steps
    tool_timeout = tool_timeout if tool_timeout is not None else settings.tool_timeout_seconds
    scan_timeout = scan_timeout if scan_timeout is not None else settings.scan_timeout_seconds

    stats = ScanStats()
    meter = _TokenMeter(stats)
    start_time = now()

    def finish(summary: str, reason: str) -> ScanResult:
        if settings.forced_report:
            _forced_report(provider, messages, meter, settings.max_prompt_tokens, repo_tools, compactable, stats)
        stats.finish_reason = reason
        stats.duration_seconds = max(now() - start_time, 0.0)
        return ScanResult(summary=summary, findings=list(repo_tools.findings), stats=stats)

    root_listing = repo_tools.list_directory("")
    # Approach 3: fixed candidate searches by the loop itself, not a tool or
    # a step -- their matching lines go straight into the first message.
    candidates = repo_tools.candidate_search(CANDIDATE_TERMS) if settings.candidate_searches else ""
    messages: list[Message] = [
        Message(role="system", content=prompts.SYSTEM_PROMPT),
        Message(
            role="user",
            content=prompts.initial_user_message(ref.owner, ref.repo, branch, language, root_listing, candidates),
        ),
    ]
    meter.add_chars(sum(len(m.content or "") for m in messages) + len(json.dumps(TOOLS)))

    # tool_call_id -> (tool name, read_file path or None), so compaction can
    # tell repo_tools which seen lines are no longer visible to the model.
    compactable: dict[str, tuple[str, str | None]] = {}
    latest_step_call_ids: set[str] = set()

    def fallback(reason: str) -> ScanResult:
        summary = _finish_with_fallback(
            provider, messages, reason, meter, settings.max_prompt_tokens, repo_tools, compactable, stats
        )
        return finish(summary, reason)

    coverage = _ReviewCoverage()
    if settings.candidate_searches:
        # The loop already searched for both issue types on the model's behalf.
        coverage.checked_secrets = coverage.checked_sql = True
    seen_calls: set[tuple[str, str]] = set()
    for step in range(1, max_steps + 1):
        if now() - start_time > scan_timeout:
            return fallback(f"scan timeout ({scan_timeout}s) reached")
        if meter.total >= settings.max_total_tokens:
            return fallback(f"token budget ({settings.max_total_tokens} tokens) reached")
        if meter.projected_prompt_tokens() > settings.max_prompt_tokens:
            _compact(messages, meter, settings.max_prompt_tokens, repo_tools, compactable, latest_step_call_ids)
            if meter.projected_prompt_tokens() > settings.max_prompt_tokens:
                return fallback(f"context budget ({settings.max_prompt_tokens} tokens per prompt) reached")

        response = provider.chat(messages, tools=TOOLS)
        meter.record(response)
        stats.steps = step
        _record_text_findings(response.content, repo_tools, stats)

        if not response.tool_calls:
            if on_step:
                on_step(step, response, {})
            nudge = None
            if settings.agent_nudges and step < max_steps:
                nudge = _pick_nudge(response.content, coverage, stats)
            if nudge is None:
                # Keep the final answer in the conversation for the forced report step.
                messages.append(Message(role="assistant", content=response.content))
                return finish(response.content or "", "completed")
            # Don't accept this as the final answer: keep the reply in the
            # conversation and follow it with one corrective message.
            messages.append(Message(role="assistant", content=response.content))
            messages.append(Message(role="user", content=nudge))
            meter.add_chars(len(nudge))
            if on_nudge:
                on_nudge(step, nudge)
            continue

        messages.append(Message(role="assistant", content=response.content, tool_calls=response.tool_calls))
        tool_results: dict[str, str] = {}
        latest_step_call_ids = set()
        for call in response.tool_calls:
            result = _cap_tool_result(_execute_tool_call(repo_tools, call, tool_timeout), settings.max_tool_result_chars)

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

            args, _ = _parse_tool_arguments(call.arguments)
            read_path = args.get("path") if call.name == "read_file" and isinstance(args.get("path"), str) else None
            compactable[call.id] = (call.name, read_path)
            if call.name == "report_finding":
                _log_attempt(stats, args, "tool_call", result)

            coverage.observe(call, result)
            tool_results[call.id] = result
            latest_step_call_ids.add(call.id)
            messages.append(Message(role="tool", content=result, tool_call_id=call.id))
            meter.add_chars(len(result))

        if on_step:
            on_step(step, response, tool_results)

    return fallback(f"step limit ({max_steps} steps) reached")


TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS)
# Substrings of a search_code query that count as having looked for each
# v1 issue type (case-insensitive).
SECRET_SEARCH_TERMS = ("password", "passwd", "secret", "token", "key", "credential")
SQL_SEARCH_TERMS = ("select", "insert", "update", "delete", "sql", "query", "execute", "cursor", "where")


class _ReviewCoverage:
    """What the model has actually done so far, for the early-stop nudge: has
    it read a source file, and has it looked for each v1 issue type (by a
    related search_code query, or by reporting a finding of that type)?"""

    def __init__(self) -> None:
        self.read_source = False
        self.checked_secrets = False
        self.checked_sql = False

    def observe(self, call: ToolCall, result: str) -> None:
        args, err = _parse_tool_arguments(call.arguments)
        if err:
            return
        if call.name == "read_file":
            path = args.get("path")
            if (
                isinstance(path, str)
                and PurePosixPath(path).suffix.lower() in SOURCE_EXTENSIONS
                and not result.startswith("Error")
            ):
                self.read_source = True
        elif call.name == "search_code":
            query = str(args.get("query", "")).lower()
            self.checked_secrets |= any(term in query for term in SECRET_SEARCH_TERMS)
            self.checked_sql |= any(term in query for term in SQL_SEARCH_TERMS)
        elif call.name == "report_finding":
            issue_type = str(args.get("issue_type", "")).lower()
            self.checked_secrets |= "secret" in issue_type
            self.checked_sql |= "sql" in issue_type

    @property
    def complete(self) -> bool:
        return self.read_source and self.checked_secrets and self.checked_sql


def _pick_nudge(content: str | None, coverage: _ReviewCoverage, stats: ScanStats) -> str | None:
    """The corrective message for a reply with no tool calls, or None to
    accept it as the final answer. Each kind of nudge is sent at most once
    per scan, so a model that keeps failing still finishes."""
    if stats.text_tool_call_nudges == 0:
        tool_name = _find_text_tool_call(content)
        # Findings written as text are executed (_record_text_findings), so
        # only other tools written as text cost a nudge.
        if tool_name is not None and tool_name != "report_finding":
            stats.text_tool_call_nudges += 1
            return prompts.text_tool_call_message(tool_name)
    if stats.early_stop_nudges == 0 and not coverage.complete:
        stats.early_stop_nudges += 1
        return prompts.early_stop_message(coverage.read_source, coverage.checked_secrets, coverage.checked_sql)
    return None


def _find_text_tool_call(content: str | None) -> str | None:
    """If the reply contains a tool call (or a finding) written out as JSON
    text instead of a real tool call, return that tool's name.

    Small models do this in a few shapes: a ```json block, Qwen's
    <tool_call>...</tool_call> tags, or a bare finding object. Any JSON
    object in the text counts if it names a known tool ({"name": ...}) or
    looks like a finding (has issue_type, or severity + file)."""
    if not content or "{" not in content:
        return None
    decoder = json.JSONDecoder()
    for i, char in enumerate(content):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if isinstance(name, str) and name in TOOL_NAMES:
            return name
        if "issue_type" in obj or {"severity", "file"} <= obj.keys():
            return "report_finding"
    return None


def _cap_tool_result(result: str, max_chars: int) -> str:
    """Bound every tool result, whichever tool produced it, so one oversized
    result can't blow the context window on its own."""
    limit = max_chars + TOOL_RESULT_HEADROOM_CHARS
    if len(result) <= limit:
        return result
    return result[:max_chars] + f"\n... (tool result truncated to {max_chars} characters to fit the context window)"


def _compact(
    messages: list[Message],
    meter: _TokenMeter,
    budget: int,
    repo_tools: ToolExecutor,
    compactable: dict[str, tuple[str, str | None]],
    protected_call_ids: set[str],
    forget_seen: bool = True,
) -> None:
    """Replace the oldest tool results with a short marker until the projected
    prompt fits the budget. Results in protected_call_ids (the latest step's,
    which the model hasn't seen yet) are left alone. Message structure is
    unchanged -- every tool reply still answers its tool call -- so the
    conversation stays valid for the OpenAI-compatible API.

    forget_seen: mid-scan, a compacted read/search result stops counting as
    seen evidence (the model can still re-read it). For the final fallback
    and forced-report calls it must be False: there's no further turn to
    re-read, and those lines *were* seen earlier in this scan -- forgetting
    them there rejected every forced_json finding (round-3 bug, NOTES.md)."""
    for message in messages:
        if meter.projected_prompt_tokens() <= budget:
            return
        if message.role != "tool" or message.tool_call_id in protected_call_ids:
            continue
        if message.content == COMPACTED_TOOL_RESULT:
            continue
        meter.add_chars(len(COMPACTED_TOOL_RESULT) - len(message.content or ""))
        message.content = COMPACTED_TOOL_RESULT
        meter.stats.compactions += 1
        if not forget_seen:
            continue
        tool_name, read_path = compactable.get(message.tool_call_id or "", ("", None))
        if read_path is not None:
            repo_tools.forget_shown(read_path)
        elif tool_name == "search_code":
            repo_tools.forget_search_lines()


def _finish_with_fallback(
    provider: LLMProvider,
    messages: list[Message],
    reason: str,
    meter: _TokenMeter,
    budget: int,
    repo_tools: ToolExecutor,
    compactable: dict[str, tuple[str, str | None]],
    stats: ScanStats,
) -> str:
    """Force one last tool-less reply summarizing whatever's been learned so far.

    The instruction and reply are appended to `messages`, so the forced
    report step that follows sees them too."""
    instruction = Message(role="user", content=prompts.fallback_message(reason))
    meter.add_chars(len(instruction.content or "") - len(json.dumps(TOOLS)))  # no tool schemas this time
    if meter.projected_prompt_tokens() > budget:
        # Nothing is protected here: the model gets no further turn to use
        # a tool result anyway, so everything is fair game to fit the window.
        _compact(messages, meter, budget, repo_tools, compactable, protected_call_ids=set(), forget_seen=False)

    messages.append(instruction)
    try:
        response = provider.chat(messages)
        meter.record(response)
        summary = response.content or "(the model did not return a summary)"
        messages.append(Message(role="assistant", content=response.content))
        _record_text_findings(response.content, repo_tools, stats)
    except Exception as exc:  # best-effort: still return *something* if even this call fails
        summary = f"(could not produce a final summary: {exc})"
    return f"[Scan ended: {reason}.]\n\n{summary}"


def _forced_report(
    provider: LLMProvider,
    messages: list[Message],
    meter: _TokenMeter,
    budget: int,
    repo_tools: ToolExecutor,
    compactable: dict[str, tuple[str, str | None]],
    stats: ScanStats,
) -> None:
    """Approach 2: one schema-constrained JSON call listing the findings, each
    recorded through report_finding (source="forced_json"), plus at most one
    correction round if some were rejected. Never raises."""
    pending: str | None = prompts.forced_report_message()
    while pending is not None and stats.forced_report_rounds < MAX_FORCED_REPORT_ROUNDS:
        messages.append(Message(role="user", content=pending))
        meter.add_chars(len(pending))
        if meter.projected_prompt_tokens() > budget:
            _compact(messages, meter, budget, repo_tools, compactable, protected_call_ids=set(), forget_seen=False)
        try:
            response = provider.chat(messages, response_format=FORCED_REPORT_FORMAT)
        except Exception as exc:
            stats.forced_report_note = f"forced report call failed: {exc}"
            return
        meter.record(response)
        stats.forced_report_rounds += 1
        messages.append(Message(role="assistant", content=response.content))

        items = _parse_forced_report(response.content)
        if items is None:
            stats.forced_report_note = f"round {stats.forced_report_rounds}: no valid JSON in the reply"
            return
        rejected = []
        for args in items:
            result = _log_attempt(stats, args, "forced_json", repo_tools.report_finding(args, source="forced_json"))
            if result.startswith("Error"):
                where = f"{args.get('file')} lines {args.get('line_start')}-{args.get('line_end')}"
                rejected.append(f"{where}: {' '.join(result.split())}")
        stats.forced_report_note = (
            f"round {stats.forced_report_rounds}: {len(items)} finding(s) listed, {len(rejected)} rejected"
        )
        pending = prompts.forced_report_correction_message(rejected) if rejected else None


def _parse_forced_report(content: str | None) -> list[dict] | None:
    """Finding dicts from the forced report reply, or None if it held no valid JSON."""
    try:
        return findings_in_json(json.loads(content or ""))
    except json.JSONDecodeError:
        found = extract_text_findings(content)
        return found or None


def _record_text_findings(content: str | None, repo_tools: ToolExecutor, stats: ScanStats) -> None:
    """Approach 1: findings the model wrote as text in a reply go through the
    same report_finding checks as a real tool call (source="parsed_text")."""
    for args in extract_text_findings(content):
        _log_attempt(stats, args, "parsed_text", repo_tools.report_finding(args, source="parsed_text"))


def _log_attempt(stats: ScanStats, args: dict, source: str, result: str) -> str:
    """Record one report_finding attempt (for the per-path breakdown) and pass the result through."""
    stats.finding_attempts.append(
        {
            "source": source,
            "issue_type": args.get("issue_type"),
            "file": args.get("file"),
            "line_start": args.get("line_start"),
            "line_end": args.get("line_end"),
            "outcome": " ".join(result.split())[:400],
        }
    )
    return result


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


def _parse_tool_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
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
    if name == "get_dependencies":
        return repo_tools.get_dependencies()
    if name == "search_code":
        return repo_tools.search_code(args.get("query", ""))
    if name == "report_finding":
        # The whole argument object is the finding; RepoTools validates it.
        return repo_tools.report_finding(args)
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
