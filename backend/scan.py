"""Run a full agent scan against a public GitHub repo (or a local fixture
directory) from the terminal.

Usage (from backend/):
    python scan.py <repo_url>
    python scan.py --local ../evals/repos/py-notes-api
    python scan.py <repo_url> --out result.json

Wires together the LLM provider (app.llm), the repo access layer
(app.github_client, or app.local_source for --local), the read-only tools
(app.agent.tools), and the agent loop (app.agent.loop). Prints the
architecture summary, verified findings (sorted by severity), unverified
findings separately, and run statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from app.agent.loop import ScanResult, run_scan
from app.agent.tools import RepoTools
from app.config import get_settings
from app.github_client import GitHubError, RepoRef, get_repo_info, parse_repo_url
from app.llm import get_provider
from app.local_source import LocalRepoSource


def _print_step(step: int, response, tool_results: dict[str, str]) -> None:
    if not response.tool_calls:
        print(f"[step {step}] text reply (no tool call)")
        return
    for call in response.tool_calls:
        result = tool_results.get(call.id, "")
        preview = result if len(result) <= 200 else result[:200] + "... (truncated for display)"
        print(f"[step {step}] {call.name}({call.arguments}) -> {preview}")


def _print_nudge(step: int, message: str) -> None:
    print(f"[step {step}] nudge -> {message}")


def _print_report(result: ScanResult) -> None:
    report = result.to_dict()
    stats = report["stats"]

    print("\n=== Architecture summary ===\n")
    print(report["summary"])

    print(f"\n=== Findings ({len(report['findings'])} verified) ===\n")
    print(json.dumps(report["findings"], indent=2) if report["findings"] else "(none)")

    if report["unverified_findings"]:
        print(f"\n=== Unverified findings ({len(report['unverified_findings'])}, evidence did not match) ===\n")
        print(json.dumps(report["unverified_findings"], indent=2))

    estimated = " (estimated)" if stats["tokens_estimated"] else ""
    print("\n=== Run stats ===\n")
    print(f"finished:   {stats['finish_reason']}")
    print(f"steps:      {stats['steps']}")
    print(f"tokens:     {stats['prompt_tokens']} prompt + {stats['completion_tokens']} completion{estimated}")
    print(f"max prompt: {stats['max_prompt_tokens_seen']} tokens, compactions: {stats['compactions']}")
    print(f"nudges:     {stats['text_tool_call_nudges']} text-tool-call, {stats['early_stop_nudges']} early-stop")
    print(f"forced:     {stats['forced_report_rounds']} round(s); {stats['forced_report_note'] or '-'}")

    print(f"\n=== Finding attempts ({len(stats['finding_attempts'])}) ===\n")
    for a in stats["finding_attempts"] or []:
        where = f"{a['file']}:{a['line_start']}-{a['line_end']}"
        print(f"[{a['source']}] {a['issue_type']} {where} -> {a['outcome'][:200]}")
    if not stats["finding_attempts"]:
        print("(none)")
    print(f"duration:   {stats['duration_seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a repo, explain its architecture, and report findings.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("repo_url", nargs="?", help="e.g. github.com/pallets/click")
    target.add_argument("--local", metavar="DIR", help="scan a local directory instead (e.g. an evals/repos fixture)")
    parser.add_argument("--out", metavar="FILE", help="also write the full report as JSON to FILE")
    args = parser.parse_args()

    settings = get_settings()

    if args.local:
        try:
            source = LocalRepoSource(args.local)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)
        ref, branch, language = RepoRef(owner="local", repo=source.root.name), "local", None
        repo_tools = RepoTools(ref, branch=branch, settings=settings, source=source)
    else:
        try:
            ref = parse_repo_url(args.repo_url)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)
        try:
            info = get_repo_info(ref, token=settings.github_token)
        except GitHubError as exc:
            print(f"Error: could not read repo info from GitHub: {exc}", file=sys.stderr)
            raise SystemExit(1)
        branch, language = info.default_branch, info.language
        repo_tools = RepoTools(ref, branch=branch, token=settings.github_token, settings=settings)

    print(f"[scan] {ref.owner}/{ref.repo}@{branch} ({language or 'unknown language'}) model={settings.llm_model}")

    try:
        result = run_scan(
            ref,
            branch=branch,
            provider=get_provider(settings),
            repo_tools=repo_tools,
            language=language,
            settings=settings,
            on_step=_print_step,
            on_nudge=_print_nudge,
        )
    except httpx.HTTPError as exc:
        print(f"Error: could not reach the LLM provider (is Ollama running?): {exc}", file=sys.stderr)
        raise SystemExit(1)
    except GitHubError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    _print_report(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\n[scan] report written to {args.out}")


if __name__ == "__main__":
    main()
