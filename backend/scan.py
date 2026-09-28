"""Run a full M1 agent scan against a public GitHub repo from the terminal.

Usage (from backend/):
    python scan.py <repo_url>

Wires together the LLM provider (app.llm), the GitHub repo access layer
(app.github_client), the read-only tools (app.agent.tools), and the
agent loop (app.agent.loop) into the one end-to-end path described by
PLAN.md's M1 milestone.
"""

from __future__ import annotations

import argparse
import sys

import httpx

from app.agent.loop import run_scan
from app.agent.tools import RepoTools
from app.config import get_settings
from app.github_client import GitHubError, get_repo_info, parse_repo_url
from app.llm import get_provider


def _print_step(step: int, response, tool_results: dict[str, str]) -> None:
    if not response.tool_calls:
        print(f"[step {step}] final answer")
        return
    for call in response.tool_calls:
        result = tool_results.get(call.id, "")
        preview = result if len(result) <= 200 else result[:200] + "... (truncated for display)"
        print(f"[step {step}] {call.name}({call.arguments}) -> {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a public GitHub repo and explain its architecture.")
    parser.add_argument("repo_url", help="e.g. github.com/pallets/click")
    args = parser.parse_args()

    settings = get_settings()

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

    print(f"[scan] {ref.owner}/{ref.repo}@{info.default_branch} ({info.language or 'unknown language'})")

    repo_tools = RepoTools(ref, branch=info.default_branch, token=settings.github_token, settings=settings)
    provider = get_provider(settings)

    try:
        summary = run_scan(
            ref,
            branch=info.default_branch,
            provider=provider,
            repo_tools=repo_tools,
            language=info.language,
            settings=settings,
            on_step=_print_step,
        )
    except httpx.HTTPError as exc:
        print(f"Error: could not reach the LLM provider (is Ollama running?): {exc}", file=sys.stderr)
        raise SystemExit(1)
    except GitHubError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print("\n=== Architecture summary ===\n")
    print(summary)


if __name__ == "__main__":
    main()
