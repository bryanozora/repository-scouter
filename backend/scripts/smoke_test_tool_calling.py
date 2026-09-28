"""Tool-calling smoke test for local Ollama models.

M1 version: a single-tool sanity check. M2a version (this one): compares
tool-*selection* accuracy between two tool-set sizes -- 2 tools
(list_directory, read_file) vs all 4 (+ get_dependencies, search_code) --
on the same scenarios, to check whether adding tools makes the model worse
at picking the right one. Uses the real tool schemas from app.agent.tools,
not hand-rolled ones, so this measures what the agent loop actually sends.

Uses the LLM provider abstraction (app.llm.OllamaProvider) for the actual
chat calls, so this script exercises the same code path the agent loop
will use. It still talks to Ollama's native /api/version and /api/tags
endpoints directly with httpx, since those aren't part of the chat
provider abstraction.

Usage (from backend/):
    python scripts/smoke_test_tool_calling.py
    python scripts/smoke_test_tool_calling.py --models qwen2.5:7b-instruct --attempts 3
"""

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES_PATH = REPO_ROOT / "docs" / "NOTES.md"

sys.path.insert(0, str(REPO_ROOT / "backend"))
from app.agent.tools import (  # noqa: E402
    GET_DEPENDENCIES_SCHEMA,
    LIST_DIRECTORY_SCHEMA,
    READ_FILE_SCHEMA,
    SEARCH_CODE_SCHEMA,
)
from app.config import get_settings  # noqa: E402
from app.llm import DEFAULT_TEMPERATURE, OllamaProvider  # noqa: E402
from app.models import Message  # noqa: E402

# Only the production default -- this is about tool-set size, not a
# model comparison (that's already recorded from the M1 smoke test above).
DEFAULT_MODELS = ["qwen2.5:7b-instruct"]
REQUEST_TIMEOUT = 300.0  # first call may include model load time

SYSTEM_PROMPT = (
    "You are a code exploration assistant. "
    "Use the provided tools to inspect the repository. Do not guess folder contents."
)

TOOL_SETS = {
    "2-tool": [LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA],
    "4-tool": [LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA, GET_DEPENDENCIES_SCHEMA, SEARCH_CODE_SCHEMA],
}


def normalize_path(p: str) -> str:
    """'./src/' -> 'src' so harmless formatting differences don't count as errors."""
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def ollama_root(base_url: str) -> str:
    """http://localhost:11434/v1 -> http://localhost:11434 (for Ollama-native endpoints)."""
    return base_url.rstrip("/").removesuffix("/v1")


def _expect_list_directory_path(expected_path: str):
    def check(args: dict) -> bool:
        path = args.get("path")
        return isinstance(path, str) and normalize_path(path) == expected_path

    return check


def _expect_query_containing(substring: str):
    def check(args: dict) -> bool:
        query = args.get("query")
        return isinstance(query, str) and substring.lower() in query.lower()

    return check


def _expect_any_args(args: dict) -> bool:
    return True  # get_dependencies takes no meaningful args


# Two scenarios carried over from M1 (list_directory, meaningful with either
# tool-set size) plus two new ones (only meaningful with all 4 tools
# available, since the "correct" tool doesn't exist in the 2-tool set).
SCENARIOS = [
    {
        "name": "short-path",
        "prompt": "What files are in the src folder?",
        "expected_tool": "list_directory",
        "check_args": _expect_list_directory_path("src"),
        "tool_sets": ["2-tool", "4-tool"],
    },
    {
        "name": "nested-path",
        "prompt": "Show me what is inside backend/app/agent.",
        "expected_tool": "list_directory",
        "check_args": _expect_list_directory_path("backend/app/agent"),
        "tool_sets": ["2-tool", "4-tool"],
    },
    {
        "name": "dependencies",
        "prompt": "What are this project's dependencies?",
        "expected_tool": "get_dependencies",
        "check_args": _expect_any_args,
        "tool_sets": ["4-tool"],
    },
    {
        "name": "search-term",
        "prompt": "Find every place in the code that mentions the word 'password'.",
        "expected_tool": "search_code",
        "check_args": _expect_query_containing("password"),
        "tool_sets": ["4-tool"],
    },
]


def one_attempt(provider: OllamaProvider, prompt: str, expected_tool: str, check_args, tools: list[dict]) -> dict:
    """Run one request through the provider and classify the outcome."""
    messages = [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=prompt)]
    result = {"called": False, "json_valid": False, "args_ok": False, "seconds": 0.0, "note": ""}
    start = time.perf_counter()
    try:
        response = provider.chat(messages, tools=tools)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        result["seconds"] = time.perf_counter() - start
        result["note"] = f"request failed: {exc}"
        return result
    result["seconds"] = time.perf_counter() - start

    if not response.tool_calls:
        # Common small-model failure: describing the call as plain text instead.
        result["note"] = "no tool_calls (answered in text)"
        return result

    call = response.tool_calls[0]
    if call.name != expected_tool:
        result["note"] = f"wrong tool: called {call.name!r}, expected {expected_tool!r}"
        return result
    result["called"] = True

    try:
        args = json.loads(call.arguments) if call.arguments else {}
    except json.JSONDecodeError:
        result["note"] = "arguments not valid JSON"
        return result
    if not isinstance(args, dict):
        result["note"] = f"arguments not a JSON object: {args!r}"
        return result
    result["json_valid"] = True

    if check_args(args):
        result["args_ok"] = True
    else:
        result["note"] = f"args did not match expectation: {args!r}"
    return result


def run_scenario(provider: OllamaProvider, scenario: dict, tool_set_name: str, model: str, attempts: int) -> dict:
    """Return one summary row for this scenario run under one tool-set size."""
    tools = TOOL_SETS[tool_set_name]
    print(f"\n=== {model} / {tool_set_name} / {scenario['name']} ===")
    results = []
    for i in range(attempts):
        r = one_attempt(provider, scenario["prompt"], scenario["expected_tool"], scenario["check_args"], tools)
        results.append(r)
        status = "ok " if r["args_ok"] else "BAD"
        print(f"  {i + 1:>2}/{attempts} {status} {r['seconds']:.1f}s {r['note']}")
    times = [r["seconds"] for r in results]
    return {
        "model": model,
        "scenario": scenario["name"],
        "tool_set": tool_set_name,
        "n": attempts,
        "called": sum(r["called"] for r in results),
        "json_valid": sum(r["json_valid"] for r in results),
        "args_ok": sum(r["args_ok"] for r in results),
        "avg_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "notes": sorted({r["note"] for r in results if r["note"]}),
    }


def run_model(base_url: str, model: str, attempts: int) -> list[dict]:
    """Return one summary row per (scenario, tool-set) combination for this model."""
    provider = OllamaProvider(base_url=base_url, model=model, timeout=REQUEST_TIMEOUT)
    print(f"\n### {model} ###")
    print("warm-up call (not counted)...")
    warmup = SCENARIOS[0]
    one_attempt(provider, warmup["prompt"], warmup["expected_tool"], warmup["check_args"], TOOL_SETS["2-tool"])

    rows = []
    for scenario in SCENARIOS:
        for tool_set_name in scenario["tool_sets"]:
            rows.append(run_scenario(provider, scenario, tool_set_name, model, attempts))
    return rows


def render_markdown(rows: list[dict], ollama_version: str, skipped: list[str], attempts: int) -> str:
    lines = [
        f"## Tool-set-size smoke test (M2a) — {date.today().isoformat()}",
        "",
        f"Ollama {ollama_version}, temperature {DEFAULT_TEMPERATURE}, {attempts} attempts per "
        "(scenario, tool-set) combination, one warm-up call excluded from timings. Tool schemas "
        "are the real ones from app.agent.tools, not hand-rolled. `short-path`/`nested-path` run "
        "under both tool-set sizes (the accuracy comparison); `dependencies`/`search-term` only "
        "make sense with all 4 tools available.",
        "",
        "| Model | Tool set | Scenario | Tool called | Valid JSON | Args OK | Avg s | Median s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["n"]
        lines.append(
            f"| `{r['model']}` | {r['tool_set']} | {r['scenario']} | {r['called']}/{n} | {r['json_valid']}/{n} "
            f"| {r['args_ok']}/{n} | {r['avg_s']:.1f} | {r['median_s']:.1f} |"
        )
    failures = [(r["model"], r["tool_set"], r["scenario"], r["notes"]) for r in rows if r["notes"]]
    if failures:
        lines += ["", "Failure notes:"]
        for model, tool_set, scenario, notes in failures:
            lines.append(f"- `{model}` / {tool_set} / {scenario}: " + "; ".join(notes))
    for s in skipped:
        lines += ["", f"Skipped: {s}"]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--no-write", action="store_true", help="print only, don't touch docs/NOTES.md")
    args = parser.parse_args()

    base_url = get_settings().ollama_base_url.rstrip("/")
    root = ollama_root(base_url)

    # Native Ollama endpoints (not part of the OpenAI-compatible chat API,
    # so not covered by the provider abstraction) -- used only to check
    # what's installed before running the chat smoke test through it.
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        try:
            version = client.get(f"{root}/api/version").json().get("version", "unknown")
            installed = {m["name"] for m in client.get(f"{root}/api/tags").json().get("models", [])}
        except httpx.HTTPError as exc:
            raise SystemExit(f"Cannot reach Ollama at {root}: {exc}\nIs `ollama serve` running?")

    rows, skipped = [], []
    for model in args.models:
        if model not in installed:
            msg = f"`{model}` is not installed (run `ollama pull {model}`)."
            print(f"\nSkipping {model}: not installed")
            skipped.append(msg)
            continue
        rows.extend(run_model(base_url, model, args.attempts))

    if not rows:
        raise SystemExit("No models were tested.")

    report = render_markdown(rows, version, skipped, args.attempts)
    print("\n" + report)
    if not args.no_write:
        NOTES_PATH.parent.mkdir(exist_ok=True)
        existing = NOTES_PATH.read_text(encoding="utf-8") if NOTES_PATH.exists() else "# Notes\n\n"
        NOTES_PATH.write_text(existing.rstrip() + "\n\n" + report, encoding="utf-8")
        print(f"Appended results to {NOTES_PATH}")


if __name__ == "__main__":
    main()
