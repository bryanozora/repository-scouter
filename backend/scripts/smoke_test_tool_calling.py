"""Tool-calling smoke test for local Ollama models.

Gives a model ONE tool (`list_directory`) and checks, over N attempts per
scenario, whether it (a) calls the tool, (b) emits valid JSON arguments,
(c) passes the right `path`, and how long each call takes.
Results are printed and appended to docs/NOTES.md.

Standalone on purpose: it only needs httpx + python-dotenv and talks to the
OpenAI-compatible endpoint directly, so it can run before the provider
abstraction exists.

Usage (from backend/):
    python scripts/smoke_test_tool_calling.py
    python scripts/smoke_test_tool_calling.py --models qwen2.5:7b-instruct --attempts 3
"""

import argparse
import json
import os
import statistics
import time
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES_PATH = REPO_ROOT / "docs" / "NOTES.md"

DEFAULT_MODELS = ["qwen2.5:7b-instruct", "qwen3:8b"]
TEMPERATURE = 0.2
REQUEST_TIMEOUT = 300.0  # first call may include model load time

SYSTEM_PROMPT = (
    "You are a code exploration assistant. "
    "Use the provided tool to inspect the repository. Do not guess folder contents."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and folders inside a directory of the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the repo root. Use '' for the root.",
                    }
                },
                "required": ["path"],
            },
        },
    }
]

# Two scenarios with different expected `path` arguments, so we test whether
# the model copies varied arguments from the prompt rather than one fixed string.
SCENARIOS = [
    {"name": "short-path", "prompt": "What files are in the src folder?", "expected_path": "src"},
    {
        "name": "nested-path",
        "prompt": "Show me what is inside backend/app/agent.",
        "expected_path": "backend/app/agent",
    },
]


def normalize_path(p: str) -> str:
    """'./src/' -> 'src' so harmless formatting differences don't count as errors."""
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def ollama_root(base_url: str) -> str:
    """http://localhost:11434/v1 -> http://localhost:11434 (for Ollama-native endpoints)."""
    return base_url.rstrip("/").removesuffix("/v1")


def one_attempt(client: httpx.Client, base_url: str, model: str, prompt: str, expected: str) -> dict:
    """Run one request and classify the outcome."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "tools": TOOLS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    result = {"called": False, "json_valid": False, "path_ok": False, "seconds": 0.0, "note": ""}
    start = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/chat/completions", json=payload)
        result["seconds"] = time.perf_counter() - start
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        result["seconds"] = time.perf_counter() - start
        result["note"] = f"request failed: {exc}"
        return result

    calls = message.get("tool_calls") or []
    if not calls:
        # Common small-model failure: describing the call as plain text instead.
        result["note"] = "no tool_calls (answered in text)"
        return result

    fn = calls[0].get("function", {})
    if fn.get("name") != "list_directory":
        result["note"] = f"wrong tool name: {fn.get('name')!r}"
        return result
    result["called"] = True

    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            result["note"] = "arguments not valid JSON"
            return result
    if not isinstance(args, dict) or not isinstance(args.get("path"), str):
        result["note"] = f"arguments missing string 'path': {args!r}"
        return result
    result["json_valid"] = True

    if normalize_path(args["path"]) == expected:
        result["path_ok"] = True
    else:
        result["note"] = f"wrong path: {args['path']!r} (expected {expected!r})"
    return result


def run_model(client: httpx.Client, base_url: str, model: str, attempts: int) -> list[dict]:
    """Return one summary row per scenario for this model."""
    print(f"\n=== {model} ===")
    print("warm-up call (not counted)...")
    one_attempt(client, base_url, model, SCENARIOS[0]["prompt"], SCENARIOS[0]["expected_path"])

    rows = []
    for sc in SCENARIOS:
        results = []
        for i in range(attempts):
            r = one_attempt(client, base_url, model, sc["prompt"], sc["expected_path"])
            results.append(r)
            status = "ok " if r["path_ok"] else "BAD"
            print(f"  [{sc['name']}] {i + 1:>2}/{attempts} {status} {r['seconds']:.1f}s {r['note']}")
        times = [r["seconds"] for r in results]
        rows.append(
            {
                "model": model,
                "scenario": sc["name"],
                "n": attempts,
                "called": sum(r["called"] for r in results),
                "json_valid": sum(r["json_valid"] for r in results),
                "path_ok": sum(r["path_ok"] for r in results),
                "avg_s": statistics.mean(times),
                "median_s": statistics.median(times),
                "notes": sorted({r["note"] for r in results if r["note"]}),
            }
        )
    return rows


def render_markdown(rows: list[dict], ollama_version: str, skipped: list[str], attempts: int) -> str:
    lines = [
        f"## Tool-calling smoke test — {date.today().isoformat()}",
        "",
        f"Ollama {ollama_version}, temperature {TEMPERATURE}, {attempts} attempts per scenario, "
        "one `list_directory(path)` tool, one warm-up call excluded from timings.",
        "",
        "| Model | Scenario | Tool called | Valid JSON | Correct path | Avg s | Median s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["n"]
        lines.append(
            f"| `{r['model']}` | {r['scenario']} | {r['called']}/{n} | {r['json_valid']}/{n} "
            f"| {r['path_ok']}/{n} | {r['avg_s']:.1f} | {r['median_s']:.1f} |"
        )
    failures = [(r["model"], r["scenario"], r["notes"]) for r in rows if r["notes"]]
    if failures:
        lines += ["", "Failure notes:"]
        for model, scenario, notes in failures:
            lines.append(f"- `{model}` / {scenario}: " + "; ".join(notes))
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

    load_dotenv(REPO_ROOT / ".env")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    root = ollama_root(base_url)

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
            rows.extend(run_model(client, base_url, model, args.attempts))

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
