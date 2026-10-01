"""Loads .env into a small Settings object.

Every field mirrors a variable in .env.example. Nothing here is
LLM-specific on purpose: config.py is the one place that reads os.environ,
so the rest of the app (llm.py, and later the agent loop) never touches
.env directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    llm_provider: str
    llm_model: str
    ollama_base_url: str
    github_token: str | None
    max_steps: int
    max_file_bytes: int
    max_tool_result_chars: int
    # Defaulted (rather than required) so existing Settings(...) call sites
    # that predate these two fields don't have to change.
    tool_timeout_seconds: float = 15.0
    # A full scan on local Ollama (8GB machine) takes several minutes: ~30-40s
    # per call, and the M2b real runs needed 3-8+ steps.
    scan_timeout_seconds: float = 400.0
    # Token guardrails (M2b). max_prompt_tokens must stay below the model's
    # *effective* context window -- measured at 4096 tokens for Ollama 0.10.1
    # defaults (see docs/NOTES.md); Ollama silently drops the start of a
    # longer prompt, system prompt included. 3300 leaves ~800 for the reply.
    max_total_tokens: int = 60_000
    max_prompt_tokens: int = 3300
    # The loop's corrective nudges for small-model failure modes (a tool call
    # written as text; stopping before reading source or searching for both
    # issue types). Switchable so evaluations can compare with/without.
    agent_nudges: bool = True
    # After the scan, one schema-constrained JSON call (response_format) asks
    # the model to list its findings; each goes through report_finding's checks.
    forced_report: bool = True
    # Before the model's first turn, the loop itself searches for a few fixed
    # terms (password, secret, token, key, SELECT, execute() and puts the
    # matching lines into the first message, so recall doesn't depend on which
    # searches a 7B model happens to choose.
    candidate_searches: bool = True


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def get_settings() -> Settings:
    """Read .env (repo root) plus process env into a Settings instance.

    Falls back to the same defaults as .env.example when a variable is
    unset, so the app still runs with sane values if .env is missing.
    """
    load_dotenv(REPO_ROOT / ".env")
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        github_token=os.getenv("GITHUB_TOKEN"),
        max_steps=int(os.getenv("MAX_STEPS", "15")),
        max_file_bytes=int(os.getenv("MAX_FILE_BYTES", "100000")),
        max_tool_result_chars=int(os.getenv("MAX_TOOL_RESULT_CHARS", "2500")),
        tool_timeout_seconds=float(os.getenv("TOOL_TIMEOUT_SECONDS", "15")),
        scan_timeout_seconds=float(os.getenv("SCAN_TIMEOUT_SECONDS", "400")),
        max_total_tokens=int(os.getenv("MAX_TOTAL_TOKENS", "60000")),
        max_prompt_tokens=int(os.getenv("MAX_PROMPT_TOKENS", "3300")),
        agent_nudges=_env_flag("AGENT_NUDGES", True),
        forced_report=_env_flag("FORCED_REPORT", True),
        candidate_searches=_env_flag("CANDIDATE_SEARCHES", True),
    )
