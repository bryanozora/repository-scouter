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
        max_steps=int(os.getenv("MAX_STEPS", "12")),
        max_file_bytes=int(os.getenv("MAX_FILE_BYTES", "100000")),
        max_tool_result_chars=int(os.getenv("MAX_TOOL_RESULT_CHARS", "6000")),
    )
