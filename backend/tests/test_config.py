"""Tests for the M2b settings in app/config.py (env parsing and defaults)."""

from __future__ import annotations

import pytest

from app import config


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    # Only process env counts in these tests, not the developer's .env file.
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in (
        "AGENT_NUDGES", "FORCED_REPORT", "CANDIDATE_SEARCHES", "SCAN_TIMEOUT_SECONDS", "MAX_PROMPT_TOKENS", "MAX_TOTAL_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_reflect_local_ollama_reality():
    settings = config.get_settings()

    assert settings.scan_timeout_seconds == 400.0
    assert settings.max_prompt_tokens == 3300
    assert settings.max_total_tokens == 60_000
    assert settings.agent_nudges is True
    assert settings.forced_report is True
    assert settings.candidate_searches is True


def test_candidate_searches_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("CANDIDATE_SEARCHES", "off")

    assert config.get_settings().candidate_searches is False


def test_forced_report_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("FORCED_REPORT", "false")

    assert config.get_settings().forced_report is False


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", " OFF "])
def test_agent_nudges_can_be_turned_off(monkeypatch, value):
    monkeypatch.setenv("AGENT_NUDGES", value)

    assert config.get_settings().agent_nudges is False


@pytest.mark.parametrize("value", ["true", "1", "yes"])
def test_agent_nudges_on_values(monkeypatch, value):
    monkeypatch.setenv("AGENT_NUDGES", value)

    assert config.get_settings().agent_nudges is True


def test_scan_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("SCAN_TIMEOUT_SECONDS", "900")

    assert config.get_settings().scan_timeout_seconds == 900.0
