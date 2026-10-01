"""Tests for LocalRepoSource (app/local_source.py): a read-only stand-in for
github_client that serves a local directory, used to scan the planted-bug
fixtures in evals/repos/ without pushing them to GitHub."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.tools import RepoTools
from app.config import Settings
from app.github_client import GitHubError, GitHubNotFoundError, RepoRef, TreeEntry
from app.local_source import LocalRepoSource

REF = RepoRef(owner="local", repo="fixture")
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "evals" / "repos"


def make_settings() -> Settings:
    return Settings(
        llm_provider="ollama",
        llm_model="m",
        ollama_base_url="http://x",
        github_token=None,
        max_steps=5,
        max_file_bytes=100_000,
        max_tool_result_chars=6000,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    # write_bytes, not write_text: on Windows write_text turns "\n" into "\r\n".
    (tmp_path / "src" / "app.py").write_bytes(b"print('hi')\n")
    (tmp_path / "README.md").write_bytes(b"# demo\n")
    return tmp_path


def test_list_tree_returns_dirs_and_files_with_posix_paths_and_sizes(repo):
    tree = LocalRepoSource(repo).list_tree(REF, "local")

    assert tree.truncated is False
    assert TreeEntry(path="src", type="tree") in tree.entries
    assert TreeEntry(path="src/app.py", type="blob", size=len("print('hi')\n")) in tree.entries
    assert TreeEntry(path="README.md", type="blob", size=len("# demo\n")) in tree.entries


def test_list_tree_never_includes_a_git_directory(repo):
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("x", encoding="utf-8")

    paths = [e.path for e in LocalRepoSource(repo).list_tree(REF, "local").entries]

    assert not any(p == ".git" or p.startswith(".git/") for p in paths)


def test_read_file_returns_text(repo):
    assert LocalRepoSource(repo).read_file(REF, "src/app.py") == "print('hi')\n"


def test_read_file_missing_path_raises_not_found(repo):
    with pytest.raises(GitHubNotFoundError):
        LocalRepoSource(repo).read_file(REF, "nope.py")


def test_read_file_directory_raises(repo):
    with pytest.raises(GitHubError):
        LocalRepoSource(repo).read_file(REF, "src")


@pytest.mark.parametrize("path", ["../outside.txt", "src/../../outside.txt"])
def test_read_file_refuses_to_escape_the_root(repo, path):
    (repo.parent / "outside.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(GitHubError):
        LocalRepoSource(repo).read_file(REF, path)


def test_missing_root_raises_a_clear_error(tmp_path):
    with pytest.raises(ValueError):
        LocalRepoSource(tmp_path / "does-not-exist")


def test_repo_tools_can_run_on_a_local_source(repo):
    rt = RepoTools(REF, branch="local", settings=make_settings(), source=LocalRepoSource(repo))

    assert "dir  src" in rt.list_directory("")
    assert "1: print('hi')" in rt.read_file("src/app.py")
    assert "src/app.py:1" in rt.search_code("print")


def test_planted_bug_fixture_end_to_end_with_report_finding():
    rt = RepoTools(REF, branch="local", settings=make_settings(), source=LocalRepoSource(FIXTURES_DIR / "py-notes-api"))
    rt.read_file("db.py", start=20, end=27)

    result = rt.report_finding(
        {
            "severity": "high",
            "category": "security",
            "issue_type": "sql_injection",
            "file": "db.py",
            "line_start": 24,
            "line_end": 25,
            "description": "owner and term are formatted into the SQL string.",
            "suggestion": "Use ? placeholders.",
            "confidence": 0.9,
        }
    )

    assert result.startswith("Recorded finding #1 (verified")


# Verbatim from the clean-control re-run (docs/NOTES.md): two findings the
# model wrote as text, both false positives on parameterized queries.
CLEAN_CONTROL_TEXT_FINDINGS = (
    '```json\n{"name": "report_finding", "arguments": {"category": "security", "confidence": 0.9, '
    '"description": "Potential SQL injection vulnerability in the query construction.", '
    '"file": "inventory/db.py", "line_start": 14, "line_end": 15, "severity": "medium", '
    '"suggestion": "Use parameterized queries to prevent SQL injection."}}\n```\n\n'
    '```json\n{"name": "report_finding", "arguments": {"category": "security", "confidence": 0.9, '
    '"description": "Potential SQL injection vulnerability in the query construction.", '
    '"file": "inventory/db.py", "line_start": 30, "line_end": 31, "severity": "medium", '
    '"suggestion": "Use parameterized queries to prevent SQL injection."}}\n```'
)


def _clean_control_tools() -> RepoTools:
    source = LocalRepoSource(FIXTURES_DIR / "clean-control")
    return RepoTools(REF, branch="local", settings=make_settings(), source=source)


def test_clean_control_verbatim_text_findings_are_rejected_for_missing_issue_type():
    # As the model actually wrote them, they have no issue_type at all, so
    # validation rejects them before read-gating or the evidence check run.
    from app.agent.findings import extract_text_findings

    rt = _clean_control_tools()
    rt.read_file("inventory/db.py")

    results = [rt.report_finding(args, source="parsed_text") for args in extract_text_findings(CLEAN_CONTROL_TEXT_FINDINGS)]

    assert len(results) == 2
    assert all(r.startswith("Error: report_finding rejected") and "issue_type: missing" in r for r in results)
    assert rt.findings == []


def test_clean_control_false_positives_with_issue_type_are_flagged_by_the_evidence_check():
    # The same two false positives, completed with issue_type (as the forced
    # JSON step's schema guarantees): lines seen, so they reach the evidence
    # check -- which flags them UNVERIFIED (parameterized queries).
    from app.agent.findings import extract_text_findings

    rt = _clean_control_tools()
    rt.read_file("inventory/db.py")

    results = [
        rt.report_finding({**args, "issue_type": "sql_injection"}, source="parsed_text")
        for args in extract_text_findings(CLEAN_CONTROL_TEXT_FINDINGS)
    ]

    assert all("UNVERIFIED" in r for r in results)
    assert [f.verified for f in rt.findings] == [False, False]
    assert all("no SQL query built with string formatting" in f.verification_note for f in rt.findings)


def test_clean_control_false_positives_are_rejected_when_lines_were_never_seen():
    from app.agent.findings import extract_text_findings

    rt = _clean_control_tools()
    rt.search_code("SELECT")  # shows lines 14 and 30 only, not 15 or 31

    results = [
        rt.report_finding({**args, "issue_type": "sql_injection"}, source="parsed_text")
        for args in extract_text_findings(CLEAN_CONTROL_TEXT_FINDINGS)
    ]

    assert all(r.startswith("Error: report_finding rejected") and "you have only seen lines" in r for r in results)
    assert rt.findings == []
