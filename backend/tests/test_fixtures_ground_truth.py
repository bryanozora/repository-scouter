"""Ground-truth check for the planted-bug fixtures in evals/repos/.

Ground truth lives in evals/expected/<fixture>.json -- deliberately *outside*
the scanned fixture directory, so the agent can never read the answer key
(an early real run did exactly that via search_code). Every planted bug
listed there must pass check_evidence, and every decoy must fail it.
This tests the evidence checker against realistic multi-file code, and it
also protects the fixtures themselves: if a fixture file is edited and a
line number in the ground truth goes stale, this fails now instead of
silently skewing an M5 evaluation run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.findings import ISSUE_TYPES, check_evidence

EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"
FIXTURES_DIR = EVALS_DIR / "repos"
EXPECTED_DIR = EVALS_DIR / "expected"


def _expected_files() -> list[Path]:
    return sorted(EXPECTED_DIR.glob("*.json"))


def _cases(kind: str) -> list[tuple[str, dict]]:
    cases = []
    for expected_file in _expected_files():
        expected = json.loads(expected_file.read_text(encoding="utf-8"))
        for entry in expected[kind]:
            cases.append((expected_file.stem, entry))
    return cases


def _check(fixture: str, entry: dict) -> tuple[bool, str]:
    path = FIXTURES_DIR / fixture / entry["file"]
    lines = path.read_text(encoding="utf-8").splitlines()
    return check_evidence(entry["issue_type"], entry["file"], lines, entry["line_start"], entry["line_end"])


def test_every_fixture_has_ground_truth_and_vice_versa():
    fixtures = {p.name for p in FIXTURES_DIR.iterdir() if p.is_dir()}
    expected = {p.stem for p in _expected_files()}

    assert {"py-notes-api", "js-shop", "clean-control", "injection-bait"} <= fixtures
    assert fixtures == expected


def test_no_answer_key_inside_a_scanned_fixture():
    leaked = [p for p in FIXTURES_DIR.rglob("expected*.json")]

    assert leaked == []


@pytest.mark.parametrize("fixture,entry", _cases("bugs") + _cases("decoys"), ids=lambda v: v["id"] if isinstance(v, dict) else v)
def test_expected_entries_are_well_formed(fixture, entry):
    assert entry["issue_type"] in ISSUE_TYPES
    assert (FIXTURES_DIR / fixture / entry["file"]).is_file()
    assert 1 <= entry["line_start"] <= entry["line_end"]


def test_entry_ids_are_unique():
    ids = [entry["id"] for _, entry in _cases("bugs") + _cases("decoys")]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("fixture,entry", _cases("bugs"), ids=lambda v: v["id"] if isinstance(v, dict) else v)
def test_planted_bug_passes_evidence_check(fixture, entry):
    ok, note = _check(fixture, entry)

    assert ok, note


@pytest.mark.parametrize("fixture,entry", _cases("decoys"), ids=lambda v: v["id"] if isinstance(v, dict) else v)
def test_decoy_fails_evidence_check(fixture, entry):
    ok, note = _check(fixture, entry)

    assert not ok, f"decoy unexpectedly verified: {note}"
