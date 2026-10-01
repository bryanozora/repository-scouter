"""Tests for the read-only agent tools (app/agent/tools.py).

github_client.list_tree/read_file are monkeypatched here rather than
mocked at the HTTP level -- their contract against the real GitHub API is
already covered by test_github_client.py, so these tests focus on what
tools.py itself adds on top: path safety, skip-dirs, size/entry caps,
binary rejection, the already-read tracker, and the never-raise guarantee.
"""

from __future__ import annotations

import time

import pytest

from app.agent import tools
from app.agent.tools import (
    GET_DEPENDENCIES_SCHEMA,
    LIST_DIRECTORY_SCHEMA,
    READ_FILE_SCHEMA,
    REPORT_FINDING_SCHEMA,
    SEARCH_CODE_SCHEMA,
    TOOLS,
    RepoTools,
)
from app.config import Settings
from app.github_client import GitHubError, RepoRef, RepoTree, TreeEntry

REF = RepoRef(owner="octocat", repo="Hello-World")

SAMPLE_TREE = RepoTree(
    entries=[
        TreeEntry(path="README.md", type="blob", size=20),
        TreeEntry(path="src", type="tree"),
        TreeEntry(path="src/main.py", type="blob", size=100),
        TreeEntry(path="src/utils.py", type="blob", size=50),
        TreeEntry(path="node_modules", type="tree"),
        TreeEntry(path="node_modules/pkg", type="tree"),
        TreeEntry(path="node_modules/pkg/index.js", type="blob", size=10),
        TreeEntry(path="image.png", type="blob", size=500),
    ],
    truncated=False,
)


def make_settings(**overrides) -> Settings:
    defaults = dict(
        llm_provider="ollama",
        llm_model="qwen2.5:7b-instruct",
        ollama_base_url="http://localhost:11434/v1",
        github_token=None,
        max_steps=12,
        max_file_bytes=100_000,
        max_tool_result_chars=6000,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_repo_tools(
    monkeypatch,
    tree: RepoTree = SAMPLE_TREE,
    file_contents: dict[str, str] | None = None,
    settings: Settings | None = None,
    list_tree_calls: list | None = None,
    read_file_calls: list | None = None,
) -> RepoTools:
    """Build a RepoTools with github_client.list_tree/read_file monkeypatched."""
    file_contents = file_contents or {}

    def fake_list_tree(ref, branch, token=None):
        if list_tree_calls is not None:
            list_tree_calls.append((ref, branch, token))
        return tree

    def fake_read_file(ref, path, branch=None, token=None):
        if read_file_calls is not None:
            read_file_calls.append((ref, path, branch, token))
        return file_contents[path]

    monkeypatch.setattr(tools.github_client, "list_tree", fake_list_tree)
    monkeypatch.setattr(tools.github_client, "read_file", fake_read_file)

    return RepoTools(REF, branch="main", token=None, settings=settings or make_settings())


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_lists_root_and_hides_skip_dirs(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("")

    assert "dir  src" in result
    assert "file README.md (20B)" in result
    assert "file image.png" in result
    assert "node_modules" not in result


def test_list_directory_lists_nested_directory(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("src")

    assert "main.py" in result
    assert "utils.py" in result
    assert "README.md" not in result


def test_list_directory_returns_already_listed_note_on_repeat(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    first = rt.list_directory("")
    second = rt.list_directory("")

    assert "dir  src" in first
    assert "already listed" in second.lower()
    assert "dir  src" not in second


def test_list_directory_different_paths_do_not_trigger_already_listed_note(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    root = rt.list_directory("")
    src = rt.list_directory("src")

    assert "already listed" not in root.lower()
    assert "already listed" not in src.lower()


def test_list_directory_rejects_dotdot(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("../etc")

    assert result.startswith("Error")


def test_list_directory_rejects_absolute_path(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("/etc")

    assert result.startswith("Error")


def test_list_directory_reports_missing_path(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("nope")

    assert result.startswith("Error")
    assert "not found" in result


def test_list_directory_reports_file_is_not_a_directory(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("README.md")

    assert result.startswith("Error")
    assert "read_file" in result


def test_list_directory_caps_number_of_entries(monkeypatch):
    extra = 20
    many_entries = [
        TreeEntry(path=f"file{i:03d}.txt", type="blob", size=1) for i in range(tools.MAX_LIST_ENTRIES + extra)
    ]
    rt = make_repo_tools(monkeypatch, tree=RepoTree(entries=many_entries, truncated=False))

    result = rt.list_directory("")

    listed = [line for line in result.splitlines() if line.startswith("file ")]
    assert len(listed) == tools.MAX_LIST_ENTRIES
    assert f"{extra} more" in result


def test_list_directory_never_raises_on_github_error(monkeypatch):
    def boom(ref, branch, token=None):
        raise GitHubError("rate limited")

    monkeypatch.setattr(tools.github_client, "list_tree", boom)
    rt = RepoTools(REF, branch="main", settings=make_settings())

    result = rt.list_directory("")

    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_line_numbered_content_in_delimiter(monkeypatch):
    rt = make_repo_tools(monkeypatch, file_contents={"src/main.py": "line1\nline2\nline3"})

    result = rt.read_file("src/main.py")

    assert "<file_content>" in result
    assert "</file_content>" in result
    assert "1: line1" in result
    assert "2: line2" in result
    assert "3: line3" in result


def test_read_file_line_range(monkeypatch):
    rt = make_repo_tools(monkeypatch, file_contents={"src/main.py": "line1\nline2\nline3"})

    result = rt.read_file("src/main.py", start=2, end=2)

    assert "2: line2" in result
    assert "line1" not in result
    assert "line3" not in result


def test_read_file_rejects_dotdot(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.read_file("../secrets.txt")

    assert result.startswith("Error")


def test_read_file_rejects_absolute_path(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.read_file("/etc/passwd")

    assert result.startswith("Error")


def test_read_file_rejects_absolute_path_as_absolute_not_as_missing(monkeypatch):
    # /etc/passwd must be refused because it's absolute -- not merely
    # because "etc/passwd" happens not to exist in this repo's tree.
    read_calls: list = []
    rt = make_repo_tools(monkeypatch, read_file_calls=read_calls)

    result = rt.read_file("/etc/passwd")

    assert result.startswith("Error")
    assert "relative to the repo root" in result
    assert "not found" not in result
    assert read_calls == []


def test_read_file_rejects_absolute_path_even_when_relative_twin_exists(monkeypatch):
    rt = make_repo_tools(monkeypatch, file_contents={"src/main.py": "line1"})

    result = rt.read_file("/src/main.py")

    assert result.startswith("Error")
    assert "relative to the repo root" in result


def test_list_directory_rejects_absolute_path_even_when_relative_twin_exists(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.list_directory("/src")

    assert result.startswith("Error")
    assert "relative to the repo root" in result


def test_read_file_still_accepts_trailing_slash_and_dot_slash(monkeypatch):
    rt = make_repo_tools(monkeypatch, file_contents={"src/main.py": "line1"})

    result = rt.read_file("./src/main.py")

    assert "1: line1" in result


def test_read_file_rejects_binary_extension_without_fetching(monkeypatch):
    read_calls: list = []
    rt = make_repo_tools(monkeypatch, read_file_calls=read_calls)

    result = rt.read_file("image.png")

    assert result.startswith("Error")
    assert read_calls == []


def test_read_file_enforces_size_cap_before_fetching(monkeypatch):
    read_calls: list = []
    settings = make_settings(max_file_bytes=10)
    tree = RepoTree(entries=[TreeEntry(path="big.py", type="blob", size=1000)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, settings=settings, read_file_calls=read_calls)

    result = rt.read_file("big.py")

    assert result.startswith("Error")
    assert read_calls == []


def test_read_file_rejects_content_with_nul_byte(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="weird.txt", type="blob", size=10)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"weird.txt": "abc\x00def"})

    result = rt.read_file("weird.txt")

    assert result.startswith("Error")


def test_read_file_returns_already_read_note_on_repeat(monkeypatch):
    read_calls: list = []
    rt = make_repo_tools(monkeypatch, file_contents={"src/main.py": "line1"}, read_file_calls=read_calls)

    first = rt.read_file("src/main.py")
    second = rt.read_file("src/main.py")

    assert "<file_content>" in first
    assert "already read" in second.lower()
    assert "<file_content>" not in second
    assert len(read_calls) == 1


TEN_LINES = "\n".join(f"line{i}" for i in range(1, 11))
TEN_LINE_TREE = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=len(TEN_LINES))], truncated=False)


def test_read_file_allows_reading_a_different_range_of_the_same_file(monkeypatch):
    read_calls: list = []
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES}, read_file_calls=read_calls)

    rt.read_file("app.py", start=1, end=3)
    second = rt.read_file("app.py", start=6, end=8)

    assert "<file_content>" in second
    assert "6: line6" in second
    assert "already read" not in second.lower()
    assert len(read_calls) == 1  # second range came from the content cache


def test_read_file_returns_note_when_requested_range_was_already_fully_shown(monkeypatch):
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES})

    rt.read_file("app.py", start=1, end=5)
    second = rt.read_file("app.py", start=2, end=4)

    assert "already read" in second.lower()
    assert "<file_content>" not in second
    assert "1-5" in second


def test_read_file_shows_a_partially_overlapping_range_in_full(monkeypatch):
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES})

    rt.read_file("app.py", start=1, end=5)
    second = rt.read_file("app.py", start=4, end=8)

    assert "4: line4" in second
    assert "8: line8" in second


def test_read_file_whole_file_after_a_partial_read_is_not_a_repeat(monkeypatch):
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES})

    rt.read_file("app.py", start=1, end=5)
    second = rt.read_file("app.py")

    assert "10: line10" in second


def test_read_file_note_after_two_disjoint_reads_lists_both_ranges(monkeypatch):
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES})

    rt.read_file("app.py", start=1, end=2)
    rt.read_file("app.py", start=7, end=8)
    third = rt.read_file("app.py", start=7, end=7)

    assert "already read" in third.lower()
    assert "1-2, 7-8" in third


def test_format_line_ranges_collapses_runs():
    assert tools._format_line_ranges({1, 2, 3, 7, 8, 12}) == "1-3, 7-8, 12"
    assert tools._format_line_ranges(set()) == ""


def test_shown_lines_tracks_exactly_the_lines_returned(monkeypatch):
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES})

    assert rt.shown_lines("app.py") == frozenset()
    rt.read_file("./app.py", start=3, end=4)
    rt.read_file("app.py", start=9)

    assert rt.shown_lines("app.py") == frozenset({3, 4, 9, 10})
    assert rt.shown_lines("./app.py") == frozenset({3, 4, 9, 10})


def test_failed_reads_do_not_count_as_shown(monkeypatch):
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, file_contents={"app.py": TEN_LINES})

    rt.read_file("app.py", start=50, end=60)  # out of range -> error

    assert rt.shown_lines("app.py") == frozenset()


def test_read_file_truncation_cuts_at_whole_lines_and_only_counts_those_as_shown(monkeypatch):
    # "1: line1" is 8 chars; each further line adds "\n" + 8 = 9 chars.
    # A 30-char cap fits lines 1-3 (8 + 9 + 9 = 26) but not line 4 (35).
    settings = make_settings(max_tool_result_chars=30)
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, settings=settings, file_contents={"app.py": TEN_LINES})

    result = rt.read_file("app.py")

    assert "3: line3" in result
    assert "4: line" not in result
    assert rt.shown_lines("app.py") == frozenset({1, 2, 3})
    assert "start=4" in result  # tells the model how to continue


def test_read_file_after_truncation_can_continue_from_the_next_line(monkeypatch):
    settings = make_settings(max_tool_result_chars=30)
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, settings=settings, file_contents={"app.py": TEN_LINES})

    rt.read_file("app.py")
    second = rt.read_file("app.py", start=4, end=5)

    assert "4: line4" in second
    assert "already read" not in second.lower()


def test_read_file_header_reports_the_lines_actually_shown(monkeypatch):
    settings = make_settings(max_tool_result_chars=30)
    rt = make_repo_tools(monkeypatch, tree=TEN_LINE_TREE, settings=settings, file_contents={"app.py": TEN_LINES})

    result = rt.read_file("app.py")

    assert result.startswith("File: app.py (lines 1-3 of 10)")


def test_read_file_single_overlong_line_is_cut_but_not_counted_as_shown(monkeypatch):
    settings = make_settings(max_tool_result_chars=20)
    content = "x" * 100
    tree = RepoTree(entries=[TreeEntry(path="min.js", type="blob", size=100)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, settings=settings, file_contents={"min.js": content})

    result = rt.read_file("min.js")

    assert "<file_content>" in result
    assert "truncated" in result.lower()
    assert rt.shown_lines("min.js") == frozenset()


def test_read_file_truncates_to_max_tool_result_chars(monkeypatch):
    settings = make_settings(max_tool_result_chars=20)
    long_content = "\n".join(f"line {i}" for i in range(100))
    tree = RepoTree(entries=[TreeEntry(path="long.py", type="blob", size=len(long_content))], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, settings=settings, file_contents={"long.py": long_content})

    result = rt.read_file("long.py")

    assert "truncated" in result.lower()


def test_read_file_reports_missing_path(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.read_file("nope.py")

    assert result.startswith("Error")
    assert "not found" in result


def test_read_file_reports_directory_is_not_a_file(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.read_file("src")

    assert result.startswith("Error")
    assert "list_directory" in result


def test_read_file_never_raises_on_github_error(monkeypatch):
    def boom(ref, branch, token=None):
        raise GitHubError("boom")

    monkeypatch.setattr(tools.github_client, "list_tree", boom)
    rt = RepoTools(REF, branch="main", settings=make_settings())

    result = rt.read_file("anything.py")

    assert result.startswith("Error")


def test_read_file_rejects_non_integer_start(monkeypatch):
    rt = make_repo_tools(monkeypatch, file_contents={"src/main.py": "line1\nline2"})

    result = rt.read_file("src/main.py", start="not-a-number")

    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# caching + schemas
# ---------------------------------------------------------------------------


def test_tree_is_cached_across_calls(monkeypatch):
    calls: list = []
    rt = make_repo_tools(monkeypatch, file_contents={"README.md": "hi"}, list_tree_calls=calls)

    rt.list_directory("")
    rt.read_file("README.md")

    assert len(calls) == 1


def test_tool_schemas_declare_expected_names_and_params():
    assert LIST_DIRECTORY_SCHEMA["function"]["name"] == "list_directory"
    assert READ_FILE_SCHEMA["function"]["name"] == "read_file"
    assert GET_DEPENDENCIES_SCHEMA["function"]["name"] == "get_dependencies"
    assert SEARCH_CODE_SCHEMA["function"]["name"] == "search_code"
    assert LIST_DIRECTORY_SCHEMA["function"]["parameters"]["required"] == ["path"]
    assert READ_FILE_SCHEMA["function"]["parameters"]["required"] == ["path"]
    assert GET_DEPENDENCIES_SCHEMA["function"]["parameters"]["required"] == []
    assert SEARCH_CODE_SCHEMA["function"]["parameters"]["required"] == ["query"]
    assert TOOLS == [
        LIST_DIRECTORY_SCHEMA,
        READ_FILE_SCHEMA,
        GET_DEPENDENCIES_SCHEMA,
        SEARCH_CODE_SCHEMA,
        REPORT_FINDING_SCHEMA,
    ]


def test_report_finding_schema_matches_the_validator():
    from app.agent.findings import FIELDS, ISSUE_TYPES, REQUIRED_FIELDS, SEVERITIES, V1_CATEGORIES

    fn = REPORT_FINDING_SCHEMA["function"]
    props = fn["parameters"]["properties"]
    assert fn["name"] == "report_finding"
    assert sorted(fn["parameters"]["required"]) == sorted(REQUIRED_FIELDS)
    assert "confidence" not in fn["parameters"]["required"]  # optional, defaults to 0.5
    assert "suggestion" not in fn["parameters"]["required"]  # optional, default message
    assert set(props) == set(FIELDS)
    assert props["severity"]["enum"] == list(SEVERITIES)
    # The schema only offers what v1 accepts, so the model isn't tempted
    # into categories/issue types the validator would reject.
    assert props["category"]["enum"] == list(V1_CATEGORIES)
    assert props["issue_type"]["enum"] == list(ISSUE_TYPES)
    assert props["line_start"]["type"] == "integer"
    assert props["confidence"]["type"] == "number"


def make_fake_clock(*values):
    """A callable returning `values` in order, then repeating the last one."""
    remaining = list(values)

    def clock():
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return clock


# ---------------------------------------------------------------------------
# get_dependencies
# ---------------------------------------------------------------------------


def test_get_dependencies_finds_and_summarizes_known_manifests(monkeypatch):
    tree = RepoTree(
        entries=[
            TreeEntry(path="requirements.txt", type="blob", size=30),
            TreeEntry(path="frontend/package.json", type="blob", size=60),
            TreeEntry(path="src/main.py", type="blob", size=10),
        ],
        truncated=False,
    )
    contents = {
        "requirements.txt": "httpx>=0.27\npytest>=8.0\n",
        "frontend/package.json": '{"dependencies": {"react": "^18.0.0"}}',
    }
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents=contents)

    result = rt.get_dependencies()

    assert "requirements.txt" in result
    assert "httpx>=0.27" in result
    assert "frontend/package.json" in result
    assert "react@^18.0.0" in result
    assert "<file_content>" in result


def test_get_dependencies_reports_none_found(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="src/main.py", type="blob", size=10)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree)

    result = rt.get_dependencies()

    assert "no known dependency manifests" in result.lower()


def test_get_dependencies_skips_manifests_in_skip_dirs(monkeypatch):
    tree = RepoTree(
        entries=[TreeEntry(path="node_modules/some-pkg/package.json", type="blob", size=20)],
        truncated=False,
    )
    rt = make_repo_tools(monkeypatch, tree=tree)

    result = rt.get_dependencies()

    assert "no known dependency manifests" in result.lower()


def test_get_dependencies_excludes_lock_files(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="package-lock.json", type="blob", size=99999)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree)

    result = rt.get_dependencies()

    assert "no known dependency manifests" in result.lower()


def test_get_dependencies_returns_already_fetched_note_on_repeat(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="requirements.txt", type="blob", size=10)], truncated=False)
    read_calls: list = []
    rt = make_repo_tools(
        monkeypatch, tree=tree, file_contents={"requirements.txt": "httpx>=0.27"}, read_file_calls=read_calls
    )

    first = rt.get_dependencies()
    second = rt.get_dependencies()

    assert "httpx" in first
    assert "already fetched" in second.lower()
    assert len(read_calls) == 1


def test_get_dependencies_never_raises_on_github_error(monkeypatch):
    def boom(ref, branch, token=None):
        raise GitHubError("boom")

    monkeypatch.setattr(tools.github_client, "list_tree", boom)
    rt = RepoTools(REF, branch="main", settings=make_settings())

    result = rt.get_dependencies()

    assert result.startswith("Error")


def test_get_dependencies_falls_back_to_raw_content_for_unparsed_manifest_types(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="pom.xml", type="blob", size=40)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"pom.xml": "<project>...</project>"})

    result = rt.get_dependencies()

    assert "pom.xml" in result
    assert "<project>" in result


# ---------------------------------------------------------------------------
# search_code: basic behavior
# ---------------------------------------------------------------------------


def test_search_code_returns_matching_lines_with_path_and_line_number(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=30)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": "line one\nneedle here\nline three"})

    result = rt.search_code("needle")

    assert "app.py:2:" in result
    assert "needle here" in result
    assert "<file_content>" in result


def test_search_code_is_case_insensitive(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=30)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": "Needle Here"})

    result = rt.search_code("needle")

    assert "app.py:1:" in result


def test_search_code_reports_no_matches(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=10)], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": "nothing to see"})

    result = rt.search_code("zzz_not_present")

    assert "no matches" in result.lower()


def test_search_code_rejects_empty_query(monkeypatch):
    rt = make_repo_tools(monkeypatch)

    result = rt.search_code("")

    assert result.startswith("Error")


def test_search_code_never_raises_on_github_error(monkeypatch):
    def boom(ref, branch, token=None):
        raise GitHubError("boom")

    monkeypatch.setattr(tools.github_client, "list_tree", boom)
    rt = RepoTools(REF, branch="main", settings=make_settings())

    result = rt.search_code("anything")

    assert result.startswith("Error")


def test_search_code_truncates_long_matching_lines(monkeypatch):
    long_line = "x" * 500 + "needle" + "y" * 500
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=len(long_line))], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": long_line})

    result = rt.search_code("needle")

    assert "..." in result
    matching_line = next(line for line in result.splitlines() if line.startswith("app.py:1:"))
    assert len(matching_line) < 300  # well under the original ~1006-char line


def test_search_code_and_read_file_share_fetched_content_cache(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=20)], truncated=False)
    read_calls: list = []
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": "needle here"}, read_file_calls=read_calls)

    rt.search_code("needle")
    result = rt.read_file("app.py")

    assert len(read_calls) == 1  # read_file reused the content search_code already fetched
    assert "needle here" in result


# ---------------------------------------------------------------------------
# search_code: candidate selection (priority, exclusions)
# ---------------------------------------------------------------------------

# 5 source/config candidates (== SEARCH_CONCURRENCY) so they fill exactly one
# fetch batch, followed by 3 doc/test/example candidates that must land in the
# next batch -- batch order is deterministic (each batch is waited on before
# the next starts) even though *within* a batch, concurrent fetches can
# complete in any order, so ordering tests assert at the batch level.
SEARCH_TREE = RepoTree(
    entries=[
        TreeEntry(path="app.py", type="blob", size=50),  # source
        TreeEntry(path="models.py", type="blob", size=50),  # source
        TreeEntry(path="utils.py", type="blob", size=50),  # source
        TreeEntry(path="config.json", type="blob", size=30),  # config
        TreeEntry(path="settings.yaml", type="blob", size=30),  # config
        TreeEntry(path="README.md", type="blob", size=40),  # doc (last)
        TreeEntry(path="tests/test_app.py", type="blob", size=60),  # test (last, despite .py)
        TreeEntry(path="examples/demo.py", type="blob", size=20),  # example (last, despite .py)
        TreeEntry(path="bundle.min.js", type="blob", size=1000),  # minified -- excluded
        TreeEntry(path="package-lock.json", type="blob", size=5000),  # lock file -- excluded
        TreeEntry(path="logo.png", type="blob", size=200),  # binary -- excluded
        TreeEntry(path="node_modules/pkg/index.js", type="blob", size=10),  # skip-dir -- excluded
    ],
    truncated=False,
)
SEARCH_TREE_SOURCE_AND_CONFIG = {"app.py", "models.py", "utils.py", "config.json", "settings.yaml"}
SEARCH_TREE_DOCS_TESTS_EXAMPLES = {"README.md", "tests/test_app.py", "examples/demo.py"}


def test_search_code_prioritizes_source_and_config_over_docs_tests_examples(monkeypatch):
    read_calls: list = []
    contents = {path: "needle here" for path in SEARCH_TREE_SOURCE_AND_CONFIG | SEARCH_TREE_DOCS_TESTS_EXAMPLES}
    rt = make_repo_tools(monkeypatch, tree=SEARCH_TREE, file_contents=contents, read_file_calls=read_calls)

    rt.search_code("needle")

    fetched_order = [call[1] for call in read_calls]
    assert len(fetched_order) == 8
    assert set(fetched_order[:5]) == SEARCH_TREE_SOURCE_AND_CONFIG
    assert set(fetched_order[5:]) == SEARCH_TREE_DOCS_TESTS_EXAMPLES


def test_search_code_skips_minified_lock_binary_and_skip_dir_files(monkeypatch):
    read_calls: list = []
    contents = {"app.py": "nothing interesting"}
    rt = make_repo_tools(monkeypatch, tree=SEARCH_TREE, file_contents=contents, read_file_calls=read_calls)

    rt.search_code("nothing")

    fetched_paths = {call[1] for call in read_calls}
    assert "bundle.min.js" not in fetched_paths
    assert "package-lock.json" not in fetched_paths
    assert "logo.png" not in fetched_paths
    assert "node_modules/pkg/index.js" not in fetched_paths


# ---------------------------------------------------------------------------
# search_code: caps and the internal time budget
# ---------------------------------------------------------------------------


def test_search_code_caps_number_of_results(monkeypatch):
    content = "\n".join(f"needle {i}" for i in range(tools.MAX_SEARCH_RESULTS + 10))
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=len(content))], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": content})

    result = rt.search_code("needle")

    matches = [line for line in result.splitlines() if line.startswith("app.py:")]
    assert len(matches) == tools.MAX_SEARCH_RESULTS
    assert "stopped early" in result.lower()


def test_search_code_caps_number_of_files_fetched(monkeypatch):
    many_entries = [TreeEntry(path=f"file{i:03d}.py", type="blob", size=10) for i in range(tools.MAX_SEARCH_FILES + 5)]
    tree = RepoTree(entries=many_entries, truncated=False)
    contents = {e.path: "no match here" for e in many_entries}
    read_calls: list = []
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents=contents, read_file_calls=read_calls)

    result = rt.search_code("xyz_never_matches")

    assert len(read_calls) == tools.MAX_SEARCH_FILES
    assert "stopped early" in result.lower()


def test_search_code_reports_bounded_result_when_deadline_already_passed(monkeypatch):
    tree = RepoTree(entries=[TreeEntry(path="file000.py", type="blob", size=10)], truncated=False)
    settings = make_settings(tool_timeout_seconds=1.0)
    read_calls: list = []
    rt = make_repo_tools(
        monkeypatch, tree=tree, file_contents={"file000.py": "x"}, settings=settings, read_file_calls=read_calls
    )
    clock = make_fake_clock(0.0, 5.0)  # deadline=1.0; already past it by the first batch check

    result = rt.search_code("anything", now=clock)

    assert len(read_calls) == 0
    assert "ran out of time" in result.lower()


def test_search_code_fetches_a_batch_of_candidates_concurrently(monkeypatch):
    """5 candidates, each slow enough that 5 sequential fetches would blow the
    deadline (5 * 0.1s = 0.5s > 0.3s) but 5 *concurrent* fetches comfortably fit."""
    entries = [TreeEntry(path=f"file{i}.py", type="blob", size=10) for i in range(5)]
    tree = RepoTree(entries=entries, truncated=False)
    settings = make_settings(tool_timeout_seconds=0.3)
    read_calls: list = []

    def fake_list_tree(ref, branch, token=None):
        return tree

    def fake_read_file(ref, path, branch=None, token=None):
        read_calls.append(path)
        time.sleep(0.1)
        return "needle here"

    monkeypatch.setattr(tools.github_client, "list_tree", fake_list_tree)
    monkeypatch.setattr(tools.github_client, "read_file", fake_read_file)
    rt = RepoTools(REF, branch="main", settings=settings)

    result = rt.search_code("needle")

    assert len(read_calls) == 5
    assert "stopped early" not in result.lower()


def test_search_code_still_stops_early_when_out_of_time_despite_concurrency(monkeypatch):
    """More candidates than fit in the budget even with concurrent batches."""
    entries = [TreeEntry(path=f"file{i:03d}.py", type="blob", size=10) for i in range(15)]
    tree = RepoTree(entries=entries, truncated=False)
    settings = make_settings(tool_timeout_seconds=0.15)

    def fake_list_tree(ref, branch, token=None):
        return tree

    def fake_read_file(ref, path, branch=None, token=None):
        time.sleep(0.1)
        return "no match here"

    monkeypatch.setattr(tools.github_client, "list_tree", fake_list_tree)
    monkeypatch.setattr(tools.github_client, "read_file", fake_read_file)
    rt = RepoTools(REF, branch="main", settings=settings)

    result = rt.search_code("anything")

    assert "ran out of time" in result.lower()


# ---------------------------------------------------------------------------
# report_finding: read-gating, verification, dedup, cap
# ---------------------------------------------------------------------------

from app.agent.findings import MAX_FINDINGS, Finding  # noqa: E402

DB_PY = "\n".join(
    [
        "import sqlite3",  # 1
        "",  # 2
        "def get(cur, uid):",  # 3
        '    cur.execute(f"SELECT * FROM users WHERE id = {uid}")',  # 4
        "",  # 5
        "",  # 6
        "",  # 7
        "",  # 8
        "def ok(cur, uid):",  # 9
        '    cur.execute("SELECT * FROM users WHERE id = ?", (uid,))',  # 10
    ]
)
DB_TREE = RepoTree(
    entries=[
        TreeEntry(path="db.py", type="blob", size=len(DB_PY)),
        TreeEntry(path="src", type="tree"),
    ],
    truncated=False,
)


def finding_args(**overrides) -> dict:
    args = dict(
        severity="high",
        category="security",
        issue_type="sql_injection",
        file="db.py",
        line_start=4,
        line_end=4,
        description="uid is formatted into the query.",
        suggestion="Use a ? placeholder.",
        confidence=0.9,
    )
    args.update(overrides)
    return args


def make_db_tools(monkeypatch) -> RepoTools:
    return make_repo_tools(monkeypatch, tree=DB_TREE, file_contents={"db.py": DB_PY})


def test_report_finding_records_a_verified_finding(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    result = rt.report_finding(finding_args())

    assert result.startswith("Recorded finding #1")
    assert "verified" in result.lower()
    assert len(rt.findings) == 1
    assert rt.findings[0].verified is True
    assert rt.findings[0].file == "db.py"
    assert "line 4" in rt.findings[0].verification_note


def test_report_finding_flags_a_finding_whose_evidence_does_not_match(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    result = rt.report_finding(finding_args(line_start=10, line_end=10))

    assert "UNVERIFIED" in result
    assert "report_finding again" in result
    assert len(rt.findings) == 1
    assert rt.findings[0].verified is False
    assert rt.findings[0].verification_note


def test_report_finding_returns_all_validation_errors_and_records_nothing(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    result = rt.report_finding(finding_args(severity="critical", confidence=7))

    assert result.startswith("Error: report_finding rejected, nothing was recorded")
    assert "- severity:" in result
    assert "- confidence:" in result
    assert rt.findings == []


def test_report_finding_rejects_a_file_that_was_never_read(monkeypatch):
    rt = make_db_tools(monkeypatch)

    result = rt.report_finding(finding_args())

    assert result.startswith("Error: report_finding rejected")
    assert "not seen" in result
    assert "read_file('db.py', start=4, end=4)" in result
    assert rt.findings == []


def test_a_search_that_did_not_show_the_cited_line_does_not_count(monkeypatch):
    # M2b decision: search lines shown in full count as seen -- but only
    # those lines, not the whole file the search touched.
    rt = make_db_tools(monkeypatch)
    rt.search_code("import")  # shows line 1 only

    result = rt.report_finding(finding_args())  # cites line 4

    assert result.startswith("Error")
    assert rt.findings == []


def test_report_finding_rejects_lines_outside_what_was_shown(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py", start=1, end=3)

    result = rt.report_finding(finding_args(line_start=4, line_end=4))

    assert result.startswith("Error")
    assert "1-3" in result
    assert "read_file('db.py', start=4, end=4)" in result
    assert rt.findings == []


def test_report_finding_rejects_a_missing_file(monkeypatch):
    rt = make_db_tools(monkeypatch)

    result = rt.report_finding(finding_args(file="nope.py"))

    assert result.startswith("Error")
    assert "not found" in result


def test_report_finding_rejects_a_directory(monkeypatch):
    rt = make_db_tools(monkeypatch)

    result = rt.report_finding(finding_args(file="src"))

    assert result.startswith("Error")
    assert "directory" in result


def test_report_finding_rejects_lines_past_the_end_of_the_file(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    result = rt.report_finding(finding_args(line_start=9, line_end=14))

    assert result.startswith("Error")
    assert "10 lines" in result


def test_report_finding_normalizes_the_file_path(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    result = rt.report_finding(finding_args(file="./db.py"))

    assert result.startswith("Recorded")
    assert rt.findings[0].file == "db.py"


def test_report_finding_does_not_record_a_duplicate_of_a_verified_finding(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")
    rt.report_finding(finding_args(line_start=3, line_end=4))

    result = rt.report_finding(finding_args(line_start=4, line_end=5))

    assert "already recorded" in result.lower()
    assert len(rt.findings) == 1


def test_report_finding_different_issue_type_on_same_lines_is_not_a_duplicate(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")
    rt.report_finding(finding_args())

    rt.report_finding(finding_args(issue_type="hardcoded_secret"))

    assert len(rt.findings) == 2


def test_report_finding_corrected_report_replaces_an_overlapping_unverified_one(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")
    # First attempt cites line 1 (window 1-3: no SQL) -> flagged.
    rt.report_finding(finding_args(line_start=1, line_end=1))
    assert rt.findings[0].verified is False

    # Corrected attempt overlaps it and verifies -> replaces it.
    result = rt.report_finding(finding_args(line_start=1, line_end=4))

    assert result.startswith("Recorded")
    assert len(rt.findings) == 1
    assert rt.findings[0].verified is True
    assert (rt.findings[0].line_start, rt.findings[0].line_end) == (1, 4)


def test_report_finding_enforces_the_per_scan_cap(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")
    # Seeded directly (distinct files, so no dedup): the cap is what's under test.
    for i in range(MAX_FINDINGS):
        rt._findings.append(
            Finding(
                severity="low", category="security", issue_type="sql_injection", file=f"f{i}.py",
                line_start=1, line_end=1, description="d", suggestion="s", confidence=0.5,
            )
        )

    result = rt.report_finding(finding_args())

    assert result.startswith("Error")
    assert str(MAX_FINDINGS) in result
    assert len(rt.findings) == MAX_FINDINGS


def test_findings_property_returns_a_copy(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")
    rt.report_finding(finding_args())

    rt.findings.clear()

    assert len(rt.findings) == 1


def test_report_finding_never_raises(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(tools, "check_evidence", boom)

    result = rt.report_finding(finding_args())

    assert result.startswith("Error")
    assert "kaboom" in result


def test_report_finding_rejects_non_dict_arguments(monkeypatch):
    rt = make_db_tools(monkeypatch)

    result = rt.report_finding("severity=high")  # type: ignore[arg-type]

    assert result.startswith("Error: report_finding rejected")


def test_forget_shown_allows_a_re_read_and_requires_it_before_reporting(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    rt.forget_shown("./db.py")

    assert rt.shown_lines("db.py") == frozenset()
    assert rt.report_finding(finding_args()).startswith("Error")
    assert "<file_content>" in rt.read_file("db.py")  # not "already read"
    assert rt.report_finding(finding_args()).startswith("Recorded")


def test_forget_shown_on_an_unread_path_is_harmless(monkeypatch):
    rt = make_db_tools(monkeypatch)

    rt.forget_shown("never-read.py")

    assert rt.shown_lines("never-read.py") == frozenset()


# ---------------------------------------------------------------------------
# search_code lines shown in full count as seen (tracked separately)
# ---------------------------------------------------------------------------


def test_search_matches_shown_in_full_are_tracked_separately_from_read_lines(monkeypatch):
    rt = make_db_tools(monkeypatch)

    rt.search_code("SELECT")

    assert rt.search_seen_lines("db.py") == frozenset({4, 10})
    assert rt.shown_lines("db.py") == frozenset()  # read_file's record is untouched


def test_report_finding_accepts_lines_seen_only_in_search_results(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.search_code("SELECT")

    result = rt.report_finding(finding_args(line_start=4, line_end=4))

    assert result.startswith("Recorded finding #1 (verified")


def test_report_finding_still_rejects_lines_not_shown_by_search_or_read(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.search_code("SELECT")

    result = rt.report_finding(finding_args(line_start=3, line_end=4))  # line 3 was never shown

    assert result.startswith("Error")
    assert "3-4" in result


def test_search_and_read_lines_combine_to_cover_a_range(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.search_code("SELECT")  # shows line 4
    rt.read_file("db.py", start=3, end=3)

    result = rt.report_finding(finding_args(line_start=3, line_end=4))

    assert result.startswith("Recorded")


def test_truncated_search_snippets_do_not_count_as_seen(monkeypatch):
    long_line = "x = '" + "a" * 300 + "'  # SELECT"
    tree = RepoTree(entries=[TreeEntry(path="app.py", type="blob", size=len(long_line))], truncated=False)
    rt = make_repo_tools(monkeypatch, tree=tree, file_contents={"app.py": long_line})

    rt.search_code("SELECT")

    assert rt.search_seen_lines("app.py") == frozenset()


def test_forget_search_lines_clears_all_search_seen_lines(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.search_code("SELECT")

    rt.forget_search_lines()

    assert rt.search_seen_lines("db.py") == frozenset()
    assert rt.report_finding(finding_args()).startswith("Error")


def test_unseen_file_error_mentions_search_too(monkeypatch):
    rt = make_db_tools(monkeypatch)

    result = rt.report_finding(finding_args())

    assert "not seen" in result
    assert "read_file('db.py', start=4, end=4)" in result


# ---------------------------------------------------------------------------
# finding source
# ---------------------------------------------------------------------------


def test_report_finding_records_where_the_finding_came_from(monkeypatch):
    rt = make_db_tools(monkeypatch)
    rt.read_file("db.py")

    rt.report_finding(finding_args())
    rt.report_finding(finding_args(issue_type="hardcoded_secret", line_start=1, line_end=1), source="parsed_text")

    assert [f.source for f in rt.findings] == ["tool_call", "parsed_text"]
    assert rt.findings[0].to_dict()["source"] == "tool_call"


# ---------------------------------------------------------------------------
# candidate_search: fixed searches run by the loop (approach 3)
# ---------------------------------------------------------------------------

CANDIDATE_TREE = RepoTree(
    entries=[
        TreeEntry(path="config.py", type="blob", size=100),
        TreeEntry(path="db.py", type="blob", size=100),
    ],
    truncated=False,
)
CONFIG_PY = "\n".join(
    [
        "import os",  # 1
        'API_KEY = "9f2c4e8a1b7d3f6e"',  # 2
        "KEY_A = 1",  # 3
        "KEY_B = 2",  # 4
        "KEY_C = 3",  # 5
        "KEY_D = 4",  # 6
        'DB_PASSWORD = "Tr0ub4dor&3-prod"',  # 7
    ]
)


def make_candidate_tools(monkeypatch) -> RepoTools:
    return make_repo_tools(monkeypatch, tree=CANDIDATE_TREE, file_contents={"config.py": CONFIG_PY, "db.py": DB_PY})


def test_candidate_search_returns_matching_lines_from_several_terms(monkeypatch):
    rt = make_candidate_tools(monkeypatch)

    result = rt.candidate_search(("password", "SELECT"))

    assert result.startswith("<file_content>") and result.endswith("</file_content>")
    assert 'config.py:7: DB_PASSWORD = "Tr0ub4dor&3-prod"' in result
    assert "db.py:4:" in result


def test_candidate_search_round_robins_so_one_noisy_term_cannot_crowd_out_others(monkeypatch):
    rt = make_candidate_tools(monkeypatch)

    # "key" matches 5 lines; with max_lines=3 and per_term=2, password still gets in.
    result = rt.candidate_search(("key", "password"), per_term=2, max_lines=3)

    assert "config.py:7:" in result
    assert result.count("\n") - 1 == 3  # exactly 3 lines between the delimiters


def test_candidate_search_dedupes_lines_matched_by_several_terms(monkeypatch):
    rt = make_candidate_tools(monkeypatch)

    result = rt.candidate_search(("api_key", "API"))

    assert result.count("config.py:2:") == 1


def test_candidate_search_marks_only_the_lines_it_shows_as_seen(monkeypatch):
    rt = make_candidate_tools(monkeypatch)

    rt.candidate_search(("key",), per_term=1, max_lines=1)

    assert rt.search_seen_lines("config.py") == frozenset({2})


def test_candidate_search_lines_count_as_evidence_for_report_finding(monkeypatch):
    rt = make_candidate_tools(monkeypatch)
    rt.candidate_search(("SELECT",))

    result = rt.report_finding(finding_args())  # db.py line 4, never read_file'd

    assert result.startswith("Recorded finding #1 (verified")


def test_candidate_search_returns_empty_string_when_nothing_matches(monkeypatch):
    rt = make_candidate_tools(monkeypatch)

    assert rt.candidate_search(("no-such-term",)) == ""


def test_candidate_search_never_raises(monkeypatch):
    rt = make_candidate_tools(monkeypatch)

    def boom(*args, **kwargs):
        raise GitHubError("rate limited")

    monkeypatch.setattr(tools.github_client, "list_tree", boom)

    assert rt.candidate_search(("password",)) == ""


def test_default_candidate_terms_cover_both_issue_types():
    assert {"password", "secret", "token", "key"} <= set(tools.CANDIDATE_TERMS)
    assert {"SELECT", "execute("} <= set(tools.CANDIDATE_TERMS)
