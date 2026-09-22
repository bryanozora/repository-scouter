"""Tests for the read-only agent tools (app/agent/tools.py).

github_client.list_tree/read_file are monkeypatched here rather than
mocked at the HTTP level -- their contract against the real GitHub API is
already covered by test_github_client.py, so these tests focus on what
tools.py itself adds on top: path safety, skip-dirs, size/entry caps,
binary rejection, the already-read tracker, and the never-raise guarantee.
"""

from __future__ import annotations

import pytest

from app.agent import tools
from app.agent.tools import LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA, TOOLS, RepoTools
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
    assert LIST_DIRECTORY_SCHEMA["function"]["parameters"]["required"] == ["path"]
    assert READ_FILE_SCHEMA["function"]["parameters"]["required"] == ["path"]
    assert TOOLS == [LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA]
