"""Tests for the GitHub repo access layer (app/github_client.py).

All HTTP traffic is intercepted with respx; no real network calls. GET
requests only -- this module never writes to a repo, matching the
project's read-only rule.
"""

import base64

import httpx
import pytest
import respx

from app.github_client import (
    GitHubError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    RepoRef,
    get_repo_info,
    list_tree,
    parse_repo_url,
    read_file,
)

API_BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# parse_repo_url
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octocat/Hello-World",
        "http://github.com/octocat/Hello-World",
        "github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World/",
        "https://github.com/octocat/Hello-World.git",
        "github.com/octocat/Hello-World.git/",
    ],
)
def test_parse_repo_url_accepts_usual_forms(url):
    ref = parse_repo_url(url)

    assert ref == RepoRef(owner="octocat", repo="Hello-World")


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/octocat/Hello-World",  # wrong host
        "https://github.com/octocat",  # missing repo
        "https://github.com/",  # missing owner and repo
        "not a url at all",
        "https://github.com/octocat/Hello-World/tree/main",  # extra path segments
        "",
    ],
)
def test_parse_repo_url_rejects_anything_else(url):
    with pytest.raises(ValueError):
        parse_repo_url(url)


# ---------------------------------------------------------------------------
# get_repo_info
# ---------------------------------------------------------------------------


@respx.mock
def test_get_repo_info_returns_default_branch_and_language():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World").mock(
        return_value=httpx.Response(200, json={"default_branch": "main", "language": "Python"})
    )

    info = get_repo_info(RepoRef(owner="octocat", repo="Hello-World"))

    assert info.default_branch == "main"
    assert info.language == "Python"


@respx.mock
def test_get_repo_info_sends_auth_header_when_token_given():
    route = respx.get(f"{API_BASE}/repos/octocat/Hello-World").mock(
        return_value=httpx.Response(200, json={"default_branch": "main", "language": None})
    )

    get_repo_info(RepoRef(owner="octocat", repo="Hello-World"), token="secret-token")

    assert route.calls.last.request.headers["Authorization"] == "Bearer secret-token"


@respx.mock
def test_get_repo_info_omits_auth_header_when_no_token():
    route = respx.get(f"{API_BASE}/repos/octocat/Hello-World").mock(
        return_value=httpx.Response(200, json={"default_branch": "main", "language": None})
    )

    get_repo_info(RepoRef(owner="octocat", repo="Hello-World"))

    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_get_repo_info_raises_not_found_on_404():
    respx.get(f"{API_BASE}/repos/octocat/missing").mock(return_value=httpx.Response(404, json={}))

    with pytest.raises(GitHubNotFoundError):
        get_repo_info(RepoRef(owner="octocat", repo="missing"))


@respx.mock
def test_get_repo_info_raises_rate_limit_on_403_with_zero_remaining():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World").mock(
        return_value=httpx.Response(
            403,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"},
            json={"message": "API rate limit exceeded"},
        )
    )

    with pytest.raises(GitHubRateLimitError):
        get_repo_info(RepoRef(owner="octocat", repo="Hello-World"))


@respx.mock
def test_get_repo_info_raises_generic_error_on_other_403():
    respx.get(f"{API_BASE}/repos/octocat/private-repo").mock(
        return_value=httpx.Response(403, headers={"X-RateLimit-Remaining": "42"}, json={"message": "Forbidden"})
    )

    with pytest.raises(GitHubError):
        get_repo_info(RepoRef(owner="octocat", repo="private-repo"))


@respx.mock
def test_get_repo_info_raises_generic_error_on_server_error():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World").mock(return_value=httpx.Response(500, text="oops"))

    with pytest.raises(GitHubError):
        get_repo_info(RepoRef(owner="octocat", repo="Hello-World"))


# ---------------------------------------------------------------------------
# list_tree
# ---------------------------------------------------------------------------


@respx.mock
def test_list_tree_returns_entries_and_truncated_flag():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World/git/trees/main").mock(
        return_value=httpx.Response(
            200,
            json={
                "tree": [
                    {"path": "README.md", "type": "blob", "size": 42},
                    {"path": "src", "type": "tree"},
                    {"path": "src/main.py", "type": "blob", "size": 100},
                ],
                "truncated": False,
            },
        )
    )

    tree = list_tree(RepoRef(owner="octocat", repo="Hello-World"), branch="main")

    assert tree.truncated is False
    assert len(tree.entries) == 3
    assert tree.entries[0].path == "README.md"
    assert tree.entries[0].type == "blob"
    assert tree.entries[0].size == 42
    assert tree.entries[1].path == "src"
    assert tree.entries[1].type == "tree"
    assert tree.entries[1].size is None


@respx.mock
def test_list_tree_requests_recursive():
    route = respx.get(f"{API_BASE}/repos/octocat/Hello-World/git/trees/main").mock(
        return_value=httpx.Response(200, json={"tree": [], "truncated": False})
    )

    list_tree(RepoRef(owner="octocat", repo="Hello-World"), branch="main")

    assert route.calls.last.request.url.params["recursive"] == "1"


@respx.mock
def test_list_tree_reports_truncated_when_github_says_so():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World/git/trees/main").mock(
        return_value=httpx.Response(200, json={"tree": [], "truncated": True})
    )

    tree = list_tree(RepoRef(owner="octocat", repo="Hello-World"), branch="main")

    assert tree.truncated is True


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


@respx.mock
def test_read_file_decodes_base64_content():
    encoded = base64.b64encode("print('hi')\n".encode("utf-8")).decode("ascii")
    respx.get(f"{API_BASE}/repos/octocat/Hello-World/contents/main.py").mock(
        return_value=httpx.Response(200, json={"content": encoded, "encoding": "base64"})
    )

    content = read_file(RepoRef(owner="octocat", repo="Hello-World"), "main.py")

    assert content == "print('hi')\n"


@respx.mock
def test_read_file_passes_ref_param_when_branch_given():
    encoded = base64.b64encode(b"hi").decode("ascii")
    route = respx.get(f"{API_BASE}/repos/octocat/Hello-World/contents/main.py").mock(
        return_value=httpx.Response(200, json={"content": encoded, "encoding": "base64"})
    )

    read_file(RepoRef(owner="octocat", repo="Hello-World"), "main.py", branch="dev")

    assert route.calls.last.request.url.params["ref"] == "dev"


@respx.mock
def test_read_file_raises_when_path_is_a_directory():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World/contents/src").mock(
        return_value=httpx.Response(200, json=[{"name": "main.py", "type": "file"}])
    )

    with pytest.raises(GitHubError):
        read_file(RepoRef(owner="octocat", repo="Hello-World"), "src")


@respx.mock
def test_read_file_raises_not_found_on_404():
    respx.get(f"{API_BASE}/repos/octocat/Hello-World/contents/missing.py").mock(
        return_value=httpx.Response(404, json={})
    )

    with pytest.raises(GitHubNotFoundError):
        read_file(RepoRef(owner="octocat", repo="Hello-World"), "missing.py")
