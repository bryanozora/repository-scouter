"""GitHub repo access layer.

The agent inspects repositories through the GitHub REST API rather than
`git clone` (see PLAN.md's key design decision), so this module is the one
place that talks to api.github.com. GET requests only -- nothing here
writes to, comments on, or modifies anything on GitHub.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

import httpx

API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT = 30.0

_REPO_URL_RE = re.compile(
    r"^(?:https?://)?github\.com/"
    r"(?P<owner>[A-Za-z0-9-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?/?$",
    re.IGNORECASE,
)


class GitHubError(Exception):
    """Base class for GitHub API errors."""


class GitHubNotFoundError(GitHubError):
    """The repo, branch, or path does not exist (HTTP 404)."""


class GitHubRateLimitError(GitHubError):
    """The GitHub API rate limit has been hit (HTTP 403 + remaining=0)."""


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str


@dataclass
class RepoInfo:
    default_branch: str
    language: str | None


@dataclass
class TreeEntry:
    path: str
    type: str  # "blob" (file) or "tree" (directory)
    size: int | None = None


@dataclass
class RepoTree:
    entries: list[TreeEntry]
    truncated: bool  # True when GitHub cut off a huge tree (see list_tree)


def parse_repo_url(url: str) -> RepoRef:
    """Parse a github.com/owner/repo URL into a RepoRef.

    Accepts the usual forms: with or without a scheme, a trailing slash,
    or a `.git` suffix. Anything else (wrong host, missing repo, extra
    path segments) raises ValueError.
    """
    match = _REPO_URL_RE.match(url.strip())
    if not match:
        raise ValueError(f"Not a github.com repo URL: {url!r}")
    return RepoRef(owner=match.group("owner"), repo=match.group("repo"))


def _get_json(url: str, token: str | None, params: dict[str, str] | None = None) -> Any:
    """GET a GitHub API URL and return its parsed JSON, or raise a clear error."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        resp = client.get(url, headers=headers, params=params)

    if resp.status_code == 404:
        raise GitHubNotFoundError(f"Not found on GitHub: {url}")
    if resp.status_code == 403:
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = resp.headers.get("X-RateLimit-Reset", "unknown")
            raise GitHubRateLimitError(
                f"GitHub API rate limit exceeded (resets at unix time {reset}). "
                "Set GITHUB_TOKEN in .env to raise the limit."
            )
        raise GitHubError(f"GitHub API forbidden for {url}: {resp.text[:200]}")
    if resp.is_error:
        raise GitHubError(f"GitHub API error {resp.status_code} for {url}: {resp.text[:200]}")
    return resp.json()


def get_repo_info(ref: RepoRef, token: str | None = None) -> RepoInfo:
    """Return the repo's default branch and primary language."""
    data = _get_json(f"{API_BASE}/repos/{ref.owner}/{ref.repo}", token)
    return RepoInfo(default_branch=data["default_branch"], language=data.get("language"))


def list_tree(ref: RepoRef, branch: str, token: str | None = None) -> RepoTree:
    """List every file/folder in the repo at `branch` via the git trees API.

    Uses recursive=1 to get the whole tree in one call. GitHub truncates
    the response for very large repos (over ~100k entries or 7MB); when
    that happens `truncated` is True and the tree is incomplete.
    """
    data = _get_json(
        f"{API_BASE}/repos/{ref.owner}/{ref.repo}/git/trees/{branch}",
        token,
        params={"recursive": "1"},
    )
    entries = [
        TreeEntry(path=entry["path"], type=entry["type"], size=entry.get("size"))
        for entry in data.get("tree", [])
    ]
    return RepoTree(entries=entries, truncated=bool(data.get("truncated", False)))


def read_file(ref: RepoRef, path: str, branch: str | None = None, token: str | None = None) -> str:
    """Read one file's contents via the contents API and decode it from base64."""
    params = {"ref": branch} if branch else None
    data = _get_json(f"{API_BASE}/repos/{ref.owner}/{ref.repo}/contents/{path}", token, params=params)

    if isinstance(data, list):
        raise GitHubError(f"{path!r} is a directory, not a file")
    if data.get("encoding") != "base64" or "content" not in data:
        raise GitHubError(f"Unexpected contents API response for {path!r}: no base64 content")

    raw_bytes = base64.b64decode(data["content"])
    return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    import sys

    from .config import get_settings

    if len(sys.argv) < 2:
        print("Usage: python -m app.github_client <repo_url> [path]")
        raise SystemExit(1)

    token = get_settings().github_token
    try:
        ref = parse_repo_url(sys.argv[1])
    except ValueError as exc:
        print(
            f"Error: {exc}\n"
            "Expected a github.com repo URL, e.g. 'github.com/owner/repo' "
            "(https:// optional, trailing slash or .git allowed).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"[github] {ref.owner}/{ref.repo}")

    if len(sys.argv) >= 3:
        path = sys.argv[2]
        print(read_file(ref, path, token=token))
    else:
        info = get_repo_info(ref, token=token)
        print(f"[github] default_branch={info.default_branch} language={info.language}")

        tree = list_tree(ref, branch=info.default_branch, token=token)
        print(f"[github] {len(tree.entries)} entries, truncated={tree.truncated}")
        for entry in tree.entries:
            size = f" ({entry.size}B)" if entry.size is not None else ""
            print(f"  {entry.type:4} {entry.path}{size}")
