"""A read-only, local-directory stand-in for github_client.

Exposes the same list_tree/read_file interface RepoTools uses, so a scan
can run against a directory on disk -- the planted-bug fixtures in
evals/repos/ -- without pushing them to GitHub. Production scans still go
through github_client; this is for evaluation and development.

Read-only like every other repo access path: it lists and reads files, and
refuses any path that resolves outside the given root.
"""

from __future__ import annotations

from pathlib import Path

from .github_client import GitHubError, GitHubNotFoundError, RepoRef, RepoTree, TreeEntry


class LocalRepoSource:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"local repo directory not found: {self.root}")

    def list_tree(self, ref: RepoRef, branch: str, token: str | None = None) -> RepoTree:
        entries: list[TreeEntry] = []
        for path in sorted(self.root.rglob("*")):
            rel = path.relative_to(self.root).as_posix()
            if rel == ".git" or rel.startswith(".git/"):
                continue
            if path.is_dir():
                entries.append(TreeEntry(path=rel, type="tree"))
            elif path.is_file():
                entries.append(TreeEntry(path=rel, type="blob", size=path.stat().st_size))
        return RepoTree(entries=entries, truncated=False)

    def read_file(self, ref: RepoRef, path: str, branch: str | None = None, token: str | None = None) -> str:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise GitHubError(f"{path!r} is outside the repository")
        if not target.exists():
            raise GitHubNotFoundError(f"path not found: {path!r}")
        if target.is_dir():
            raise GitHubError(f"{path!r} is a directory, not a file")
        return target.read_bytes().decode("utf-8", errors="replace")
