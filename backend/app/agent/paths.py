"""Repo-relative path helpers shared by the read tools and report_finding.

Lives in its own module (rather than tools.py) so findings.py can use the
same normalization/safety rules without importing tools.py, which itself
depends on findings.py.
"""

from __future__ import annotations


def normalize_path(path: str) -> str:
    """'./src/' -> 'src', '' stays ''.

    A leading '/' is deliberately kept so validate_path can reject the path
    as absolute; stripping it here used to turn '/etc/passwd' into the
    repo-relative 'etc/passwd'. A bare '/' still means the root.
    """
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return "" if p == "/" else p.rstrip("/")


def validate_path(path: str) -> str | None:
    """Return an error message if path is unsafe to use, else None."""
    if path.startswith("/"):
        return "path must be relative to the repo root, not start with '/'"
    if ".." in path.split("/"):
        return "path must not contain '..'"
    return None
