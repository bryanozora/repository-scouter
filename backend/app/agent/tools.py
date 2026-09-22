"""Read-only agent tools: list_directory and read_file.

RepoTools binds these to one repo/branch/token for the duration of a scan
(repo context is session state, not something the LLM should ever have to
supply), and caches the recursive tree so the two tools share one GitHub
API call instead of one per directory listing.

Every public method returns a plain string -- either the result or a
message starting with "Error:" -- and never raises. A small local model
is unreliable at tool calling, so the agent loop must be able to feed any
tool result straight back into the conversation without a try/except of
its own.

File contents are wrapped in a <file_content> delimiter because they are
untrusted data from the scanned repo, not instructions (prompt-injection
defense -- see PLAN.md).
"""

from __future__ import annotations

from pathlib import PurePosixPath

from .. import github_client
from ..config import Settings, get_settings

# Directories that are never shown in a listing: vendored/build/cache
# output the agent has no business exploring.
SKIP_DIR_NAMES = {
    "node_modules",
    ".git",
    "dist",
    "build",
    "vendor",
    "__pycache__",
    ".venv",
    "venv",
    ".next",
    "target",
    ".pytest_cache",
    ".mypy_cache",
}

# Extensions read_file refuses outright, without ever fetching content.
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".jar", ".war", ".o", ".a",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".pdf", ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo",
    ".db", ".sqlite", ".sqlite3",
}

# Arbitrary but generous cap so one listing can't flood the model's context.
MAX_LIST_ENTRIES = 300

LIST_DIRECTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List the files and folders inside a directory of the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the repo root. Use '' for the root.",
                }
            },
            "required": ["path"],
        },
    },
}

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from the repository, optionally a specific line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the repo root."},
                "start": {
                    "type": "integer",
                    "description": "First line to read (1-based, inclusive). Omit to start from the top.",
                },
                "end": {
                    "type": "integer",
                    "description": "Last line to read (1-based, inclusive). Omit to read to the end.",
                },
            },
            "required": ["path"],
        },
    },
}

TOOLS = [LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA]


def _normalize_path(path: str) -> str:
    """'./src/' -> 'src', '' stays ''."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.strip("/")


def _validate_path(path: str) -> str | None:
    """Return an error message if path is unsafe to use, else None."""
    if path.startswith("/"):
        return "path must be relative to the repo root, not start with '/'"
    if ".." in path.split("/"):
        return "path must not contain '..'"
    return None


def _is_skipped(path: str) -> bool:
    return any(segment in SKIP_DIR_NAMES for segment in path.split("/") if segment)


def _coerce_optional_int(value: object, name: str) -> tuple[int | None, str | None]:
    """Best-effort int coercion, so a model sending "5" instead of 5 doesn't blow up."""
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None, f"{name} must be an integer"
    try:
        return int(str(value).strip()), None
    except ValueError:
        return None, f"{name} must be an integer"


class RepoTools:
    """list_directory/read_file bound to one repo, branch, and scan session."""

    def __init__(
        self,
        ref: github_client.RepoRef,
        branch: str,
        token: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.ref = ref
        self.branch = branch
        self.token = token
        settings = settings or get_settings()
        self.max_file_bytes = settings.max_file_bytes
        self.max_tool_result_chars = settings.max_tool_result_chars
        self._tree: github_client.RepoTree | None = None
        self._read_paths: set[str] = set()

    def _get_tree(self) -> github_client.RepoTree:
        """Fetch the full recursive tree once and reuse it for every call."""
        if self._tree is None:
            self._tree = github_client.list_tree(self.ref, self.branch, self.token)
        return self._tree

    def list_directory(self, path: str) -> str:
        try:
            return self._list_directory(path)
        except Exception as exc:  # tools must never raise into the agent loop
            return f"Error: unexpected failure listing {path!r}: {exc}"

    def _list_directory(self, path: str) -> str:
        norm = _normalize_path(path)
        err = _validate_path(norm)
        if err:
            return f"Error: {err}"
        if _is_skipped(norm):
            return f"Error: {norm!r} is a vendored/build directory and is not listed."

        try:
            tree = self._get_tree()
        except github_client.GitHubError as exc:
            return f"Error: {exc}"

        if norm and not any(e.path == norm and e.type == "tree" for e in tree.entries):
            if any(e.path == norm and e.type == "blob" for e in tree.entries):
                return f"Error: {norm!r} is a file, not a directory. Use read_file instead."
            return f"Error: path not found: {norm!r}"

        prefix = f"{norm}/" if norm else ""
        children: dict[str, github_client.TreeEntry] = {}
        for entry in tree.entries:
            if not entry.path.startswith(prefix):
                continue
            rest = entry.path[len(prefix) :]
            if not rest or "/" in rest or rest in SKIP_DIR_NAMES:
                continue
            children[rest] = entry

        if not children:
            return "(empty directory)"

        names = sorted(children)
        truncated_note = ""
        if len(names) > MAX_LIST_ENTRIES:
            remaining = len(names) - MAX_LIST_ENTRIES
            names = names[:MAX_LIST_ENTRIES]
            truncated_note = f"\n... and {remaining} more entries (truncated)"

        lines = []
        for name in names:
            entry = children[name]
            if entry.type == "tree":
                lines.append(f"dir  {name}")
            else:
                size = f" ({entry.size}B)" if entry.size is not None else ""
                lines.append(f"file {name}{size}")
        return "\n".join(lines) + truncated_note

    def read_file(self, path: str, start: object = None, end: object = None) -> str:
        try:
            return self._read_file(path, start, end)
        except Exception as exc:  # tools must never raise into the agent loop
            return f"Error: unexpected failure reading {path!r}: {exc}"

    def _read_file(self, path: str, start: object, end: object) -> str:
        norm = _normalize_path(path)
        err = _validate_path(norm)
        if err:
            return f"Error: {err}"

        if norm in self._read_paths:
            return f"Note: {norm!r} was already read earlier in this session; see the previous tool result."

        start_val, err = _coerce_optional_int(start, "start")
        if err:
            return f"Error: {err}"
        end_val, err = _coerce_optional_int(end, "end")
        if err:
            return f"Error: {err}"

        ext = PurePosixPath(norm).suffix.lower()
        if ext in BINARY_EXTENSIONS:
            return f"Error: {norm!r} looks like a binary file ({ext}); refusing to read it."

        try:
            tree = self._get_tree()
        except github_client.GitHubError as exc:
            return f"Error: {exc}"

        entry = next((e for e in tree.entries if e.path == norm), None)
        if entry is None:
            return f"Error: path not found: {norm!r}"
        if entry.type != "blob":
            return f"Error: {norm!r} is a directory, not a file. Use list_directory instead."
        if entry.size is not None and entry.size > self.max_file_bytes:
            return (
                f"Error: {norm!r} is {entry.size} bytes, over the {self.max_file_bytes}-byte cap; "
                "refusing to fetch it."
            )

        try:
            content = github_client.read_file(self.ref, norm, branch=self.branch, token=self.token)
        except github_client.GitHubError as exc:
            return f"Error: {exc}"

        if "\x00" in content:
            return f"Error: {norm!r} contains binary data (NUL byte found); refusing to display it."

        lines = content.splitlines()
        total_lines = len(lines)
        start_idx = max((start_val - 1) if start_val else 0, 0)
        end_idx = min(end_val, total_lines) if end_val else total_lines
        selected = lines[start_idx:end_idx]
        if not selected:
            return f"Error: no lines in that range (file has {total_lines} lines)"

        self._read_paths.add(norm)

        numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=start_idx + 1))
        truncated_note = ""
        if len(numbered) > self.max_tool_result_chars:
            numbered = numbered[: self.max_tool_result_chars]
            truncated_note = f"\n... (truncated to {self.max_tool_result_chars} characters)"

        header = f"File: {norm} (lines {start_idx + 1}-{start_idx + len(selected)} of {total_lines})"
        return f"{header}\n<file_content>\n{numbered}{truncated_note}\n</file_content>"


if __name__ == "__main__":
    import argparse

    from ..github_client import get_repo_info, parse_repo_url

    parser = argparse.ArgumentParser(description="Manually exercise list_directory/read_file against a real repo.")
    parser.add_argument("repo_url", help="e.g. github.com/pallets/click")
    parser.add_argument("--dir", default=None, help="List this directory (default: repo root)")
    parser.add_argument("--file", default=None, help="Read this file instead of listing a directory")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    cli_settings = get_settings()
    try:
        cli_ref = parse_repo_url(args.repo_url)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    cli_info = get_repo_info(cli_ref, token=cli_settings.github_token)
    print(f"[tools] {cli_ref.owner}/{cli_ref.repo}@{cli_info.default_branch}")

    repo_tools = RepoTools(cli_ref, branch=cli_info.default_branch, token=cli_settings.github_token, settings=cli_settings)
    if args.file:
        print(repo_tools.read_file(args.file, start=args.start, end=args.end))
    else:
        print(repo_tools.list_directory(args.dir or ""))
