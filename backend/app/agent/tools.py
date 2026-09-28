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

import concurrent.futures
import time
from pathlib import PurePosixPath
from typing import Callable

from . import manifests
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

# get_dependencies: cap on how many manifests get read (a JS monorepo can
# have dozens of package.json files), and how much of an unparsed (raw-only)
# manifest's content gets shown per file.
MAX_MANIFESTS = 10
MAX_RAW_MANIFEST_CHARS = 3000

# search_code candidate selection: extensions/filenames used to rank and
# filter files before any of them get fetched.
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rb", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rs", ".php", ".swift", ".scala",
}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml"}
CONFIG_FILENAMES = {"Dockerfile", "Makefile", "docker-compose.yml"}
DOC_EXTENSIONS = {".md", ".rst", ".txt"}
GENERATED_OR_MINIFIED_SUFFIXES = (".min.js", ".min.css")
LOCK_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "Cargo.lock", "go.sum", "composer.lock",
}

# search_code budgets: whichever of these is hit first stops the search.
MAX_SEARCH_FILES = 20
MAX_SEARCH_BYTES = 200_000
MAX_SEARCH_RESULTS = 20
MAX_SEARCH_LINE_CHARS = 200
# Small bounded concurrency for candidate fetches -- network-bound work, so a
# few in flight at once meaningfully improves coverage within the time budget
# without needing a large worker pool.
SEARCH_CONCURRENCY = 5

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

GET_DEPENDENCIES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_dependencies",
        "description": (
            "Find and summarize the repository's dependency manifests "
            "(package.json, requirements.txt, pyproject.toml, etc.). Takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

SEARCH_CODE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Search a bounded, priority-ordered subset of the repository's text files "
            "(source code first) for a literal substring, case-insensitive. Not "
            "guaranteed to cover every file on a large repo."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The literal text to search for."}},
            "required": ["query"],
        },
    },
}

TOOLS = [LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA, GET_DEPENDENCIES_SCHEMA, SEARCH_CODE_SCHEMA]


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


def _is_low_value_for_search(path: str) -> bool:
    """Minified/generated/lock files: never worth fetching for a text search."""
    name = PurePosixPath(path).name
    if name in LOCK_FILENAMES:
        return True
    return any(name.endswith(suffix) for suffix in GENERATED_OR_MINIFIED_SUFFIXES)


def _search_priority(path: str) -> int:
    """Lower = fetched earlier. 0=source, 1=config, 2=docs/tests/examples, 3=other."""
    path_lower = path.lower()
    name = PurePosixPath(path).name
    ext = PurePosixPath(path).suffix.lower()

    is_test = (
        "/test" in path_lower
        or path_lower.startswith("test")
        or name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )
    is_example = any(marker in path_lower for marker in ("/example", "/sample", "/demo"))
    is_doc = ext in DOC_EXTENSIONS

    if is_test or is_example or is_doc:
        return 2
    if ext in SOURCE_EXTENSIONS:
        return 0
    if ext in CONFIG_EXTENSIONS or name in CONFIG_FILENAMES:
        return 1
    return 3


def _select_search_candidates(entries: list[github_client.TreeEntry]) -> list[github_client.TreeEntry]:
    """Eligible files, priority-ordered (source > config > docs/tests/examples > other)."""
    eligible = [
        e
        for e in entries
        if e.type == "blob"
        and not _is_skipped(e.path)
        and PurePosixPath(e.path).suffix.lower() not in BINARY_EXTENSIONS
        and not _is_low_value_for_search(e.path)
    ]
    eligible.sort(key=lambda e: (_search_priority(e.path), e.path.count("/"), e.path))
    return eligible


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
        self.tool_timeout_seconds = settings.tool_timeout_seconds
        self._tree: github_client.RepoTree | None = None
        self._read_paths: set[str] = set()
        self._listed_paths: set[str] = set()
        self._dependencies_result: str | None = None
        # Shared between read_file and search_code so a file fetched by one
        # isn't re-fetched by the other within the same scan.
        self._content_cache: dict[str, str] = {}

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
        if norm in self._listed_paths:
            return f"Note: {norm!r} was already listed earlier in this session; use read_file on one of its files to continue."
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
            result = "(empty directory)"
        else:
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
            result = "\n".join(lines) + truncated_note

        self._listed_paths.add(norm)
        return result

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

        content = self._content_cache.get(norm)
        if content is None:
            try:
                content = github_client.read_file(self.ref, norm, branch=self.branch, token=self.token)
            except github_client.GitHubError as exc:
                return f"Error: {exc}"
            self._content_cache[norm] = content

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

    def get_dependencies(self) -> str:
        try:
            return self._get_dependencies()
        except Exception as exc:  # tools must never raise into the agent loop
            return f"Error: unexpected failure reading dependencies: {exc}"

    def _get_dependencies(self) -> str:
        if self._dependencies_result is not None:
            return "Note: dependencies were already fetched earlier in this session; see the previous tool result."

        try:
            tree = self._get_tree()
        except github_client.GitHubError as exc:
            return f"Error: {exc}"

        candidates = [
            e
            for e in tree.entries
            if e.type == "blob" and PurePosixPath(e.path).name in manifests.KNOWN_MANIFESTS and not _is_skipped(e.path)
        ]
        candidates.sort(key=lambda e: (e.path.count("/"), e.path))

        truncated_note = ""
        if len(candidates) > MAX_MANIFESTS:
            truncated_note = f"\n\n... and {len(candidates) - MAX_MANIFESTS} more manifest(s) not shown (truncated)"
            candidates = candidates[:MAX_MANIFESTS]

        if not candidates:
            result = "No known dependency manifests found in this repository."
            self._dependencies_result = result
            return result

        sections = []
        for entry in candidates:
            name = PurePosixPath(entry.path).name
            if entry.size is not None and entry.size > self.max_file_bytes:
                sections.append(f"{entry.path}: (skipped, {entry.size} bytes over the {self.max_file_bytes}-byte cap)")
                continue

            content = self._content_cache.get(entry.path)
            if content is None:
                try:
                    content = github_client.read_file(self.ref, entry.path, branch=self.branch, token=self.token)
                except github_client.GitHubError as exc:
                    sections.append(f"{entry.path}: (could not fetch: {exc})")
                    continue
                self._content_cache[entry.path] = content

            if "\x00" in content:
                sections.append(f"{entry.path}: (skipped, contains binary data)")
                continue

            parser = manifests.PARSERS.get(name)
            if parser:
                summary = parser(content)
            else:
                summary = content
                if len(summary) > MAX_RAW_MANIFEST_CHARS:
                    summary = summary[:MAX_RAW_MANIFEST_CHARS] + f"\n... (truncated to {MAX_RAW_MANIFEST_CHARS} characters)"
            sections.append(f"{entry.path}:\n{summary}")

        body = "\n\n".join(sections) + truncated_note
        if len(body) > self.max_tool_result_chars:
            body = body[: self.max_tool_result_chars] + f"\n... (truncated to {self.max_tool_result_chars} characters)"

        result = f"<file_content>\n{body}\n</file_content>"
        self._dependencies_result = result
        return result

    def search_code(self, query: str, now: Callable[[], float] = time.monotonic) -> str:
        try:
            return self._search_code(query, now)
        except Exception as exc:  # tools must never raise into the agent loop
            return f"Error: unexpected failure searching for {query!r}: {exc}"

    def _search_code(self, query: str, now: Callable[[], float]) -> str:
        if not query or not query.strip():
            return "Error: query must not be empty"

        try:
            tree = self._get_tree()
        except github_client.GitHubError as exc:
            return f"Error: {exc}"

        candidates = _select_search_candidates(tree.entries)
        deadline = now() + self.tool_timeout_seconds

        matches: list[tuple[str, int, str]] = []
        files_fetched = 0
        bytes_fetched = 0
        bounded_reason: str | None = None
        index = 0

        # Fetch candidates in small concurrent batches (network-bound work, so
        # a few files in flight at once meaningfully improves coverage within
        # the time budget) while still respecting every cap between batches.
        with concurrent.futures.ThreadPoolExecutor(max_workers=SEARCH_CONCURRENCY) as executor:
            while index < len(candidates):
                if files_fetched >= MAX_SEARCH_FILES:
                    bounded_reason = f"reached the {MAX_SEARCH_FILES}-file search limit"
                    break
                if bytes_fetched >= MAX_SEARCH_BYTES:
                    bounded_reason = f"reached the {MAX_SEARCH_BYTES}-byte search budget"
                    break
                remaining_time = deadline - now()
                if remaining_time <= 0:
                    bounded_reason = f"ran out of time (search is capped at {self.tool_timeout_seconds}s)"
                    break

                batch: list[github_client.TreeEntry] = []
                while (
                    index < len(candidates)
                    and len(batch) < SEARCH_CONCURRENCY
                    and files_fetched + len(batch) < MAX_SEARCH_FILES
                ):
                    batch.append(candidates[index])
                    index += 1

                batch_futures: list[tuple[github_client.TreeEntry, concurrent.futures.Future]] = []
                for entry in batch:
                    cached = self._content_cache.get(entry.path)
                    if cached is not None:
                        fut: concurrent.futures.Future = concurrent.futures.Future()
                        fut.set_result(cached)
                    else:
                        fut = executor.submit(
                            github_client.read_file, self.ref, entry.path, branch=self.branch, token=self.token
                        )
                    batch_futures.append((entry, fut))

                done, not_done = concurrent.futures.wait(
                    [fut for _, fut in batch_futures], timeout=max(remaining_time, 0)
                )

                for entry, fut in batch_futures:
                    if fut not in done:
                        continue  # still running when the deadline hit -- its thread
                        # keeps going in the background (can't cancel a running
                        # thread in Python), but we don't wait on or count it.
                    try:
                        content = fut.result()
                    except github_client.GitHubError:
                        continue  # unreadable file -- skip it, don't abort the whole search
                    if "\x00" in content:
                        continue
                    self._content_cache[entry.path] = content

                    files_fetched += 1
                    bytes_fetched += len(content)

                    for lineno, line in enumerate(content.splitlines(), start=1):
                        if query.lower() in line.lower():
                            snippet = line.strip()
                            if len(snippet) > MAX_SEARCH_LINE_CHARS:
                                snippet = snippet[:MAX_SEARCH_LINE_CHARS] + "..."
                            matches.append((entry.path, lineno, snippet))
                            if len(matches) >= MAX_SEARCH_RESULTS:
                                break
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        break

                if not_done:
                    bounded_reason = f"ran out of time (search is capped at {self.tool_timeout_seconds}s)"
                    break
                if len(matches) >= MAX_SEARCH_RESULTS:
                    bounded_reason = f"reached the {MAX_SEARCH_RESULTS}-match limit"
                    break

        if not matches:
            body = f"No matches for {query!r} in {files_fetched} file(s) searched."
        else:
            body = "\n".join(f"{path}:{lineno}: {snippet}" for path, lineno, snippet in matches)

        if bounded_reason:
            body += (
                f"\n\n(Search stopped early: {bounded_reason}. Results may be incomplete -- "
                f"searched {files_fetched} of {len(candidates)} candidate file(s).)"
            )

        return f"<file_content>\n{body}\n</file_content>"


if __name__ == "__main__":
    import argparse

    from ..github_client import get_repo_info, parse_repo_url

    parser = argparse.ArgumentParser(description="Manually exercise the read-only tools against a real repo.")
    parser.add_argument("repo_url", help="e.g. github.com/pallets/click")
    parser.add_argument("--dir", default=None, help="List this directory (default: repo root)")
    parser.add_argument("--file", default=None, help="Read this file instead of listing a directory")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dependencies", action="store_true", help="Call get_dependencies() instead")
    parser.add_argument("--search", default=None, help="Call search_code() with this query instead")
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
    if args.dependencies:
        print(repo_tools.get_dependencies())
    elif args.search is not None:
        print(repo_tools.search_code(args.search))
    elif args.file:
        print(repo_tools.read_file(args.file, start=args.start, end=args.end))
    else:
        print(repo_tools.list_directory(args.dir or ""))
