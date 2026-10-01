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
from .findings import (
    ISSUE_TYPES,
    MAX_FINDINGS,
    REQUIRED_FIELDS,
    SEVERITIES,
    V1_CATEGORIES,
    Finding,
    check_evidence,
    format_validation_errors,
    validate_finding_args,
)
from .paths import normalize_path, validate_path
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

# candidate_search (approach 3): fixed terms the loop searches for before the
# model's first turn, and how much of that goes into the first message. Kept
# small -- ~20 lines is roughly 600 tokens of the 3300-token prompt budget.
CANDIDATE_TERMS = ("password", "secret", "token", "key", "SELECT", "execute(")
CANDIDATE_LINES_PER_TERM = 4
MAX_CANDIDATE_LINES = 20

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

# Short descriptions on purpose: every character here is resent on every
# call, and the model's context window is small (see docs/NOTES.md).
REPORT_FINDING_SCHEMA = {
    "type": "function",
    "function": {
        "name": "report_finding",
        "description": (
            "Record one security issue you have confirmed with read_file. "
            "Rejected if you have not seen the cited lines (via read_file or search_code)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": list(SEVERITIES)},
                "category": {"type": "string", "enum": list(V1_CATEGORIES)},
                "issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
                "file": {"type": "string", "description": "File path relative to the repo root."},
                "line_start": {"type": "integer", "description": "First line of the evidence (1-based)."},
                "line_end": {"type": "integer", "description": "Last line of the evidence, at most 30 lines after line_start."},
                "description": {"type": "string", "description": "What is wrong, in one or two sentences."},
                "suggestion": {"type": "string", "description": "How to fix it (optional)."},
                "confidence": {"type": "number", "description": "0.0 to 1.0 (optional, default 0.5)."},
            },
            "required": list(REQUIRED_FIELDS),
        },
    },
}

TOOLS = [LIST_DIRECTORY_SCHEMA, READ_FILE_SCHEMA, GET_DEPENDENCIES_SCHEMA, SEARCH_CODE_SCHEMA, REPORT_FINDING_SCHEMA]


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


def _format_line_ranges(lines: set[int]) -> str:
    """{1, 2, 3, 7, 8, 12} -> '1-3, 7-8, 12'."""
    spans: list[list[int]] = []
    for n in sorted(lines):
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans)


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
        source: object | None = None,
    ) -> None:
        self.ref = ref
        # Where trees and file contents come from: the GitHub API by default,
        # or anything with the same list_tree/read_file interface (e.g.
        # app.local_source.LocalRepoSource for the evals/repos fixtures).
        self._source = source if source is not None else github_client
        self.branch = branch
        self.token = token
        settings = settings or get_settings()
        self.max_file_bytes = settings.max_file_bytes
        self.max_tool_result_chars = settings.max_tool_result_chars
        self.tool_timeout_seconds = settings.tool_timeout_seconds
        self._tree: github_client.RepoTree | None = None
        # Line numbers actually returned to the model by read_file, per file.
        # Drives the "already read" note, and (M2b) report_finding only
        # accepts evidence from lines in here.
        self._shown_lines: dict[str, set[int]] = {}
        # Lines shown *in full* as search_code matches, per file. Kept apart
        # from _shown_lines so read_file's "already read" logic is unaffected;
        # report_finding accepts evidence from either.
        self._search_seen_lines: dict[str, set[int]] = {}
        self._listed_paths: set[str] = set()
        self._dependencies_result: str | None = None
        # Shared between read_file and search_code so a file fetched by one
        # isn't re-fetched by the other within the same scan.
        self._content_cache: dict[str, str] = {}
        # Findings recorded by report_finding. In-memory scan state only --
        # nothing is ever written back to the scanned repository.
        self._findings: list[Finding] = []

    @property
    def findings(self) -> list[Finding]:
        """A copy of the findings recorded so far (verified and flagged)."""
        return list(self._findings)

    def shown_lines(self, path: str) -> frozenset[int]:
        """Line numbers of `path` that read_file has shown in full this scan."""
        return frozenset(self._shown_lines.get(normalize_path(path), ()))

    def search_seen_lines(self, path: str) -> frozenset[int]:
        """Line numbers of `path` shown in full as search_code matches this scan."""
        return frozenset(self._search_seen_lines.get(normalize_path(path), ()))

    def forget_search_lines(self) -> None:
        """Called by the agent loop when an old search_code result is removed
        from the conversation. Clears every search-seen line, not just that
        search's -- deliberately conservative: over-forgetting only means the
        model must re-read before reporting, never that unseen lines count."""
        self._search_seen_lines.clear()

    def forget_shown(self, path: str) -> None:
        """Called by the agent loop when an old read_file result is removed from
        the conversation to fit the context window: those lines are no longer
        visible to the model, so they stop counting as evidence (report_finding)
        and may be read again without an "already read" note."""
        self._shown_lines.pop(normalize_path(path), None)

    def _get_tree(self) -> github_client.RepoTree:
        """Fetch the full recursive tree once and reuse it for every call."""
        if self._tree is None:
            self._tree = self._source.list_tree(self.ref, self.branch, self.token)
        return self._tree

    def list_directory(self, path: str) -> str:
        try:
            return self._list_directory(path)
        except Exception as exc:  # tools must never raise into the agent loop
            return f"Error: unexpected failure listing {path!r}: {exc}"

    def _list_directory(self, path: str) -> str:
        norm = normalize_path(path)
        err = validate_path(norm)
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
        norm = normalize_path(path)
        err = validate_path(norm)
        if err:
            return f"Error: {err}"

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
                content = self._source.read_file(self.ref, norm, branch=self.branch, token=self.token)
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

        # Only a request for lines that were *all* already shown is a repeat;
        # a different or wider range of the same file is new information.
        already_shown = self._shown_lines.get(norm, set())
        requested = set(range(start_idx + 1, start_idx + len(selected) + 1))
        if requested <= already_shown:
            return (
                f"Note: {norm!r} lines {_format_line_ranges(already_shown)} were already read earlier "
                "in this session; see the previous tool result."
            )

        # Cut at whole lines so "shown" means the model saw the entire line.
        numbered_lines = [f"{i}: {line}" for i, line in enumerate(selected, start=start_idx + 1)]
        kept: list[str] = []
        length = 0
        for numbered_line in numbered_lines:
            added = len(numbered_line) + (1 if kept else 0)
            if length + added > self.max_tool_result_chars:
                break
            kept.append(numbered_line)
            length += added

        first_line = start_idx + 1
        if kept:
            last_line = start_idx + len(kept)
            body = "\n".join(kept)
            already_shown.update(range(first_line, last_line + 1))
            self._shown_lines[norm] = already_shown
        else:
            # A single line longer than the whole cap (minified code): show a
            # cut-off piece of it, but don't count it as shown.
            last_line = first_line
            body = numbered_lines[0][: self.max_tool_result_chars]

        truncated_note = ""
        if len(kept) < len(selected):
            next_line = last_line + 1 if kept else first_line + 1
            truncated_note = (
                f"\n... (truncated to {self.max_tool_result_chars} characters. "
                f"To see more, call read_file('{norm}', start={next_line}).)"
            )

        header = f"File: {norm} (lines {first_line}-{last_line} of {total_lines})"
        return f"{header}\n<file_content>\n{body}{truncated_note}\n</file_content>"

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
                    content = self._source.read_file(self.ref, entry.path, branch=self.branch, token=self.token)
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
            matches, files_fetched, total_candidates, bounded_reason = self._find_matches(query, now)
        except github_client.GitHubError as exc:
            return f"Error: {exc}"
        self._mark_search_seen(matches)

        if not matches:
            body = f"No matches for {query!r} in {files_fetched} file(s) searched."
        else:
            body = "\n".join(f"{path}:{lineno}: {snippet}" for path, lineno, snippet, _ in matches)

        if bounded_reason:
            body += (
                f"\n\n(Search stopped early: {bounded_reason}. Results may be incomplete -- "
                f"searched {files_fetched} of {total_candidates} candidate file(s).)"
            )

        return f"<file_content>\n{body}\n</file_content>"

    def candidate_search(
        self,
        terms: tuple[str, ...],
        per_term: int = CANDIDATE_LINES_PER_TERM,
        max_lines: int = MAX_CANDIDATE_LINES,
        now: Callable[[], float] = time.monotonic,
    ) -> str:
        """Approach 3: fixed searches run by the agent loop itself before the
        model's first turn (not a tool the model calls). Returns one compact,
        de-duplicated block of matching lines -- a few per term, round-robin,
        so one noisy term ("key") can't crowd out the rest -- or "" if nothing
        matched. Lines shown here in full count as seen evidence, exactly like
        a search_code result. Never raises."""
        try:
            per_term_hits: list[list[tuple[str, int, str, bool]]] = []
            for term in terms:
                matches, _, _, _ = self._find_matches(term, now)
                per_term_hits.append(matches)
        except Exception:  # an auto-search failing must never stop the scan
            return ""

        chosen: list[tuple[str, int, str, bool]] = []
        seen_keys: set[tuple[str, int]] = set()
        for rank in range(per_term):
            for hits in per_term_hits:
                if rank < len(hits) and len(chosen) < max_lines:
                    path, lineno, snippet, full = hits[rank]
                    if (path, lineno) not in seen_keys:
                        seen_keys.add((path, lineno))
                        chosen.append((path, lineno, snippet, full))
        if not chosen:
            return ""
        chosen.sort(key=lambda m: (m[0], m[1]))
        self._mark_search_seen(chosen)
        body = "\n".join(f"{path}:{lineno}: {snippet}" for path, lineno, snippet, _ in chosen)
        return f"<file_content>\n{body}\n</file_content>"

    def _mark_search_seen(self, matches: list[tuple[str, int, str, bool]]) -> None:
        """Lines shown in full (minus indentation) count as seen evidence."""
        for path, lineno, _, full in matches:
            if full:
                self._search_seen_lines.setdefault(path, set()).add(lineno)

    def _find_matches(
        self, query: str, now: Callable[[], float]
    ) -> tuple[list[tuple[str, int, str, bool]], int, int, str | None]:
        """(matches, files_fetched, total_candidates, bounded_reason) for one
        literal, case-insensitive query. Each match is (path, lineno, snippet,
        shown_in_full). Raises GitHubError if the tree can't be fetched."""
        tree = self._get_tree()
        candidates = _select_search_candidates(tree.entries)
        deadline = now() + self.tool_timeout_seconds

        matches: list[tuple[str, int, str, bool]] = []
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
                            self._source.read_file, self.ref, entry.path, branch=self.branch, token=self.token
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
                            full = len(snippet) <= MAX_SEARCH_LINE_CHARS
                            if not full:
                                snippet = snippet[:MAX_SEARCH_LINE_CHARS] + "..."
                            matches.append((entry.path, lineno, snippet, full))
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

        return matches, files_fetched, len(candidates), bounded_reason

    def report_finding(self, args: dict, source: str = "tool_call") -> str:
        """Validate, read-gate, evidence-check and record one finding.

        source says which path it came in through -- "tool_call" (a real
        report_finding call), "parsed_text" (a finding the model wrote as
        text), or "forced_json" (the loop's final structured reporting step).
        Every path goes through exactly the same checks."""
        try:
            return self._report_finding(args, source)
        except Exception as exc:  # tools must never raise into the agent loop
            return f"Error: unexpected failure recording the finding: {exc}"

    def _report_finding(self, args: dict, source: str) -> str:
        finding, errors = validate_finding_args(args)
        if errors:
            return format_validation_errors(errors)
        assert finding is not None
        finding.source = source

        if len(self._findings) >= MAX_FINDINGS:
            return (
                f"Error: the limit of {MAX_FINDINGS} findings per scan was reached; nothing was recorded. "
                "Give your final answer now."
            )

        path, start, end = finding.file, finding.line_start, finding.line_end
        read_hint = f"read_file('{path}', start={start}, end={end})"

        try:
            tree = self._get_tree()
        except github_client.GitHubError as exc:
            return f"Error: {exc}"
        entry = next((e for e in tree.entries if e.path == path), None)
        if entry is None:
            return format_validation_errors([f"file: {path!r} was not found in the repository"])
        if entry.type != "blob":
            return format_validation_errors([f"file: {path!r} is a directory, not a file"])

        # Evidence must be lines the model actually saw: shown by read_file, or
        # shown in full as a search_code match. Filenames and truncated search
        # snippets don't count.
        seen = self._shown_lines.get(path, set()) | self._search_seen_lines.get(path, set())
        content = self._content_cache.get(path)
        if not seen or content is None:
            return format_validation_errors(
                [
                    f"file: you have not seen any lines of {path!r} in this scan (via read_file or a "
                    f"search_code match). Call {read_hint} first, then report it"
                ]
            )
        lines = content.splitlines()
        if end > len(lines):
            return format_validation_errors(
                [f"line_end: {path!r} has only {len(lines)} lines, so lines {start}-{end} do not exist"]
            )
        if not set(range(start, end + 1)) <= seen:
            return format_validation_errors(
                [
                    f"line_start: you have only seen lines {_format_line_ranges(seen)} of {path!r}, which do "
                    f"not include all of lines {start}-{end}. Call {read_hint} first, then report it"
                ]
            )

        finding.verified, finding.verification_note = check_evidence(finding.issue_type, path, lines, start, end)

        # An overlapping report of the same kind in the same file: a duplicate
        # if the earlier one verified; otherwise this is a corrected retry and
        # replaces it.
        overlapping = [
            f
            for f in self._findings
            if f.file == path and f.issue_type == finding.issue_type and f.line_start <= end and start <= f.line_end
        ]
        verified_dup = next((f for f in overlapping if f.verified), None)
        if verified_dup is not None:
            number = self._findings.index(verified_dup) + 1
            return (
                f"Note: this was already recorded as finding #{number} ({path} lines "
                f"{verified_dup.line_start}-{verified_dup.line_end}); not recorded again."
            )
        for stale in overlapping:
            self._findings.remove(stale)

        self._findings.append(finding)
        number = len(self._findings)
        if finding.verified:
            return f"Recorded finding #{number} (verified: {finding.verification_note})."
        return (
            f"Recorded finding #{number}, but flagged UNVERIFIED: {finding.verification_note}. It will be listed "
            "separately as unverified. If you cited the wrong lines, call report_finding again with the correct range."
        )


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
