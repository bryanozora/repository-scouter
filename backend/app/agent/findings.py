"""Structured findings: validation of report_finding arguments, and a
deterministic evidence check against the cited lines.

Both entry points are pure functions -- no GitHub, no LLM, no RepoTools --
so they can be tested on plain dicts and lists of lines:

  - validate_finding_args(args) -> (Finding | None, errors)
      Strict on *values* (enums, line range, confidence 0-1, non-empty
      text), tolerant on *formatting* ("High", "10", "SQL injection"), and
      it reports every problem at once so a small local model can fix them
      all in a single retry instead of one per step.

  - check_evidence(issue_type, path, lines, line_start, line_end) -> (ok, note)
      A plausibility check, not a vulnerability scanner: does the cited
      range (plus a little slack, since small models are often off by a
      line or two) actually contain the kind of code the finding claims?
      Failing it doesn't delete a finding -- the caller flags it as
      unverified -- so a regex miss costs visibility, not data.

v1 scope is deliberately narrow (PLAN.md section 4): only hardcoded
secrets and SQL injection, both in the "security" category.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from .paths import normalize_path, validate_path

SEVERITIES = ("high", "medium", "low", "info")
CATEGORIES = ("security", "bug", "performance", "maintainability")
V1_CATEGORIES = ("security",)
ISSUE_TYPES = ("hardcoded_secret", "sql_injection")

# Evidence should point at the specific lines, not a whole file.
MAX_LINE_SPAN = 30
# Keeps findings (and the tokens they cost when echoed back) short.
MAX_TEXT_CHARS = 500
# How far outside the cited range the evidence check still looks.
EVIDENCE_SLACK = 2
# Per-scan cap, so a confused model can't flood the report.
MAX_FINDINGS = 15

# Every field report_finding accepts, in schema order.
FIELDS = (
    "severity",
    "category",
    "issue_type",
    "file",
    "line_start",
    "line_end",
    "description",
    "suggestion",
    "confidence",
)
# confidence and suggestion are optional: the 7B model left each of them out
# in real runs, and a sensible default beats burning a step on a retry.
OPTIONAL_FIELDS = ("suggestion", "confidence")
REQUIRED_FIELDS = tuple(f for f in FIELDS if f not in OPTIONAL_FIELDS)
DEFAULT_CONFIDENCE = 0.5
DEFAULT_SUGGESTION = "No fix was suggested by the model; review the cited lines."


@dataclass
class Finding:
    """One validated finding. verified/verification_note are set by the caller
    after check_evidence runs."""

    severity: str
    category: str
    issue_type: str
    file: str
    line_start: int
    line_end: int
    description: str
    suggestion: str
    confidence: float
    verified: bool = False
    verification_note: str = ""
    # Which path recorded it: "tool_call", "parsed_text" or "forced_json".
    source: str = "tool_call"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_finding_args(args: object) -> tuple[Finding | None, list[str]]:
    """Validate raw report_finding arguments. Returns (finding, []) or (None, errors)."""
    if not isinstance(args, dict):
        return None, ["arguments must be a JSON object with the finding's fields"]

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in args or args[field] is None:
            errors.append(f"{field}: missing required field")

    def present(field: str) -> bool:
        return field in args and args[field] is not None

    severity = _check_enum(args, "severity", SEVERITIES, errors) if present("severity") else None
    category = _check_category(args, errors) if present("category") else None
    issue_type = _check_issue_type(args, errors) if present("issue_type") else None
    file = _check_file(args, errors) if present("file") else None
    line_start, line_end = _check_lines(args, errors)
    description = _check_text(args, "description", errors) if present("description") else None
    suggestion = _check_text(args, "suggestion", errors) if present("suggestion") else DEFAULT_SUGGESTION
    confidence = _check_confidence(args, errors) if present("confidence") else DEFAULT_CONFIDENCE

    # Missing-field errors were collected first; re-sort so every message
    # appears in schema order, which reads more naturally to the model.
    order = {field: i for i, field in enumerate(FIELDS)}
    errors.sort(key=lambda e: order.get(e.split(":", 1)[0], len(order)))

    if errors:
        return None, errors
    return (
        Finding(
            severity=severity,
            category=category,
            issue_type=issue_type,
            file=file,
            line_start=line_start,
            line_end=line_end,
            description=description,
            suggestion=suggestion,
            confidence=confidence,
        ),
        [],
    )


def format_validation_errors(errors: list[str]) -> str:
    """Render validation errors as one tool result the model can act on."""
    bullets = "\n".join(f"- {e}" for e in errors)
    return f"Error: report_finding rejected, nothing was recorded. Fix these and call it again:\n{bullets}"


def _normalize_enum(value: str) -> str:
    return value.strip().lower()


def _check_enum(args: dict, field: str, allowed: tuple[str, ...], errors: list[str]) -> str | None:
    value = args[field]
    if not isinstance(value, str):
        errors.append(f"{field}: must be a string, one of: {', '.join(allowed)}")
        return None
    norm = _normalize_enum(value)
    if norm not in allowed:
        errors.append(f"{field}: {value!r} is not allowed; use one of: {', '.join(allowed)}")
        return None
    return norm


def _check_category(args: dict, errors: list[str]) -> str | None:
    category = _check_enum(args, "category", CATEGORIES, errors)
    if category is not None and category not in V1_CATEGORIES:
        errors.append(
            f"category: {category!r} is valid but out of scope; this scan only reports "
            f"{', '.join(V1_CATEGORIES)} issues, so use 'security'"
        )
        return None
    return category


def _check_issue_type(args: dict, errors: list[str]) -> str | None:
    value = args["issue_type"]
    allowed = ", ".join(ISSUE_TYPES)
    if not isinstance(value, str):
        errors.append(f"issue_type: must be a string, one of: {allowed}")
        return None
    norm = re.sub(r"[\s-]+", "_", value.strip().lower())
    if norm not in ISSUE_TYPES:
        errors.append(f"issue_type: {value!r} is not supported; this scan only reports: {allowed}")
        return None
    return norm


def _check_file(args: dict, errors: list[str]) -> str | None:
    value = args["file"]
    if not isinstance(value, str) or not value.strip():
        errors.append("file: must be a file path relative to the repo root, e.g. 'src/app.py'")
        return None
    norm = normalize_path(value)
    err = validate_path(norm)
    if err:
        errors.append(f"file: {err}")
        return None
    if not norm:
        errors.append("file: must be a file path relative to the repo root, e.g. 'src/app.py'")
        return None
    return norm


def _coerce_int(value: object) -> int | None:
    """int, integral float, or numeric string -> int; anything else -> None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _check_lines(args: dict, errors: list[str]) -> tuple[int | None, int | None]:
    values: dict[str, int | None] = {}
    for field in ("line_start", "line_end"):
        if field not in args or args[field] is None:
            values[field] = None
            continue
        n = _coerce_int(args[field])
        if n is None:
            errors.append(f"{field}: must be a whole line number, got {args[field]!r}")
        elif n < 1:
            errors.append(f"{field}: must be >= 1 (line numbers start at 1), got {n}")
            n = None
        values[field] = n

    start, end = values["line_start"], values["line_end"]
    if start is not None and end is not None:
        if end < start:
            errors.append(f"line_end: must be >= line_start (got line_start={start}, line_end={end})")
            return None, None
        span = end - start + 1
        if span > MAX_LINE_SPAN:
            errors.append(
                f"line_end: the range covers {span} lines; cite at most {MAX_LINE_SPAN} lines -- "
                "only the specific lines that show the issue"
            )
            return None, None
    return start, end


def _check_text(args: dict, field: str, errors: list[str]) -> str | None:
    value = args[field]
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field}: must be a non-empty string")
        return None
    text = value.strip()
    if len(text) > MAX_TEXT_CHARS:
        errors.append(f"{field}: is {len(text)} characters; keep it under {MAX_TEXT_CHARS}")
        return None
    return text


def _check_confidence(args: dict, errors: list[str]) -> float | None:
    value = args["confidence"]
    number: float | None = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            number = None

    if number is None or math.isnan(number):
        errors.append(f"confidence: must be a number between 0.0 and 1.0, got {value!r}")
        return None
    if 1.0 < number <= 100.0:
        errors.append(
            f"confidence: {value!r} looks like a percentage; use a number between 0.0 and 1.0, "
            f"e.g. {number / 100:g}"
        )
        return None
    if not 0.0 <= number <= 1.0:
        errors.append(f"confidence: must be between 0.0 and 1.0, got {value!r}")
        return None
    return number


# ---------------------------------------------------------------------------
# Evidence: hardcoded_secret
# ---------------------------------------------------------------------------

# Provider key shapes that are secrets no matter what they're assigned to.
_KNOWN_KEY_RES = [
    re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bxox[abpors]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bsk_live_[A-Za-z0-9]{16,}"),
]

_SECRET_WORDS = r"(?:password|passwd|pwd|secret|token|api[_-]?key|private[_-]?key|access[_-]?key|credentials?)"

# A credential-looking name (optionally quoted, optionally type-annotated)
# assigned a quoted literal: `x = "..."`, `"x": "..."`, `'x' => '...'`,
# `x := "..."`, `x: str = "..."`, `` x = `...` ``.
_QUOTED_ASSIGN_RE = re.compile(
    r"[\"']?\b[\w.-]*" + _SECRET_WORDS + r"[\w.-]*[\"']?"
    r"\s*(?::\s*[\w\[\]., |]+?\s*(?==))?"
    r"(?::=|=>|=|:)\s*"
    r"([\"'`])(.*?)\1",
    re.IGNORECASE,
)

# Config files also allow unquoted values: `DB_PASSWORD=abc`, `password: abc`.
# Only applied to config-style files -- in source code an unquoted right-hand
# side is an expression (`token = request.headers`), not a literal.
_UNQUOTED_CONFIG_RE = re.compile(
    r"^\s*(?:export\s+)?[\w.-]*" + _SECRET_WORDS + r"[\w.-]*\s*[:=]\s*([^\s\"'#]+)\s*(?:#.*)?$",
    re.IGNORECASE,
)
_CONFIG_SUFFIXES = {".env", ".ini", ".cfg", ".conf", ".properties", ".yml", ".yaml", ".toml"}

_PLACEHOLDER_MARKERS = (
    "change", "your", "example", "placeholder", "dummy", "sample", "fake", "test",
    "replace", "todo", "redacted", "xxxx", "****", "<", ">", "${", "{{", "%(",
)
_ENV_REFERENCE_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


def _is_config_file(path: str) -> bool:
    p = PurePosixPath(path)
    return p.name.startswith(".env") or p.suffix.lower() in _CONFIG_SUFFIXES


def _looks_like_real_secret(value: str) -> bool:
    v = value.strip()
    if len(v) < 8 or any(c.isspace() for c in v):
        return False
    low = v.lower()
    if any(marker in low for marker in _PLACEHOLDER_MARKERS):
        return False
    if _ENV_REFERENCE_RE.fullmatch(v):
        return False
    if len(set(low)) <= 2:  # "aaaaaaaa", "--------"
        return False
    return True


def _line_has_secret(line: str, config_file: bool) -> bool:
    if any(r.search(line) for r in _KNOWN_KEY_RES):
        return True
    if any(_looks_like_real_secret(m.group(2)) for m in _QUOTED_ASSIGN_RE.finditer(line)):
        return True
    if config_file:
        m = _UNQUOTED_CONFIG_RE.match(line)
        if m and _looks_like_real_secret(m.group(1)):
            return True
    return False


# ---------------------------------------------------------------------------
# Evidence: sql_injection
# ---------------------------------------------------------------------------

# Matched against the whole window joined into one line, so a query split
# across lines ("SELECT id " / "FROM notes") is still recognized.
_SQL_KEYWORD_RES = [
    re.compile(r"\bselect\b.+?\bfrom\b", re.IGNORECASE),
    re.compile(r"\binsert\s+into\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+\S+\s+set\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+from\b", re.IGNORECASE),
]

# Ways a string gets built from variables. A parameterized query
# (`execute("... = ?", (x,))`) matches none of these.
_DYNAMIC_SQL_RES = [
    # Python f-string. Matched per quote type, because the SQL inside usually
    # has the *other* quote around values: f"... = '{owner}'". A triple-quoted
    # f-string counts on its own -- its braces are typically on later lines.
    re.compile(r"\b[rRbB]?[fF][rRbB]?(?:\"\"\"|'''|\"[^\"\n]*\{|'[^'\n]*\{)"),
    re.compile(r"\.format\(", re.IGNORECASE),  # str.format / String.format
    re.compile(r"\bSprintf\("),  # Go
    re.compile(r"[\"'][ \t]*%[ \t]+[\w(\[]|[\"'][ \t]+%[ \t]*[\w(\[]"),  # "..." % x
    re.compile(r"[\"'`]\s*\+\s*[\w(]|[\w)\]]\s*\+=?\s*[\"'`]"),  # concatenation
    re.compile(r"`[^`]*\$\{"),  # JS template literal
    re.compile(r"\$\"[^\"]*\{"),  # C# interpolated string
    re.compile(r"\"[^\"]*#\{"),  # Ruby interpolation
    re.compile(r"[\"']\s*\.\s*\$\w|\$\w+\s*\.\s*[\"']"),  # PHP concatenation
    re.compile(r"\"[^\"\n]*\$[A-Za-z_]\w*"),  # PHP "...$var"
]


# ---------------------------------------------------------------------------
# check_evidence
# ---------------------------------------------------------------------------


def check_evidence(
    issue_type: str, path: str, lines: list[str], line_start: int, line_end: int
) -> tuple[bool, str]:
    """Does the cited range (± EVIDENCE_SLACK lines) plausibly show this issue?

    `lines` is the whole file split into lines; line numbers are 1-based.
    Returns (ok, note) where note is a short, model-readable explanation.
    """
    total = len(lines)
    if line_start < 1 or line_start > total:
        return False, f"lines {line_start}-{line_end} are outside {path!r} (it has {total} lines)"

    lo = max(1, line_start - EVIDENCE_SLACK)
    hi = min(total, line_end + EVIDENCE_SLACK)
    window = [(n, lines[n - 1]) for n in range(lo, hi + 1)]
    cited = f"lines {line_start}-{line_end} (±{EVIDENCE_SLACK}) of {path!r}"

    if issue_type == "hardcoded_secret":
        config_file = _is_config_file(path)
        for n, line in window:
            if _line_has_secret(line, config_file):
                return True, f"credential-like literal found on line {n}"
        return False, (
            f"no hardcoded secret found in {cited}: expected a literal value assigned to a "
            "password/secret/token/key name, or a known key format"
        )

    if issue_type == "sql_injection":
        joined = " ".join(line for _, line in window)
        has_sql = any(r.search(joined) for r in _SQL_KEYWORD_RES)
        dynamic_lines = [n for n, line in window if any(r.search(line) for r in _DYNAMIC_SQL_RES)]
        if has_sql and dynamic_lines:
            return True, f"SQL query built with string formatting/concatenation on line {dynamic_lines[0]}"
        return False, (
            f"no SQL query built with string formatting or concatenation found in {cited}: "
            "expected SQL text combined with variables via f-string, +, %, .format or a template literal"
        )

    return False, f"unknown issue_type {issue_type!r}"


# ---------------------------------------------------------------------------
# Findings written as text (approach 1)
# ---------------------------------------------------------------------------

_PY_CALL_RE = re.compile(r"\breport_finding\s*\(")


def extract_text_findings(content: str | None) -> list[dict]:
    """Pull finding-shaped argument dicts out of free text.

    qwen2.5:7b often writes findings into its reply instead of calling
    report_finding. Shapes seen in real runs: a {"name": "report_finding",
    "arguments": {...}} wrapper, a bare finding object, and Python call
    syntax report_finding("medium", "security", ...). Extraction is
    deliberately permissive -- every dict returned must still go through
    RepoTools.report_finding (validation, read-gating, evidence check).
    Python-call arguments are read with ast.literal_eval only, so nothing
    in the text is ever executed.
    """
    if not content:
        return []
    found: list[dict] = []

    decoder = json.JSONDecoder()
    i = 0
    while (start := content.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        found.extend(findings_in_json(obj))
        i = end

    for match in _PY_CALL_RE.finditer(content):
        args = _parse_python_call(content, match.end() - 1)
        if args:
            found.append(args)
    return found


def _looks_like_finding(obj: dict) -> bool:
    return "issue_type" in obj or {"severity", "file"} <= obj.keys()


def findings_in_json(obj: object) -> list[dict]:
    if isinstance(obj, list):
        return [f for item in obj for f in findings_in_json(item)]
    if not isinstance(obj, dict):
        return []
    if obj.get("name") == "report_finding":
        arguments = obj.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return []
        return [arguments] if isinstance(arguments, dict) else []
    if isinstance(obj.get("findings"), list):
        return findings_in_json(obj["findings"])
    return [obj] if _looks_like_finding(obj) else []


def _parse_python_call(content: str, paren: int) -> dict | None:
    """report_finding(...) text -> argument dict (positional args in schema order)."""
    end = _matching_paren(content, paren)
    if end is None:
        return None
    try:
        call = ast.parse("f" + content[paren : end + 1], mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(call, ast.Call):
        return None
    try:
        positional = [ast.literal_eval(a) for a in call.args]
        keywords = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    args = dict(zip(FIELDS, positional))
    args.update(keywords)
    return args or None


def _matching_paren(text: str, open_index: int) -> int | None:
    """Index of the ')' closing text[open_index], skipping quoted strings."""
    depth, quote, escaped = 0, None, False
    for i in range(open_index, len(text)):
        ch = text[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None
