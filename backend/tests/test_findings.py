"""Tests for finding validation and evidence checks (app/agent/findings.py).

Both functions under test are pure (no GitHub, no LLM, no RepoTools), so
these tests just feed them dicts and lists of lines. The wiring into
RepoTools.report_finding -- read-gating, dedup, the per-scan cap -- is
tested separately in test_tools.py.
"""

from __future__ import annotations

import pytest

from app.agent.findings import (
    DEFAULT_CONFIDENCE,
    DEFAULT_SUGGESTION,
    MAX_LINE_SPAN,
    MAX_TEXT_CHARS,
    Finding,
    check_evidence,
    format_validation_errors,
    validate_finding_args,
)


def valid_args(**overrides) -> dict:
    args = dict(
        severity="high",
        category="security",
        issue_type="sql_injection",
        file="app/db.py",
        line_start=10,
        line_end=12,
        description="User input is formatted straight into a SQL query.",
        suggestion="Use a parameterized query instead.",
        confidence=0.8,
    )
    args.update(overrides)
    return args


# ---------------------------------------------------------------------------
# validate_finding_args: the happy path and tolerant normalization
# ---------------------------------------------------------------------------


def test_valid_args_produce_a_finding_and_no_errors():
    finding, errors = validate_finding_args(valid_args())

    assert errors == []
    assert finding == Finding(
        severity="high",
        category="security",
        issue_type="sql_injection",
        file="app/db.py",
        line_start=10,
        line_end=12,
        description="User input is formatted straight into a SQL query.",
        suggestion="Use a parameterized query instead.",
        confidence=0.8,
    )
    assert finding.verified is False
    assert finding.verification_note == ""


def test_enum_values_are_case_and_whitespace_insensitive():
    finding, errors = validate_finding_args(valid_args(severity="  High ", category="SECURITY"))

    assert errors == []
    assert finding.severity == "high"
    assert finding.category == "security"


@pytest.mark.parametrize("raw", ["SQL injection", "sql-injection", "Sql_Injection"])
def test_issue_type_accepts_spaces_hyphens_and_case(raw):
    finding, errors = validate_finding_args(valid_args(issue_type=raw))

    assert errors == []
    assert finding.issue_type == "sql_injection"


def test_numeric_strings_are_coerced_for_lines_and_confidence():
    finding, errors = validate_finding_args(valid_args(line_start="10", line_end=" 12 ", confidence="0.75"))

    assert errors == []
    assert (finding.line_start, finding.line_end, finding.confidence) == (10, 12, 0.75)


def test_integral_floats_are_accepted_as_line_numbers():
    finding, errors = validate_finding_args(valid_args(line_start=10.0, line_end=12.0))

    assert errors == []
    assert (finding.line_start, finding.line_end) == (10, 12)


def test_file_path_is_normalized():
    finding, errors = validate_finding_args(valid_args(file="./app\\db.py"))

    assert errors == []
    assert finding.file == "app/db.py"


def test_description_and_suggestion_are_stripped():
    finding, errors = validate_finding_args(valid_args(description="  desc  ", suggestion="\nfix it\n"))

    assert errors == []
    assert finding.description == "desc"
    assert finding.suggestion == "fix it"


def test_unknown_extra_keys_are_ignored():
    finding, errors = validate_finding_args(valid_args(cwe="CWE-89", title="SQLi"))

    assert errors == []
    assert finding is not None


def test_single_line_finding_is_allowed():
    finding, errors = validate_finding_args(valid_args(line_start=7, line_end=7))

    assert errors == []
    assert finding is not None


# ---------------------------------------------------------------------------
# validate_finding_args: rejections
# ---------------------------------------------------------------------------


def test_non_dict_arguments_are_rejected():
    finding, errors = validate_finding_args(["not", "a", "dict"])

    assert finding is None
    assert errors and "object" in errors[0]


@pytest.mark.parametrize(
    "field",
    ["severity", "category", "issue_type", "file", "line_start", "line_end", "description"],
)
def test_each_missing_required_field_is_reported(field):
    args = valid_args()
    del args[field]

    finding, errors = validate_finding_args(args)

    assert finding is None
    assert any(e.startswith(f"{field}:") and "missing" in e for e in errors)


def test_bad_severity_lists_the_allowed_values():
    finding, errors = validate_finding_args(valid_args(severity="critical"))

    assert finding is None
    assert len(errors) == 1
    assert errors[0].startswith("severity:")
    assert "'critical'" in errors[0]
    assert "high, medium, low, info" in errors[0]


def test_bad_category_lists_the_allowed_values():
    finding, errors = validate_finding_args(valid_args(category="style"))

    assert finding is None
    assert errors[0].startswith("category:")
    assert "security, bug, performance, maintainability" in errors[0]


@pytest.mark.parametrize("category", ["bug", "performance", "maintainability"])
def test_v1_only_accepts_the_security_category(category):
    finding, errors = validate_finding_args(valid_args(category=category))

    assert finding is None
    assert errors[0].startswith("category:")
    assert "security" in errors[0]


def test_bad_issue_type_explains_the_v1_scope():
    finding, errors = validate_finding_args(valid_args(issue_type="xss"))

    assert finding is None
    assert errors[0].startswith("issue_type:")
    assert "hardcoded_secret" in errors[0]
    assert "sql_injection" in errors[0]


@pytest.mark.parametrize("value", [123, None, ["high"]])
def test_non_string_enum_values_are_rejected(value):
    finding, errors = validate_finding_args(valid_args(severity=value))

    assert finding is None
    assert errors[0].startswith("severity:")


@pytest.mark.parametrize("path", ["../secrets.py", "/etc/passwd", "", "   "])
def test_unsafe_or_empty_file_paths_are_rejected(path):
    finding, errors = validate_finding_args(valid_args(file=path))

    assert finding is None
    assert errors[0].startswith("file:")


@pytest.mark.parametrize("value", ["ten", 1.5, True, None, [10]])
def test_non_integer_line_numbers_are_rejected(value):
    finding, errors = validate_finding_args(valid_args(line_start=value))

    assert finding is None
    assert any(e.startswith("line_start:") for e in errors)


def test_line_numbers_must_be_at_least_one():
    finding, errors = validate_finding_args(valid_args(line_start=0, line_end=0))

    assert finding is None
    assert any(e.startswith("line_start:") for e in errors)
    assert any(e.startswith("line_end:") for e in errors)


def test_line_end_before_line_start_is_rejected_with_both_values():
    finding, errors = validate_finding_args(valid_args(line_start=40, line_end=12))

    assert finding is None
    assert errors == ["line_end: must be >= line_start (got line_start=40, line_end=12)"]


def test_line_span_is_capped():
    finding, errors = validate_finding_args(valid_args(line_start=1, line_end=MAX_LINE_SPAN + 1))

    assert finding is None
    assert errors[0].startswith("line_end:")
    assert str(MAX_LINE_SPAN) in errors[0]


def test_line_span_at_the_cap_is_allowed():
    finding, errors = validate_finding_args(valid_args(line_start=1, line_end=MAX_LINE_SPAN))

    assert errors == []
    assert finding is not None


@pytest.mark.parametrize("value", [-0.1, 1.01, "high", True, ""])
def test_out_of_range_or_non_numeric_confidence_is_rejected(value):
    finding, errors = validate_finding_args(valid_args(confidence=value))

    assert finding is None
    assert errors[0].startswith("confidence:")


def test_missing_confidence_defaults_to_one_half():
    # The one field qwen2.5:7b left out in the first real runs; a missing
    # confidence shouldn't cost a whole step to fix.
    args = valid_args()
    del args["confidence"]

    finding, errors = validate_finding_args(args)

    assert errors == []
    assert finding.confidence == DEFAULT_CONFIDENCE == 0.5


@pytest.mark.parametrize("value", ["<missing>", None])
def test_missing_suggestion_gets_a_non_empty_default(value):
    # qwen2.5:7b left out suggestion twice in round-3 tool calls.
    args = valid_args()
    if value == "<missing>":
        del args["suggestion"]
    else:
        args["suggestion"] = value

    finding, errors = validate_finding_args(args)

    assert errors == []
    assert finding.suggestion == DEFAULT_SUGGESTION
    assert DEFAULT_SUGGESTION.strip()


def test_a_provided_but_blank_suggestion_is_still_rejected():
    finding, errors = validate_finding_args(valid_args(suggestion="   "))

    assert finding is None
    assert errors[0].startswith("suggestion:")


def test_null_confidence_defaults_to_one_half():
    finding, errors = validate_finding_args(valid_args(confidence=None))

    assert errors == []
    assert finding.confidence == 0.5


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
def test_confidence_bounds_are_inclusive(value):
    finding, errors = validate_finding_args(valid_args(confidence=value))

    assert errors == []
    assert finding.confidence == float(value)


def test_percentage_style_confidence_gets_a_hint():
    finding, errors = validate_finding_args(valid_args(confidence=80))

    assert finding is None
    assert "0.8" in errors[0]


@pytest.mark.parametrize(
    "field,value",
    [("description", v) for v in ("", "   ", None, 42)]
    # suggestion=None means "not given" and gets the default (see below).
    + [("suggestion", v) for v in ("", "   ", 42)],
)
def test_description_and_suggestion_must_be_non_empty_strings(field, value):
    finding, errors = validate_finding_args(valid_args(**{field: value}))

    assert finding is None
    assert errors[0].startswith(f"{field}:")


@pytest.mark.parametrize("field", ["description", "suggestion"])
def test_description_and_suggestion_are_length_capped(field):
    finding, errors = validate_finding_args(valid_args(**{field: "x" * (MAX_TEXT_CHARS + 1)}))

    assert finding is None
    assert errors[0].startswith(f"{field}:")
    assert str(MAX_TEXT_CHARS) in errors[0]


def test_every_problem_is_reported_at_once():
    finding, errors = validate_finding_args(
        valid_args(severity="critical", issue_type="xss", confidence=2, description="")
    )

    assert finding is None
    fields = [e.split(":", 1)[0] for e in errors]
    assert fields == ["severity", "issue_type", "description", "confidence"]


# ---------------------------------------------------------------------------
# format_validation_errors
# ---------------------------------------------------------------------------


def test_format_validation_errors_is_one_readable_error_string():
    message = format_validation_errors(
        [
            "severity: 'critical' is not allowed; use one of: high, medium, low, info",
            "line_end: must be >= line_start (got line_start=40, line_end=12)",
        ]
    )

    assert message == (
        "Error: report_finding rejected, nothing was recorded. Fix these and call it again:\n"
        "- severity: 'critical' is not allowed; use one of: high, medium, low, info\n"
        "- line_end: must be >= line_start (got line_start=40, line_end=12)"
    )


# ---------------------------------------------------------------------------
# Finding.to_dict
# ---------------------------------------------------------------------------


def test_finding_to_dict_includes_verification_fields():
    finding, _ = validate_finding_args(valid_args())
    finding.verified = True
    finding.verification_note = "ok"

    data = finding.to_dict()

    assert data["issue_type"] == "sql_injection"
    assert data["line_start"] == 10
    assert data["verified"] is True
    assert data["verification_note"] == "ok"


# ---------------------------------------------------------------------------
# check_evidence: hardcoded_secret
# ---------------------------------------------------------------------------


def _check_one(issue_type: str, line: str, path: str = "app/config.py") -> bool:
    ok, _ = check_evidence(issue_type, path, [line], 1, 1)
    return ok


@pytest.mark.parametrize(
    "line",
    [
        'DB_PASSWORD = "Tr0ub4dor&3-prod"',
        "api_key = '9f2c4e8a1b7d3f6e0a5c8b2d4f7e1a3c'",
        'SECRET_TOKEN: str = "q8Zr2LmP0vXy7Tn4"',
        '    "apiKey": "9f2c4e8a1b7d3f6e0a5c8b2d",',
        "const paymentToken = 'pt_9a8b7c6d5e4f3a2b1c0d';",
        "  'db_password' => 'Tr0ub4dor&3-prod',",
        'password := "Tr0ub4dor&3-prod"',
        "const ACCESS_KEY = `k3y-9a8b7c6d5e4f`;",
    ],
)
def test_secret_assigned_as_a_literal_is_verified(line):
    assert _check_one("hardcoded_secret", line)


@pytest.mark.parametrize(
    "line",
    [
        # Built at runtime so the repo itself never contains a literal that
        # matches a real provider's secret-scanning pattern.
        "key = " + repr("AKIA" + "Q7RT2X9MZK4PLW3N"),
        "t = " + repr("gh" + "p_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"),
        "slack = " + repr("xo" + "xb-" + "1234567890-abcdefghij"),
        "-----BEGIN " + "RSA PRIVATE KEY-----",
        "-----BEGIN " + "PRIVATE KEY-----",
    ],
)
def test_known_key_shapes_are_verified_regardless_of_variable_name(line):
    assert _check_one("hardcoded_secret", line)


@pytest.mark.parametrize(
    "line",
    [
        'api_key = os.environ["API_KEY"]',
        'DB_PASSWORD = os.getenv("DB_PASSWORD")',
        'password = os.environ.get("DB_PASSWORD", "")',
        "const apiKey = process.env.API_KEY;",
        'token = config.get("token")',
        'SECRET_KEY = "change-me"',
        'API_KEY = "your-api-key-here"',
        'password = "<password>"',
        'password = "${DB_PASSWORD}"',
        'token = "{{ secrets.TOKEN }}"',
        'password = "xxxxxxxxxx"',
        'password = "********"',
        'password_label = "Enter your password"',
        'password = "short"',
        'TEST_PASSWORD = "test-password-123"',
        'api_key = ""',
        "def check_password(password):",
        'user = "alice_the_admin"',
    ],
)
def test_env_lookups_placeholders_and_non_secrets_are_not_verified(line):
    assert not _check_one("hardcoded_secret", line)


@pytest.mark.parametrize("path", [".env", ".env.production", "config/settings.ini", "deploy/app.yaml"])
def test_unquoted_secrets_count_in_config_files(path):
    ok_env, _ = check_evidence("hardcoded_secret", path, ["DB_PASSWORD=Tr0ub4dor3prod"], 1, 1)
    ok_yaml, _ = check_evidence("hardcoded_secret", path, ["  db_password: Tr0ub4dor3prod"], 1, 1)

    assert ok_env and ok_yaml


def test_unquoted_assignments_do_not_count_in_source_files():
    ok, _ = check_evidence("hardcoded_secret", "app/auth.py", ["token = request.headers"], 1, 1)

    assert not ok


def test_unquoted_env_references_do_not_count_in_config_files():
    ok, _ = check_evidence("hardcoded_secret", "docker.yaml", ["  db_password: ${DB_PASSWORD}"], 1, 1)

    assert not ok


# ---------------------------------------------------------------------------
# check_evidence: sql_injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines",
    [
        ['cur.execute(f"SELECT * FROM users WHERE id = {user_id}")'],
        # The common shape: SQL string quotes inside the f-string.
        ['query = f"SELECT * FROM notes WHERE owner = \'{owner}\'"'],
        ["query = f'SELECT * FROM notes WHERE owner = \"{owner}\"'"],
        ['query = f"""', "    SELECT * FROM notes", "    WHERE owner = '{owner}'", '"""'],
        ['query = "SELECT * FROM users WHERE name = \'" + name + "\'"'],
        ['cur.execute("DELETE FROM notes WHERE id = %s" % note_id)'],
        ['sql = "UPDATE users SET email = \'{}\' WHERE id = {}".format(email, uid)'],
        ["db.query(`SELECT * FROM orders WHERE customer = '${req.query.customer}'`);"],
        ['q := fmt.Sprintf("SELECT * FROM t WHERE id = %s", id)'],
        ['$sql = "SELECT * FROM users WHERE id = " . $id;'],
        # Query split across lines: keyword on one, dynamic part on the next.
        ['query = ("SELECT id, body FROM notes "', '         "WHERE owner = \'" + owner + "\'")'],
        ['query = "SELECT * FROM notes WHERE 1=1"', 'query += " AND tag = \'" + tag + "\'"'],
        ['cur.execute("INSERT INTO logs (msg) VALUES (\'%s\')" % msg)'],
    ],
)
def test_sql_built_with_string_formatting_is_verified(lines):
    ok, _ = check_evidence("sql_injection", "app/db.py", lines, 1, len(lines))

    assert ok


@pytest.mark.parametrize(
    "lines",
    [
        ['cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))'],
        ['cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))'],
        ['db.query("SELECT * FROM orders WHERE customer = ?", [customer]);'],
        ['session.execute(text("SELECT * FROM t WHERE id = :id"), {"id": tid})'],
        ['greeting = f"Hello {name}"'],
        ["from flask import Flask, request"],
        ['settings.update({"theme": theme})'],
        ['log.info("user %s selected item from menu" % user)'],
    ],
)
def test_parameterized_queries_and_non_sql_strings_are_not_verified(lines):
    ok, _ = check_evidence("sql_injection", "app/db.py", lines, 1, len(lines))

    assert not ok


# ---------------------------------------------------------------------------
# check_evidence: window, slack, and robustness
# ---------------------------------------------------------------------------


def _file_with(line_no: int, text: str, total: int = 20) -> list[str]:
    lines = ["x = 1"] * total
    lines[line_no - 1] = text
    return lines


def test_evidence_within_two_lines_of_the_cited_range_is_accepted():
    lines = _file_with(10, 'cur.execute(f"SELECT * FROM t WHERE id = {i}")')

    ok_before, _ = check_evidence("sql_injection", "a.py", lines, 12, 13)
    ok_after, _ = check_evidence("sql_injection", "a.py", lines, 7, 8)

    assert ok_before and ok_after


def test_evidence_further_than_the_slack_is_not_accepted():
    lines = _file_with(10, 'cur.execute(f"SELECT * FROM t WHERE id = {i}")')

    ok, _ = check_evidence("sql_injection", "a.py", lines, 13, 14)

    assert not ok


def test_verified_note_names_the_matching_line():
    lines = _file_with(10, 'DB_PASSWORD = "Tr0ub4dor&3-prod"')

    ok, note = check_evidence("hardcoded_secret", "config.py", lines, 10, 10)

    assert ok
    assert "line 10" in note


def test_unverified_notes_explain_what_was_missing():
    lines = ["x = 1"] * 5

    _, secret_note = check_evidence("hardcoded_secret", "a.py", lines, 1, 2)
    _, sql_note = check_evidence("sql_injection", "a.py", lines, 1, 2)

    assert "lines 1-2" in secret_note and "secret" in secret_note
    assert "lines 1-2" in sql_note and "SQL" in sql_note


def test_range_outside_the_file_is_not_verified():
    ok, note = check_evidence("sql_injection", "a.py", ["x = 1"] * 5, 9, 10)

    assert not ok
    assert "5 lines" in note


def test_unknown_issue_type_is_not_verified():
    ok, _ = check_evidence("xss", "a.py", ["x = 1"], 1, 1)

    assert not ok


# ---------------------------------------------------------------------------
# extract_text_findings: findings the model wrote as text (approach 1)
# ---------------------------------------------------------------------------

from app.agent.findings import extract_text_findings  # noqa: E402


def test_extracts_a_name_arguments_wrapper_from_a_json_block():
    # Verbatim shape from the clean-control re-run.
    text = (
        "Let's report these.\n```json\n"
        '{"name": "report_finding", "arguments": {"category": "security", "confidence": 0.9, '
        '"description": "d", "file": "inventory/db.py", "line_start": 14, "line_end": 15, '
        '"severity": "medium", "suggestion": "s"}}\n```'
    )

    found = extract_text_findings(text)

    assert len(found) == 1
    assert found[0]["file"] == "inventory/db.py"
    assert found[0]["line_start"] == 14


def test_extracts_a_bare_finding_object():
    text = 'Found: {"severity": "high", "category": "security", "issue_type": "sql_injection", "file": "db.py", "line_start": 24, "line_end": 25, "description": "d", "suggestion": "s"}'

    found = extract_text_findings(text)

    assert found == [
        {
            "severity": "high", "category": "security", "issue_type": "sql_injection", "file": "db.py",
            "line_start": 24, "line_end": 25, "description": "d", "suggestion": "s",
        }
    ]


def test_extracts_positional_python_call_syntax():
    # Verbatim shape from the js-shop re-run.
    text = (
        '```json\nreport_finding("medium", "security", "sql_injection", "src/routes/orders.js", 8, 16, '
        '"The SQL query is built from user input.", "Use parameterized queries.")\n```'
    )

    found = extract_text_findings(text)

    assert found == [
        {
            "severity": "medium", "category": "security", "issue_type": "sql_injection",
            "file": "src/routes/orders.js", "line_start": 8, "line_end": 16,
            "description": "The SQL query is built from user input.", "suggestion": "Use parameterized queries.",
        }
    ]


def test_extracts_keyword_python_call_syntax():
    text = 'report_finding(severity="high", issue_type="hardcoded_secret", file="config.py", line_start=17, line_end=17)'

    found = extract_text_findings(text)

    assert found == [
        {"severity": "high", "issue_type": "hardcoded_secret", "file": "config.py", "line_start": 17, "line_end": 17}
    ]


def test_extracts_several_findings_in_one_reply():
    one = '{"name": "report_finding", "arguments": {"issue_type": "sql_injection", "file": "a.py", "line_start": 1, "line_end": 1}}'
    two = '{"name": "report_finding", "arguments": {"issue_type": "sql_injection", "file": "a.py", "line_start": 9, "line_end": 9}}'

    found = extract_text_findings(f"First:\n{one}\nSecond:\n{two}")

    assert [f["line_start"] for f in found] == [1, 9]


def test_extracts_findings_inside_a_findings_list():
    text = '{"findings": [{"issue_type": "sql_injection", "file": "a.py", "line_start": 3, "line_end": 3}]}'

    assert [f["line_start"] for f in extract_text_findings(text)] == [3]


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "No issues found. The code uses parameterized queries.",
        'Config looks like {"debug": true}.',
        '{"name": "read_file", "arguments": {"path": "db.py"}}',
        "Call report_finding when you find something.",  # mentioned, not called
        "report_finding(this is not valid python",
        "report_finding(open('/etc/passwd').read())",  # never evaluated: literals only
    ],
)
def test_ignores_text_without_a_finding(text):
    assert extract_text_findings(text) == []


def test_the_extracted_dict_still_has_to_pass_validation():
    # Extraction is deliberately permissive; validation stays the gatekeeper.
    found = extract_text_findings('report_finding("critical", "style", "xss", "a.py", 0, 0, "", "")')

    finding, errors = validate_finding_args(found[0])

    assert finding is None
    assert errors
