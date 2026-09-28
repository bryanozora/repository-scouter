"""Tests for the pure dependency-manifest parsers (app/agent/manifests.py).

No I/O, no RepoTools/github_client involved -- each function is just
content-in, compact-summary-out, so these are plain unit tests.
"""

from __future__ import annotations

from app.agent import manifests


# ---------------------------------------------------------------------------
# package.json / composer.json (JSON)
# ---------------------------------------------------------------------------


def test_parse_package_json_extracts_dependencies_and_dev_dependencies():
    content = """{
        "name": "demo",
        "dependencies": {"react": "^18.0.0"},
        "devDependencies": {"typescript": "^5.0.0"}
    }"""

    result = manifests.parse_package_json(content)

    assert "react@^18.0.0" in result
    assert "typescript@^5.0.0" in result


def test_parse_package_json_handles_missing_sections():
    result = manifests.parse_package_json('{"name": "demo"}')

    assert "no dependencies" in result.lower()


def test_parse_package_json_handles_invalid_json():
    result = manifests.parse_package_json("not json at all")

    assert "could not parse" in result.lower()


def test_parse_composer_json_extracts_require_and_require_dev():
    content = """{
        "require": {"php": ">=8.0", "monolog/monolog": "^2.0"},
        "require-dev": {"phpunit/phpunit": "^9.0"}
    }"""

    result = manifests.parse_composer_json(content)

    assert "monolog/monolog" in result
    assert "phpunit/phpunit" in result


# ---------------------------------------------------------------------------
# pyproject.toml / Cargo.toml / Pipfile (TOML)
# ---------------------------------------------------------------------------


def test_parse_pyproject_toml_extracts_pep621_dependencies():
    content = """
[project]
name = "demo"
dependencies = ["httpx>=0.27", "python-dotenv>=1.0"]
"""

    result = manifests.parse_pyproject_toml(content)

    assert "httpx>=0.27" in result
    assert "python-dotenv>=1.0" in result


def test_parse_pyproject_toml_extracts_poetry_dependencies():
    content = """
[tool.poetry.dependencies]
python = "^3.11"
requests = "^2.31"
"""

    result = manifests.parse_pyproject_toml(content)

    assert "requests" in result
    assert "2.31" in result


def test_parse_pyproject_toml_handles_invalid_toml():
    result = manifests.parse_pyproject_toml("this is not [valid toml")

    assert "could not parse" in result.lower()


def test_parse_cargo_toml_extracts_dependencies():
    content = """
[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
"""

    result = manifests.parse_cargo_toml(content)

    assert "serde" in result
    assert "1.0" in result
    assert "tokio" in result


def test_parse_pipfile_extracts_packages_and_dev_packages():
    content = """
[packages]
requests = "*"

[dev-packages]
pytest = "*"
"""

    result = manifests.parse_pipfile(content)

    assert "requests" in result
    assert "pytest" in result


# ---------------------------------------------------------------------------
# plain-text line lists (requirements.txt, go.mod, Gemfile)
# ---------------------------------------------------------------------------


def test_parse_plain_text_lines_filters_comments_and_blanks():
    content = "httpx>=0.27\n# a comment\n\npytest>=8.0\n   \n"

    result = manifests.parse_plain_text_lines(content)

    assert result == "httpx>=0.27\npytest>=8.0"


def test_parse_plain_text_lines_handles_empty_content():
    result = manifests.parse_plain_text_lines("\n\n   \n")

    assert "no entries" in result.lower()


# ---------------------------------------------------------------------------
# manifest registry
# ---------------------------------------------------------------------------


def test_known_manifests_includes_parsed_and_raw_only_names():
    assert "package.json" in manifests.KNOWN_MANIFESTS
    assert "requirements.txt" in manifests.KNOWN_MANIFESTS
    assert "pom.xml" in manifests.KNOWN_MANIFESTS  # raw-only, not in PARSERS
    assert "pom.xml" not in manifests.PARSERS


def test_known_manifests_excludes_lock_files():
    assert "package-lock.json" not in manifests.KNOWN_MANIFESTS
    assert "poetry.lock" not in manifests.KNOWN_MANIFESTS
