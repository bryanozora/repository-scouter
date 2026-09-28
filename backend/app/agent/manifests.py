"""Pure parsers for known dependency-manifest formats.

No I/O here -- each function takes an already-fetched file's content and
returns a compact plain-text summary. Kept separate from tools.py so each
parser is directly unit-testable without any RepoTools/github_client setup.

Deliberately shallow: JSON and TOML manifests get a real (if partial)
parse; everything else recognized (pom.xml, build.gradle, ...) falls back
to raw capped content in tools.py rather than a bespoke XML/Gradle-DSL
parser -- out of scope for v1.
"""

from __future__ import annotations

import json
import tomllib


def parse_package_json(content: str) -> str:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return "(could not parse package.json as JSON)"

    lines = []
    for section in ("dependencies", "devDependencies"):
        deps = data.get(section)
        if not deps:
            continue
        lines.append(f"{section}:")
        for name, version in deps.items():
            lines.append(f"  {name}@{version}")
    return "\n".join(lines) if lines else "(no dependencies declared)"


def parse_composer_json(content: str) -> str:
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return "(could not parse composer.json as JSON)"

    lines = []
    for section in ("require", "require-dev"):
        deps = data.get(section)
        if not deps:
            continue
        lines.append(f"{section}:")
        for name, version in deps.items():
            lines.append(f"  {name} {version}")
    return "\n".join(lines) if lines else "(no dependencies declared)"


def parse_pyproject_toml(content: str) -> str:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return "(could not parse pyproject.toml as TOML)"

    lines = []
    project_deps = data.get("project", {}).get("dependencies")
    if project_deps:
        lines.append("dependencies:")
        for dep in project_deps:
            lines.append(f"  {dep}")

    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies")
    if poetry_deps:
        lines.append("tool.poetry.dependencies:")
        for name, version in poetry_deps.items():
            lines.append(f"  {name} {version}")

    return "\n".join(lines) if lines else "(no dependencies declared)"


def parse_cargo_toml(content: str) -> str:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return "(could not parse Cargo.toml as TOML)"

    deps = data.get("dependencies")
    if not deps:
        return "(no dependencies declared)"

    lines = ["dependencies:"]
    for name, spec in deps.items():
        if isinstance(spec, str):
            version = spec
        elif isinstance(spec, dict):
            version = spec.get("version", "*")
        else:
            version = str(spec)
        lines.append(f"  {name} {version}")
    return "\n".join(lines)


def parse_pipfile(content: str) -> str:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return "(could not parse Pipfile as TOML)"

    lines = []
    for section in ("packages", "dev-packages"):
        deps = data.get(section)
        if not deps:
            continue
        lines.append(f"{section}:")
        for name, version in deps.items():
            lines.append(f"  {name} {version}")
    return "\n".join(lines) if lines else "(no dependencies declared)"


def parse_plain_text_lines(content: str) -> str:
    """requirements.txt, go.mod, Gemfile -- the non-comment, non-blank lines as-is."""
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    return "\n".join(lines) if lines else "(no entries found)"


# Manifest filename -> parser.
PARSERS = {
    "package.json": parse_package_json,
    "composer.json": parse_composer_json,
    "pyproject.toml": parse_pyproject_toml,
    "Cargo.toml": parse_cargo_toml,
    "Pipfile": parse_pipfile,
    "requirements.txt": parse_plain_text_lines,
    "go.mod": parse_plain_text_lines,
    "Gemfile": parse_plain_text_lines,
}

# Manifests recognized as dependency-relevant but not parsed here -- tools.py
# falls back to filename + raw capped content for these.
RAW_ONLY_MANIFESTS = {"pom.xml", "build.gradle", "build.gradle.kts"}

# What get_dependencies() looks for in the repo tree. Lock files are
# deliberately excluded -- they're resolved/huge, not what "what does this
# project depend on" needs.
KNOWN_MANIFESTS = set(PARSERS) | RAW_ONLY_MANIFESTS
