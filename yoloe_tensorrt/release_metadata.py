from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT_VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def normalize_expected_version(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized.startswith("refs/tags/"):
        normalized = normalized.split("/", 2)[-1]
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized or None


def load_project_version(pyproject_path: str | Path) -> str:
    content = Path(pyproject_path).read_text()
    match = PROJECT_VERSION_PATTERN.search(content)
    if match is None:
        raise ValueError(f"Unable to find a project version in '{pyproject_path}'")
    return match.group(1)


def changelog_has_version(changelog_path: str | Path, version: str) -> bool:
    content = Path(changelog_path).read_text()
    return bool(re.search(rf"^##\s+{re.escape(version)}\s*$", content, re.MULTILINE))


def verify_release_metadata(
    *,
    pyproject_path: str | Path = "pyproject.toml",
    changelog_path: str | Path = "CHANGELOG.md",
    expected_version: str | None = None,
) -> str:
    version = load_project_version(pyproject_path)
    normalized_expected = normalize_expected_version(expected_version)
    if normalized_expected is not None and normalized_expected != version:
        raise ValueError(
            f"Release tag/version mismatch: expected '{normalized_expected}' but pyproject.toml declares '{version}'"
        )
    if not changelog_has_version(changelog_path, version):
        raise ValueError(f"CHANGELOG.md does not contain a released section for version '{version}'")
    return version


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify release metadata for tag-based release automation.")
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml. Defaults to pyproject.toml in the current working directory.",
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md. Defaults to CHANGELOG.md in the current working directory.",
    )
    parser.add_argument(
        "--expected-version",
        default=None,
        help="Expected version or tag value, for example 0.1.0 or v0.1.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    version = verify_release_metadata(
        pyproject_path=args.pyproject,
        changelog_path=args.changelog,
        expected_version=args.expected_version,
    )
    print(version)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
