from __future__ import annotations

from pathlib import Path

import pytest
from yoloe_tensorrt.release_metadata import (
    changelog_has_version,
    load_project_version,
    normalize_expected_version,
    verify_release_metadata,
)


def test_normalize_expected_version_accepts_tags_and_refs() -> None:
    assert normalize_expected_version("v0.1.0") == "0.1.0"
    assert normalize_expected_version("refs/tags/v0.1.0") == "0.1.0"
    assert normalize_expected_version("0.1.0") == "0.1.0"


def test_load_project_version_reads_current_pyproject() -> None:
    assert load_project_version("pyproject.toml") == "0.1.0"


def test_changelog_has_version_matches_current_release() -> None:
    assert changelog_has_version("CHANGELOG.md", "0.1.0")


def test_verify_release_metadata_checks_expected_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    changelog = tmp_path / "CHANGELOG.md"
    pyproject.write_text('[project]\nversion = "1.2.3"\n')
    changelog.write_text("# Changelog\n\n## 1.2.3\n\n- released\n")

    assert (
        verify_release_metadata(
            pyproject_path=pyproject,
            changelog_path=changelog,
            expected_version="v1.2.3",
        )
        == "1.2.3"
    )


def test_verify_release_metadata_raises_for_mismatch(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    changelog = tmp_path / "CHANGELOG.md"
    pyproject.write_text('[project]\nversion = "1.2.3"\n')
    changelog.write_text("# Changelog\n\n## 1.2.3\n\n- released\n")

    with pytest.raises(ValueError, match="Release tag/version mismatch"):
        verify_release_metadata(
            pyproject_path=pyproject,
            changelog_path=changelog,
            expected_version="v9.9.9",
        )


def test_verify_release_metadata_raises_for_missing_changelog_section(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    changelog = tmp_path / "CHANGELOG.md"
    pyproject.write_text('[project]\nversion = "1.2.3"\n')
    changelog.write_text("# Changelog\n\n## Unreleased\n\n- pending\n")

    with pytest.raises(ValueError, match="CHANGELOG.md does not contain"):
        verify_release_metadata(
            pyproject_path=pyproject,
            changelog_path=changelog,
        )
