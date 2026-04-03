# Changelog

## Unreleased

- Prepared the repository for public release and VCS installs.
- Added public CLI entry points: `yoloe-camera-gui`, `yoloe-export`, and `python -m yoloe_tensorrt`.
- Reworked default asset handling so the example YOLOE checkpoint resolves from a user cache instead of a checked-in `models/` directory.
- Added production-facing documentation, a docs site config, CI, and standard repo policy files.
- Added an optional fallback prompt-encoder path when `mobileclip2_b.ts` is not bundled locally.
- Added GitHub Actions CI job separation for Ruff, docs, runner-safe tests, install smoke checks, and source-distribution builds.
- Added a tag-triggered release scaffold that verifies `pyproject.toml` and `CHANGELOG.md` before uploading release artifacts.

## 0.1.0

- Initial package release.
