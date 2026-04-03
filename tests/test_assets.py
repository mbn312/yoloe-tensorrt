from __future__ import annotations

from pathlib import Path

import pytest
import yoloe_tensorrt.assets as assets_mod


def test_resolve_model_checkpoint_reuses_existing_local_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"pt")

    resolved = assets_mod.resolve_model_checkpoint(checkpoint)

    assert resolved == checkpoint.resolve()


def test_resolve_model_checkpoint_downloads_into_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLOE_TRT_CACHE_DIR", str(tmp_path / "cache"))

    def _fake_download(path: str | Path, **_kwargs) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")
        return str(target)

    monkeypatch.setattr("ultralytics.utils.downloads.attempt_download_asset", _fake_download)

    resolved = assets_mod.resolve_model_checkpoint("yoloe-26s-seg.pt")

    assert resolved == (tmp_path / "cache" / "assets" / "yoloe-26s-seg.pt").resolve()


def test_resolve_model_checkpoint_rejects_missing_relative_path() -> None:
    with pytest.raises(FileNotFoundError):
        assets_mod.resolve_model_checkpoint("missing/model.pt", allow_download=False)


def test_download_text_asset_downloads_into_text_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YOLOE_TRT_CACHE_DIR", str(tmp_path / "cache"))

    def _fake_download(path: str | Path, **_kwargs) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"text-weights")
        return str(target)

    monkeypatch.setattr("ultralytics.utils.downloads.attempt_download_asset", _fake_download)

    resolved = assets_mod.download_text_asset("mobileclip2_b.ts")

    assert resolved == (tmp_path / "cache" / "assets" / "text" / "mobileclip2_b.ts").resolve()
