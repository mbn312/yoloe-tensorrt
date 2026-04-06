from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch
import yoloe_tensorrt.export as export_mod
from yoloe_tensorrt import export_model


def test_export_model_creates_expected_onnx_artifacts(tmp_path: Path, model_checkpoint: Path) -> None:
    artifact_dir = export_model(model_checkpoint, artifact_dir=tmp_path / "artifacts", formats=("onnx",), dynamic=True)

    expected_files = {
        "main.onnx",
        "visual_prompt.onnx",
        "prompt_projector.ts",
        "metadata.json",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})

    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert metadata["task"] == "segment"
    assert metadata["embed_dim"] == 512
    assert metadata["visual_stride"] == 8

    main_model = onnx.load(artifact_dir / "main.onnx")
    main_inputs = [node.name for node in main_model.graph.input]
    assert main_inputs == ["images", "prompt_embeddings"]

    visual_model = onnx.load(artifact_dir / "visual_prompt.onnx")
    visual_inputs = [node.name for node in visual_model.graph.input]
    assert visual_inputs == ["images", "visual_prompts"]


def test_export_model_resumes_partial_main_only_bundle_without_reexport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_checkpoint: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "main.onnx").write_text("existing-onnx")
    (artifact_dir / "prompt_projector.ts").write_text("existing-projector")

    class _DummyHead:
        embed = 512
        nm = 32
        reprta = object()

    class _DummyModel:
        task = "segment"
        end2end = False
        stride = torch.tensor([8, 16, 32])
        yaml = {"text_model": "mobileclip2:b"}
        args = {"imgsz": 320}
        model = [_DummyHead()]

    monkeypatch.setattr(export_mod, "_load_yoloe_model", lambda _path: _DummyModel())
    monkeypatch.setattr(export_mod, "_configure_export_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(export_mod, "_save_text_encoder_asset", lambda *args, **kwargs: "mobileclip2_b.ts")
    monkeypatch.setattr(
        export_mod,
        "save_prompt_projector",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt projector should be reused")),
    )
    monkeypatch.setattr(
        export_mod,
        "_export_onnx",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ONNX export should be reused")),
    )

    engine_builds: list[tuple[str, str]] = []

    def _fake_build_engine(onnx_path: str | Path, engine_path: str | Path, **_kwargs) -> Path:
        engine_builds.append((Path(onnx_path).name, Path(engine_path).name))
        Path(engine_path).write_bytes(b"engine")
        return Path(engine_path)

    monkeypatch.setattr(export_mod, "build_engine_from_onnx", _fake_build_engine)

    output_dir = export_model(
        model_checkpoint,
        artifact_dir=artifact_dir,
        formats=("engine",),
        dynamic=False,
        build_visual_engine=False,
        overwrite=False,
    )

    assert output_dir == artifact_dir
    assert engine_builds == [("main.onnx", "main.engine")]

    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert metadata["main_engine_filename"] == "main.engine"
    assert metadata["visual_onnx_filename"] is None
    assert metadata["visual_engine_filename"] is None


def test_export_onnx_auto_falls_back_to_legacy_when_dynamo_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def _fake_dynamo(*args, **kwargs) -> None:
        calls.append("dynamo")
        raise RuntimeError("boom")

    def _fake_legacy(*args, **kwargs) -> None:
        calls.append("legacy")
        Path(args[2]).write_bytes(b"onnx")

    monkeypatch.setattr(export_mod, "_torch_onnx_supports", lambda option: option == "dynamo")
    monkeypatch.setattr(export_mod, "_export_onnx_dynamo", _fake_dynamo)
    monkeypatch.setattr(export_mod, "_export_onnx_legacy", _fake_legacy)

    output_path = tmp_path / "main.onnx"
    export_mod._export_onnx(
        torch.nn.Identity(),
        (torch.randn(1, 3, 32, 32),),
        output_path,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}},
        exporter="auto",
    )

    assert output_path.is_file()
    assert calls == ["dynamo", "legacy"]


def test_export_onnx_dynamo_mode_uses_dynamic_shapes_and_opset_18(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_export(*args, **kwargs) -> None:
        captured.update(kwargs)
        Path(args[2]).write_bytes(b"onnx")

    monkeypatch.setattr(export_mod.torch.onnx, "export", _fake_export)
    monkeypatch.setattr(export_mod, "_torch_onnx_supports", lambda option: option in {"dynamo", "dynamic_shapes"})
    monkeypatch.setattr(
        export_mod.torch,
        "export",
        type("_ExportNS", (), {"Dim": staticmethod(lambda name: f"Dim({name})")}),
    )

    output_path = tmp_path / "main.onnx"
    export_mod._export_onnx(
        torch.nn.Identity(),
        (torch.randn(1, 3, 32, 32), torch.randn(1, 80, 512)),
        output_path,
        input_names=["images", "prompt_embeddings"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "prompt_embeddings": {0: "batch", 1: "num_prompts"},
        },
        exporter="dynamo",
    )

    assert output_path.is_file()
    assert captured["dynamo"] is True
    assert captured["opset_version"] == 18
    assert "dynamic_axes" not in captured
    assert captured["dynamic_shapes"] == (
        {0: "Dim(batch)", 2: "Dim(height)", 3: "Dim(width)"},
        {0: "Dim(batch)", 1: "Dim(num_prompts)"},
    )


def test_export_onnx_dynamo_mode_raises_helpful_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_export(*args, **kwargs) -> None:
        raise RuntimeError("bad dynamo export")

    monkeypatch.setattr(export_mod.torch.onnx, "export", _fake_export)
    monkeypatch.setattr(export_mod, "_torch_onnx_supports", lambda option: option in {"dynamo", "dynamic_shapes"})
    monkeypatch.setattr(export_mod.torch, "export", type("_ExportNS", (), {"Dim": staticmethod(lambda name: name)}))

    with pytest.raises(RuntimeError, match="Retry with exporter='legacy' or exporter='auto'"):
        export_mod._export_onnx(
            torch.nn.Identity(),
            (torch.randn(1, 3, 32, 32),),
            tmp_path / "main.onnx",
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}},
            exporter="dynamo",
        )
