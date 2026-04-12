from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import yoloe_tensorrt.trt as trt_mod
from yoloe_tensorrt.artifacts import ShapeProfile


def _install_fake_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, object],
    *,
    with_parse_from_file: bool,
) -> None:
    class _Logger:
        WARNING = 1

        def __init__(self, _level: int) -> None:
            pass

    class _NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 0

    class _MemoryPoolType:
        WORKSPACE = "workspace"

    class _BuilderFlag:
        FP16 = "fp16"

    class _Profile:
        def __init__(self) -> None:
            self.shapes: list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

        def set_shape(
            self,
            name: str,
            minimum: tuple[int, ...],
            optimum: tuple[int, ...],
            maximum: tuple[int, ...],
        ) -> None:
            self.shapes.append((name, minimum, optimum, maximum))

    class _Config:
        def __init__(self) -> None:
            self.memory_pool_limits: list[tuple[object, int]] = []
            self.flags: list[object] = []
            self.profiles: list[_Profile] = []

        def set_memory_pool_limit(self, pool: object, size: int) -> None:
            self.memory_pool_limits.append((pool, size))

        def set_flag(self, flag: object) -> None:
            self.flags.append(flag)

        def add_optimization_profile(self, profile: _Profile) -> None:
            self.profiles.append(profile)

    class _Builder:
        platform_has_fast_fp16 = True

        def __init__(self, _logger: _Logger) -> None:
            state["builder"] = self

        def create_network(self, _flags: int) -> object:
            network = object()
            state["network"] = network
            return network

        def create_builder_config(self) -> _Config:
            config = _Config()
            state["config"] = config
            return config

        def create_optimization_profile(self) -> _Profile:
            profile = _Profile()
            state["profile"] = profile
            return profile

        def build_serialized_network(self, _network: object, _config: _Config) -> bytes:
            return b"engine"

    if with_parse_from_file:

        class _OnnxParser:
            def __init__(self, _network: object, _logger: _Logger) -> None:
                self.parse_from_file_calls: list[str] = []
                self.parse_calls: list[bytes] = []
                self.num_errors = 0
                state["parser"] = self

            def parse_from_file(self, path: str) -> bool:
                self.parse_from_file_calls.append(path)
                state["cwd_during_parse"] = os.getcwd()
                return True

            def parse(self, payload: bytes) -> bool:
                self.parse_calls.append(payload)
                return True

    else:

        class _OnnxParser:
            def __init__(self, _network: object, _logger: _Logger) -> None:
                self.parse_calls: list[bytes] = []
                self.num_errors = 0
                state["parser"] = self

            def parse(self, payload: bytes) -> bool:
                self.parse_calls.append(payload)
                state["cwd_during_parse"] = os.getcwd()
                return True

    fake_tensorrt = SimpleNamespace(
        Logger=_Logger,
        Builder=_Builder,
        OnnxParser=_OnnxParser,
        NetworkDefinitionCreationFlag=_NetworkDefinitionCreationFlag,
        MemoryPoolType=_MemoryPoolType,
        BuilderFlag=_BuilderFlag,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_tensorrt)


def test_build_engine_from_onnx_uses_parse_from_file_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}
    _install_fake_tensorrt(monkeypatch, state, with_parse_from_file=True)

    onnx_path = tmp_path / "main.onnx"
    onnx_path.write_bytes(b"onnx")
    (tmp_path / "main.onnx.data").write_bytes(b"weights")
    engine_path = tmp_path / "main.engine"

    output = trt_mod.build_engine_from_onnx(
        onnx_path,
        engine_path,
        profiles={"images": ShapeProfile((1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))},
    )

    parser = state["parser"]
    assert parser.parse_from_file_calls == [str(onnx_path.resolve())]
    assert parser.parse_calls == []
    assert output == engine_path
    assert engine_path.read_bytes() == b"engine"


def test_build_engine_from_onnx_falls_back_to_model_directory_for_parse_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}
    _install_fake_tensorrt(monkeypatch, state, with_parse_from_file=False)

    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    onnx_path = model_dir / "main.onnx"
    onnx_path.write_bytes(b"onnx")
    engine_path = model_dir / "main.engine"
    original_cwd = os.getcwd()

    trt_mod.build_engine_from_onnx(
        onnx_path,
        engine_path,
        profiles={"images": ShapeProfile((1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))},
    )

    parser = state["parser"]
    assert parser.parse_calls == [b"onnx"]
    assert state["cwd_during_parse"] == str(model_dir.resolve())
    assert os.getcwd() == original_cwd

