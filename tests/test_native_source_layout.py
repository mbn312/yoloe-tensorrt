from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_DIR = REPO_ROOT / "src" / "native"


def test_native_extension_is_split_into_multiple_translation_units() -> None:
    assert not (NATIVE_DIR / "main.cpp").exists()

    expected_sources = {
        "module.cpp",
        "main_runtime.cpp",
        "visual_runtime.cpp",
        "jetson_camera.cpp",
    }
    for filename in expected_sources:
        assert (NATIVE_DIR / filename).exists(), filename


def test_native_shared_helpers_are_split_by_responsibility() -> None:
    expected_headers = {
        "runtime_common.h",
        "postprocess_common.h",
        "visual_prompt_batch.h",
        "tensor_views.h",
    }
    for filename in expected_headers:
        assert (NATIVE_DIR / filename).exists(), filename

    common_lines = (NATIVE_DIR / "native_common.h").read_text(encoding="utf-8").splitlines()
    assert len(common_lines) < 80


def test_cmake_native_sources_reference_split_runtime_files() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "src/native/module.cpp" in cmake
    assert "src/native/main_runtime.cpp" in cmake
    assert "src/native/visual_runtime.cpp" in cmake
    assert "src/native/jetson_camera.cpp" in cmake
