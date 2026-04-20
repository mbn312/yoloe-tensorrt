from __future__ import annotations

import pytest
import yoloe_tensorrt.native_backend as native_backend


def test_build_native_visual_runtime_propagates_constructor_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_runtime_error(*args, **kwargs):
        raise RuntimeError("synthetic native visual constructor failure")

    monkeypatch.setattr(native_backend, "NATIVE_AVAILABLE", True)
    monkeypatch.setattr(native_backend, "NativeVisualPromptRuntime", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="synthetic native visual constructor failure"):
        native_backend.build_native_visual_runtime(
            "visual.engine",
            image_input_name="images",
            visual_input_name="visual_prompts",
            visual_stride=32,
        )
