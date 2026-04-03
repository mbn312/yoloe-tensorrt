from __future__ import annotations

from pathlib import Path

import torch
import yoloe_tensorrt.prompts as prompts_mod


def test_text_prompt_encoder_falls_back_when_mobileclip2_torchscript_is_missing(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeMobileCLIPTS:
        def __init__(self, *, device, weight):
            calls.append(("ts", str(weight)))

    class _FakeMobileCLIP:
        def __init__(self, *, size, device):
            calls.append(("fallback", str(size)))

    monkeypatch.setattr(prompts_mod, "MobileCLIPTS", _FakeMobileCLIPTS)
    monkeypatch.setattr(prompts_mod, "MobileCLIP", _FakeMobileCLIP)
    monkeypatch.setattr(prompts_mod, "resolve_text_asset_path", lambda *args, **kwargs: None)

    prompts_mod.TextPromptEncoder("mobileclip2:b", device=torch.device("cpu"), weight_path=None)

    assert calls == [("fallback", "b")]


def test_text_prompt_encoder_uses_explicit_torchscript_weight(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeMobileCLIPTS:
        def __init__(self, *, device, weight):
            calls.append(("ts", str(weight)))

    monkeypatch.setattr(prompts_mod, "MobileCLIPTS", _FakeMobileCLIPTS)

    prompts_mod.TextPromptEncoder("mobileclip2:b", device=torch.device("cpu"), weight_path="/tmp/mobileclip2_b.ts")

    assert calls == [("ts", "/tmp/mobileclip2_b.ts")]


def test_text_prompt_encoder_auto_resolves_torchscript_weight(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeMobileCLIPTS:
        def __init__(self, *, device, weight):
            calls.append(("ts", str(weight)))

    class _FakeMobileCLIP:
        def __init__(self, *, size, device):
            calls.append(("fallback", str(size)))

    monkeypatch.setattr(prompts_mod, "MobileCLIPTS", _FakeMobileCLIPTS)
    monkeypatch.setattr(prompts_mod, "MobileCLIP", _FakeMobileCLIP)
    monkeypatch.setattr(prompts_mod, "text_asset_search_roots", lambda *args, **kwargs: ["/tmp"])
    monkeypatch.setattr(prompts_mod, "resolve_text_asset_path", lambda *args, **kwargs: Path("/tmp/mobileclip2_b.ts"))

    prompts_mod.TextPromptEncoder("mobileclip2:b", device=torch.device("cpu"), weight_path=None)

    assert calls == [("ts", "/tmp/mobileclip2_b.ts")]


def test_text_prompt_encoder_fallback_uses_text_cache_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class _FakeMobileCLIP:
        def __init__(self, *, size, device):
            calls.append((str(size), str(Path.cwd())))

    monkeypatch.setenv("YOLOE_TRT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(prompts_mod, "resolve_text_asset_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(prompts_mod, "MobileCLIP", _FakeMobileCLIP)

    prompts_mod.TextPromptEncoder("mobileclip2:b", device=torch.device("cpu"), weight_path=None)

    assert calls == [("b", str((tmp_path / "cache" / "assets" / "text").resolve()))]
