from __future__ import annotations

import numpy as np
from yoloe_tensorrt.prompts import build_visual_prompt_batch


def test_visual_prompt_batch_groups_duplicate_class_names() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    prompts = build_visual_prompt_batch(
        image=image,
        dst_shape=(640, 640),
        visual_stride=8,
        bboxes=[[10, 20, 50, 80], [120, 100, 220, 260], [260, 160, 320, 220]],
        classes=["apple", "apple", "banana"],
    )

    assert prompts.names == ["apple", "banana"]
    assert prompts.tensor.shape == (1, 2, 80, 80)
