from __future__ import annotations

from io import StringIO

import numpy as np
from yoloe_tensorrt import configure_logging
from yoloe_tensorrt.prompts import build_visual_prompt_batch


def test_configure_logging_retargets_stream_and_formats_messages() -> None:
    stream = StringIO()
    configure_logging(level="INFO", stream=stream, include_timestamps=True)

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    build_visual_prompt_batch(
        image=image,
        dst_shape=(640, 640),
        visual_stride=8,
        bboxes=[[10, 20, 50, 80]],
        classes=["apple"],
    )

    logged = stream.getvalue()
    assert "INFO yoloe_tensorrt.prompts:" in logged
    assert "Building visual prompt batch" in logged
    assert logged[:4].isdigit()
    configure_logging(level="INFO")
