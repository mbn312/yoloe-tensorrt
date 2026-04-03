from __future__ import annotations

import numpy as np
import torch

_NUMPY_TO_TORCH = {
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.bool_): torch.bool,
}


def torch_from_numpy_safe(array: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    contiguous = np.ascontiguousarray(array)
    torch_dtype = _NUMPY_TO_TORCH.get(contiguous.dtype)
    if torch_dtype is None:
        raise TypeError(f"Unsupported numpy dtype '{contiguous.dtype}'")
    tensor = torch.frombuffer(contiguous.data, dtype=torch_dtype).reshape(contiguous.shape).clone()
    if device is not None:
        tensor = tensor.to(device)
    return tensor
