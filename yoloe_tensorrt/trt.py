from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .artifacts import ShapeProfile
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def _numpy_to_torch_dtype(dtype: np.dtype) -> torch.dtype:
    normalized = np.dtype(dtype)
    if normalized == np.float16:
        return torch.float16
    if normalized == np.float32:
        return torch.float32
    if normalized == np.int32:
        return torch.int32
    if normalized == np.int64:
        return torch.int64
    if normalized == np.bool_:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype '{normalized}'")


@dataclass(frozen=True)
class TensorBinding:
    name: str
    index: int
    is_input: bool
    dtype: np.dtype
    torch_dtype: torch.dtype
    shape: tuple[int, ...]
    profile_shape: ShapeProfile | None


class TensorRTRuntime:
    def __init__(self, engine_path: str | Path, device: str | torch.device = "cuda:0") -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT is required to load an engine") from exc

        self.trt = trt
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine_path = Path(engine_path)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine '{self.engine_path}'")
        self.context = self.engine.create_execution_context()
        LOGGER.info("Loaded TensorRT engine '%s' on device '%s'", self.engine_path, self.device)
        self.uses_tensor_io_api = all(
            hasattr(self.engine, name)
            for name in (
                "num_io_tensors",
                "get_tensor_name",
                "get_tensor_mode",
                "get_tensor_dtype",
                "get_tensor_shape",
                "get_tensor_profile_shape",
            )
        ) and hasattr(self.context, "set_input_shape")
        self._binding_count = self.engine.num_io_tensors if self.uses_tensor_io_api else self.engine.num_bindings
        self.bindings: OrderedDict[str, TensorBinding] = OrderedDict()
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        if self.uses_tensor_io_api:
            self._binding_name = self.engine.get_tensor_name
            self._binding_is_input = self._trt10_binding_is_input
            self._binding_dtype = self._trt10_binding_dtype
            self._binding_shape = self._trt10_binding_shape
            self._binding_profile = self._trt10_binding_profile
            self._set_context_shape = self.context.set_input_shape
            self._get_context_shape = self.context.get_tensor_shape
            if hasattr(self.context, "execute_async_v2"):
                self._execute = self._execute_async
            else:
                self._execute = self._execute_sync
        else:
            self._binding_name = self.engine.get_binding_name
            self._binding_is_input = self.engine.binding_is_input
            self._binding_dtype = self._trt8_binding_dtype
            self._binding_shape = self._trt8_binding_shape
            self._binding_profile = self._trt8_binding_profile
            self._set_context_shape = self._set_binding_shape
            self._get_context_shape = self._get_binding_shape
            if hasattr(self.context, "execute_async_v2"):
                self._execute = self._execute_async
            else:
                self._execute = self._execute_sync

        for index in range(self._binding_count):
            name = self._binding_name(index)
            is_input = self._binding_is_input(index)
            dtype = self._binding_dtype(index)
            shape = self._binding_shape(index)
            profile = None
            if is_input and -1 in shape:
                minimum, optimum, maximum = self._binding_profile(index)
                profile = ShapeProfile(
                    minimum=tuple(int(v) for v in minimum),
                    optimum=tuple(int(v) for v in optimum),
                    maximum=tuple(int(v) for v in maximum),
                )
            binding = TensorBinding(
                name=name,
                index=index,
                is_input=is_input,
                dtype=dtype,
                torch_dtype=_numpy_to_torch_dtype(dtype),
                shape=shape,
                profile_shape=profile,
            )
            self.bindings[name] = binding
            if is_input:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        self.fp16 = any(self.bindings[name].torch_dtype == torch.float16 for name in self.input_names)
        LOGGER.info(
            "Engine '%s' bindings ready: inputs=%s outputs=%s fp16=%s",
            self.engine_path.name,
            self.input_names,
            self.output_names,
            self.fp16,
        )

    def _trt10_binding_is_input(self, index: int) -> bool:
        return self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == self.trt.TensorIOMode.INPUT

    def _trt10_binding_dtype(self, index: int) -> np.dtype:
        return np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(self.engine.get_tensor_name(index))))

    def _trt10_binding_shape(self, index: int) -> tuple[int, ...]:
        return tuple(int(v) for v in self.engine.get_tensor_shape(self.engine.get_tensor_name(index)))

    def _trt10_binding_profile(self, index: int):
        return self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(index), 0)

    def _trt8_binding_dtype(self, index: int) -> np.dtype:
        return np.dtype(self.trt.nptype(self.engine.get_binding_dtype(index)))

    def _trt8_binding_shape(self, index: int) -> tuple[int, ...]:
        return tuple(int(v) for v in self.engine.get_binding_shape(index))

    def _trt8_binding_profile(self, index: int):
        return self.engine.get_profile_shape(0, index)

    def _set_binding_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self.context.set_binding_shape(self.bindings[name].index, shape)

    def _get_binding_shape(self, name: str):
        return self.context.get_binding_shape(self.bindings[name].index)

    def _execute_async(self, binding_addrs: list[int], stream) -> bool:
        return bool(self.context.execute_async_v2(binding_addrs, stream.cuda_stream))

    def _execute_sync(self, binding_addrs: list[int], _stream) -> bool:
        return bool(self.context.execute_v2(binding_addrs))

    def _set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        self._set_context_shape(name, shape)

    def _get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(v) for v in self._get_context_shape(name))

    def _profile_shape(self, name: str) -> tuple[int, ...]:
        binding = self.bindings[name]
        if binding.profile_shape is not None:
            return binding.profile_shape.optimum
        return tuple(int(v) for v in binding.shape)

    def infer(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise KeyError(f"Missing TensorRT inputs: {missing}")
        LOGGER.debug(
            "Running TensorRT inference for '%s' with inputs=%s",
            self.engine_path.name,
            {name: tuple(int(v) for v in tensor.shape) for name, tensor in inputs.items()},
        )

        tensors: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            binding = self.bindings[name]
            tensor = inputs[name]
            if tensor.device != self.device:
                tensor = tensor.to(self.device, non_blocking=True)
            if tensor.dtype != binding.torch_dtype:
                tensor = tensor.to(binding.torch_dtype)
            tensor = tensor.contiguous()
            tensors[name] = tensor
            self._set_input_shape(name, tuple(int(v) for v in tensor.shape))

        if self.uses_tensor_io_api and hasattr(self.context, "infer_shapes"):
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(f"TensorRT could not infer all shapes: {unresolved}")

        binding_addrs: list[int] = []
        outputs: dict[str, torch.Tensor] = {}
        for name, binding in self.bindings.items():
            if binding.is_input:
                tensor = tensors[name]
            else:
                shape = self._get_tensor_shape(name)
                tensor = torch.empty(shape, device=self.device, dtype=binding.torch_dtype)
                outputs[name] = tensor
            binding_addrs.append(int(tensor.data_ptr()))

        stream = torch.cuda.current_stream(device=self.device)
        ok = self._execute(binding_addrs, stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for '{self.engine_path}'")
        LOGGER.debug(
            "Finished TensorRT inference for '%s' with outputs=%s",
            self.engine_path.name,
            {name: tuple(int(v) for v in tensor.shape) for name, tensor in outputs.items()},
        )
        return outputs

    def warmup(self, shapes: Mapping[str, tuple[int, ...]] | None = None, runs: int = 2) -> None:
        LOGGER.info("Warming up TensorRT engine '%s' for %d run(s)", self.engine_path.name, runs)
        inputs = {}
        for name in self.input_names:
            shape = shapes[name] if shapes and name in shapes else self._profile_shape(name)
            dtype = self.bindings[name].torch_dtype
            inputs[name] = torch.zeros(shape, device=self.device, dtype=dtype)
        for _ in range(runs):
            self.infer(inputs)
        LOGGER.info("Warmup complete for '%s'", self.engine_path.name)


def build_engine_from_onnx(
    onnx_path: str | Path,
    engine_path: str | Path,
    profiles: Mapping[str, ShapeProfile],
    fp16: bool = True,
    workspace_bytes: int = 2 << 30,
) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is required to build an engine") from exc

    LOGGER.info(
        "Building TensorRT engine '%s' from ONNX '%s' (fp16=%s, workspace_bytes=%d)",
        engine_path,
        onnx_path,
        fp16,
        workspace_bytes,
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse ONNX model '{onnx_path}':\n{errors}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    builder_optimization_level = os.environ.get("YOLOE_TRT_BUILDER_OPT_LEVEL")
    if builder_optimization_level is not None and hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(builder_optimization_level)
        LOGGER.info("TensorRT builder optimization level set to %s", builder_optimization_level)

    avg_timing_iterations = os.environ.get("YOLOE_TRT_AVG_TIMING_ITERATIONS")
    if avg_timing_iterations is not None and hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = int(avg_timing_iterations)
        LOGGER.info("TensorRT avg timing iterations set to %s", avg_timing_iterations)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    for name, shape_profile in profiles.items():
        profile.set_shape(name, shape_profile.minimum, shape_profile.optimum, shape_profile.maximum)
        LOGGER.info(
            "TensorRT profile for '%s': min=%s opt=%s max=%s",
            name,
            shape_profile.minimum,
            shape_profile.optimum,
            shape_profile.maximum,
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build engine from '{onnx_path}'")

    output_path = Path(engine_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(serialized))
    LOGGER.info("Finished building TensorRT engine '%s'", output_path)
    return output_path
