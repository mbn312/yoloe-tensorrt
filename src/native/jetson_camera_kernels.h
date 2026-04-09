#pragma once

#include <cuda_runtime_api.h>

void launch_jetson_zero_copy_preprocess(
    void const* src_ptr,
    int src_width,
    int src_height,
    int src_pitch,
    int dst_width,
    int dst_height,
    bool input_is_bgrx,
    bool output_fp16,
    void* dst_ptr,
    cudaStream_t stream);
