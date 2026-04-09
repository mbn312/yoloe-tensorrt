#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>

enum class CudaPreprocessInputFormat : int {
    kBGR = 0,
    kBGRX = 1,
    kRGBA = 2,
};

struct LetterboxGeometry {
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
};

inline LetterboxGeometry compute_letterbox_geometry(int src_width, int src_height, int dst_width, int dst_height)
{
    double const gain = std::min(
        static_cast<double>(dst_width) / static_cast<double>(src_width),
        static_cast<double>(dst_height) / static_cast<double>(src_height));
    LetterboxGeometry geometry;
    geometry.resized_width = std::max(1, static_cast<int>(std::llround(static_cast<double>(src_width) * gain)));
    geometry.resized_height = std::max(1, static_cast<int>(std::llround(static_cast<double>(src_height) * gain)));
    geometry.pad_left = (dst_width - geometry.resized_width) / 2;
    geometry.pad_top = (dst_height - geometry.resized_height) / 2;
    return geometry;
}

void launch_cuda_preprocess(
    void const* src_ptr,
    int src_width,
    int src_height,
    int src_pitch,
    CudaPreprocessInputFormat input_format,
    int dst_width,
    int dst_height,
    bool output_fp16,
    void* dst_ptr,
    cudaStream_t stream);
