#include "preprocess_kernels.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

template <typename T>
__device__ inline void store_value(T* dst, int offset, float value);

template <>
__device__ inline void store_value<float>(float* dst, int offset, float value)
{
    dst[offset] = value;
}

template <>
__device__ inline void store_value<__half>(__half* dst, int offset, float value)
{
    dst[offset] = __float2half(value);
}

__device__ inline float bilinear_sample_channel(
    unsigned char const* src,
    int src_width,
    int src_height,
    int src_pitch,
    int src_channels,
    int x,
    int y,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    int channel_index)
{
    float const src_x = ((static_cast<float>(x - pad_left) + 0.5f) * static_cast<float>(src_width) /
                            static_cast<float>(resized_width)) -
        0.5f;
    float const src_y = ((static_cast<float>(y - pad_top) + 0.5f) * static_cast<float>(src_height) /
                            static_cast<float>(resized_height)) -
        0.5f;

    int const x0 = max(0, min(static_cast<int>(floorf(src_x)), src_width - 1));
    int const y0 = max(0, min(static_cast<int>(floorf(src_y)), src_height - 1));
    int const x1 = max(0, min(x0 + 1, src_width - 1));
    int const y1 = max(0, min(y0 + 1, src_height - 1));
    float const wx = src_x - floorf(src_x);
    float const wy = src_y - floorf(src_y);

    unsigned char const* row0 = src + static_cast<std::size_t>(y0 * src_pitch);
    unsigned char const* row1 = src + static_cast<std::size_t>(y1 * src_pitch);
    unsigned char const* p00 = row0 + static_cast<std::size_t>(x0 * src_channels);
    unsigned char const* p01 = row0 + static_cast<std::size_t>(x1 * src_channels);
    unsigned char const* p10 = row1 + static_cast<std::size_t>(x0 * src_channels);
    unsigned char const* p11 = row1 + static_cast<std::size_t>(x1 * src_channels);

    float const c00 = static_cast<float>(p00[channel_index]);
    float const c01 = static_cast<float>(p01[channel_index]);
    float const c10 = static_cast<float>(p10[channel_index]);
    float const c11 = static_cast<float>(p11[channel_index]);
    float const top = c00 + (c01 - c00) * wx;
    float const bottom = c10 + (c11 - c10) * wx;
    return (top + (bottom - top) * wy) / 255.0f;
}

template <typename T>
__global__ void letterbox_u8_to_nchw_kernel(
    unsigned char const* src,
    int src_width,
    int src_height,
    int src_pitch,
    int src_channels,
    int red_index,
    int green_index,
    int blue_index,
    int dst_width,
    int dst_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    T* dst)
{
    int const x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int const y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_width || y >= dst_height) {
        return;
    }

    float constexpr kPadValue = 114.0f / 255.0f;
    int const plane_stride = dst_width * dst_height;
    int const dst_offset = y * dst_width + x;

    if (x < pad_left || x >= (pad_left + resized_width) || y < pad_top || y >= (pad_top + resized_height)) {
        store_value(dst, dst_offset, kPadValue);
        store_value(dst, plane_stride + dst_offset, kPadValue);
        store_value(dst, (2 * plane_stride) + dst_offset, kPadValue);
        return;
    }

    float const red = bilinear_sample_channel(
        src,
        src_width,
        src_height,
        src_pitch,
        src_channels,
        x,
        y,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        red_index);
    float const green = bilinear_sample_channel(
        src,
        src_width,
        src_height,
        src_pitch,
        src_channels,
        x,
        y,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        green_index);
    float const blue = bilinear_sample_channel(
        src,
        src_width,
        src_height,
        src_pitch,
        src_channels,
        x,
        y,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        blue_index);

    store_value(dst, dst_offset, red);
    store_value(dst, plane_stride + dst_offset, green);
    store_value(dst, (2 * plane_stride) + dst_offset, blue);
}

} // namespace

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
    cudaStream_t stream)
{
    int src_channels = 0;
    int red_index = 0;
    int green_index = 1;
    int blue_index = 2;
    switch (input_format) {
    case CudaPreprocessInputFormat::kBGR:
        src_channels = 3;
        red_index = 2;
        green_index = 1;
        blue_index = 0;
        break;
    case CudaPreprocessInputFormat::kBGRX:
        src_channels = 4;
        red_index = 2;
        green_index = 1;
        blue_index = 0;
        break;
    case CudaPreprocessInputFormat::kRGBA:
        src_channels = 4;
        red_index = 0;
        green_index = 1;
        blue_index = 2;
        break;
    }

    auto const geometry = compute_letterbox_geometry(src_width, src_height, dst_width, dst_height);

    dim3 const threads(16, 16);
    dim3 const blocks(
        static_cast<unsigned int>((dst_width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((dst_height + threads.y - 1) / threads.y));

    if (output_fp16) {
        letterbox_u8_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<unsigned char const*>(src_ptr),
            src_width,
            src_height,
            src_pitch,
            src_channels,
            red_index,
            green_index,
            blue_index,
            dst_width,
            dst_height,
            geometry.resized_width,
            geometry.resized_height,
            geometry.pad_left,
            geometry.pad_top,
            static_cast<__half*>(dst_ptr));
    } else {
        letterbox_u8_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<unsigned char const*>(src_ptr),
            src_width,
            src_height,
            src_pitch,
            src_channels,
            red_index,
            green_index,
            blue_index,
            dst_width,
            dst_height,
            geometry.resized_width,
            geometry.resized_height,
            geometry.pad_left,
            geometry.pad_top,
            static_cast<float*>(dst_ptr));
    }
}
