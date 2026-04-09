#include "jetson_camera_kernels.h"

#include <cuda_fp16.h>

#include <cmath>
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

template <typename T>
__global__ void letterbox_bgrx_to_nchw_kernel(
    unsigned char const* src,
    int src_width,
    int src_height,
    int src_pitch,
    int dst_width,
    int dst_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    bool input_is_bgrx,
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
    unsigned char const* p00 = row0 + static_cast<std::size_t>(x0 * 4);
    unsigned char const* p01 = row0 + static_cast<std::size_t>(x1 * 4);
    unsigned char const* p10 = row1 + static_cast<std::size_t>(x0 * 4);
    unsigned char const* p11 = row1 + static_cast<std::size_t>(x1 * 4);

    auto sample_channel = [&](int index) -> float {
        float const c00 = static_cast<float>(p00[index]);
        float const c01 = static_cast<float>(p01[index]);
        float const c10 = static_cast<float>(p10[index]);
        float const c11 = static_cast<float>(p11[index]);
        float const top = c00 + (c01 - c00) * wx;
        float const bottom = c10 + (c11 - c10) * wx;
        return (top + (bottom - top) * wy) / 255.0f;
    };

    float red;
    float green;
    float blue;
    if (input_is_bgrx) {
        blue = sample_channel(0);
        green = sample_channel(1);
        red = sample_channel(2);
    } else {
        red = sample_channel(0);
        green = sample_channel(1);
        blue = sample_channel(2);
    }

    store_value(dst, dst_offset, red);
    store_value(dst, plane_stride + dst_offset, green);
    store_value(dst, (2 * plane_stride) + dst_offset, blue);
}

} // namespace

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
    cudaStream_t stream)
{
    float const gain = fminf(
        static_cast<float>(dst_width) / static_cast<float>(src_width),
        static_cast<float>(dst_height) / static_cast<float>(src_height));
    int const resized_width = max(1, static_cast<int>(llroundf(static_cast<float>(src_width) * gain)));
    int const resized_height = max(1, static_cast<int>(llroundf(static_cast<float>(src_height) * gain)));
    int const pad_left = (dst_width - resized_width) / 2;
    int const pad_top = (dst_height - resized_height) / 2;

    dim3 const threads(16, 16);
    dim3 const blocks(
        static_cast<unsigned int>((dst_width + threads.x - 1) / threads.x),
        static_cast<unsigned int>((dst_height + threads.y - 1) / threads.y));

    if (output_fp16) {
        letterbox_bgrx_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<unsigned char const*>(src_ptr),
            src_width,
            src_height,
            src_pitch,
            dst_width,
            dst_height,
            resized_width,
            resized_height,
            pad_left,
            pad_top,
            input_is_bgrx,
            static_cast<__half*>(dst_ptr));
    } else {
        letterbox_bgrx_to_nchw_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<unsigned char const*>(src_ptr),
            src_width,
            src_height,
            src_pitch,
            dst_width,
            dst_height,
            resized_width,
            resized_height,
            pad_left,
            pad_top,
            input_is_bgrx,
            static_cast<float*>(dst_ptr));
    }
}
