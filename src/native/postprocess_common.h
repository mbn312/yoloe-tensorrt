#pragma once

#include "runtime_common.h"

namespace yoloe_native {

struct NativePostprocessCandidate {
    std::array<float, 4> input_box{};
    std::array<float, 4> output_box{};
    float confidence = 0.0f;
    int cls = 0;
    std::vector<float> mask_coeffs;
};

struct NativePostprocessResult {
    std::vector<float> boxes;
    std::vector<std::uint8_t> masks;
    int mask_height = 0;
    int mask_width = 0;
    bool include_masks = false;
};

inline float clip_float(float value, float low, float high)
{
    return std::max(low, std::min(value, high));
}

inline void clip_box_inplace(std::array<float, 4>& box, int image_h, int image_w)
{
    box[0] = clip_float(box[0], 0.0f, static_cast<float>(image_w));
    box[1] = clip_float(box[1], 0.0f, static_cast<float>(image_h));
    box[2] = clip_float(box[2], 0.0f, static_cast<float>(image_w));
    box[3] = clip_float(box[3], 0.0f, static_cast<float>(image_h));
}

inline std::array<float, 4> scale_box_to_shape(std::array<float, 4> box, int input_h, int input_w, int output_h, int output_w)
{
    auto const geometry = compute_letterbox_geometry(output_w, output_h, input_w, input_h);
    float const gain = static_cast<float>(geometry.resized_width) / static_cast<float>(output_w);
    float const pad_x = static_cast<float>(geometry.pad_left);
    float const pad_y = static_cast<float>(geometry.pad_top);

    box[0] = (box[0] - pad_x) / gain;
    box[1] = (box[1] - pad_y) / gain;
    box[2] = (box[2] - pad_x) / gain;
    box[3] = (box[3] - pad_y) / gain;
    clip_box_inplace(box, output_h, output_w);
    return box;
}

inline float box_iou(std::array<float, 4> const& lhs, std::array<float, 4> const& rhs)
{
    float const inter_x1 = std::max(lhs[0], rhs[0]);
    float const inter_y1 = std::max(lhs[1], rhs[1]);
    float const inter_x2 = std::min(lhs[2], rhs[2]);
    float const inter_y2 = std::min(lhs[3], rhs[3]);
    float const inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float const inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float const intersection = inter_w * inter_h;

    float const lhs_area = std::max(0.0f, lhs[2] - lhs[0]) * std::max(0.0f, lhs[3] - lhs[1]);
    float const rhs_area = std::max(0.0f, rhs[2] - rhs[0]) * std::max(0.0f, rhs[3] - rhs[1]);
    float const denominator = lhs_area + rhs_area - intersection;
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    return intersection / denominator;
}

inline cv::Mat make_float_mask_view(std::vector<float> const& data, int height, int width)
{
    return cv::Mat(height, width, CV_32FC1, const_cast<float*>(data.data()));
}

inline void crop_mask_inplace(std::vector<float>& mask, int height, int width, std::array<float, 4> const& box)
{
    int const left = std::max(0, std::min(width, static_cast<int>(std::ceil(box[0]))));
    int const top = std::max(0, std::min(height, static_cast<int>(std::ceil(box[1]))));
    int const right = std::max(0, std::min(width, static_cast<int>(std::ceil(box[2]))));
    int const bottom = std::max(0, std::min(height, static_cast<int>(std::ceil(box[3]))));

    if (left >= right || top >= bottom) {
        std::fill(mask.begin(), mask.end(), 0.0f);
        return;
    }

    for (int y = 0; y < top; ++y) {
        std::fill_n(mask.data() + static_cast<std::size_t>(y * width), width, 0.0f);
    }
    for (int y = bottom; y < height; ++y) {
        std::fill_n(mask.data() + static_cast<std::size_t>(y * width), width, 0.0f);
    }
    for (int y = top; y < bottom; ++y) {
        auto* row = mask.data() + static_cast<std::size_t>(y * width);
        std::fill(row, row + left, 0.0f);
        std::fill(row + right, row + width, 0.0f);
    }
}

inline std::vector<float> resize_mask(std::vector<float> const& mask, int src_h, int src_w, int dst_h, int dst_w)
{
    cv::Mat const src = make_float_mask_view(mask, src_h, src_w);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR);
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }
    std::vector<float> result(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w));
    std::memcpy(result.data(), resized.ptr<float>(), result.size() * sizeof(float));
    return result;
}

inline std::vector<float> scale_mask_to_shape(std::vector<float> const& mask, int src_h, int src_w, int dst_h, int dst_w)
{
    auto const geometry = compute_letterbox_geometry(dst_w, dst_h, src_w, src_h);
    int const top = std::max(0, std::min(src_h, geometry.pad_top));
    int const left = std::max(0, std::min(src_w, geometry.pad_left));
    int const bottom = std::max(top + 1, std::min(src_h, top + geometry.resized_height));
    int const right = std::max(left + 1, std::min(src_w, left + geometry.resized_width));

    cv::Mat const src = make_float_mask_view(mask, src_h, src_w);
    cv::Mat cropped = src(cv::Range(top, bottom), cv::Range(left, right));
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(dst_w, dst_h), 0.0, 0.0, cv::INTER_LINEAR);
    if (!resized.isContinuous()) {
        resized = resized.clone();
    }
    std::vector<float> result(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w));
    std::memcpy(result.data(), resized.ptr<float>(), result.size() * sizeof(float));
    return result;
}

inline bool threshold_mask(std::vector<float> const& src, std::vector<std::uint8_t>& dst)
{
    dst.resize(src.size());
    bool any = false;
    for (std::size_t i = 0; i < src.size(); ++i) {
        auto const value = static_cast<std::uint8_t>(src[i] > 0.0f ? 1U : 0U);
        dst[i] = value;
        any = any || value != 0U;
    }
    return any;
}

} // namespace yoloe_native
