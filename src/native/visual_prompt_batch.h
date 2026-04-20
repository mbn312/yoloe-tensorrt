#pragma once

#include "runtime_common.h"

namespace yoloe_native {

struct VisualPromptHostBatch {
    std::vector<float> tensor;
    int prompt_count = 0;
    int prompt_height = 0;
    int prompt_width = 0;
};

inline std::vector<int> parse_visual_categories(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> const& category_py)
{
    auto const info = category_py.request();
    if (info.ndim != 1) {
        throw std::runtime_error("visual prompt categories must have shape (N,)");
    }
    auto const count = static_cast<std::size_t>(info.shape[0]);
    auto const* src = static_cast<std::int32_t const*>(info.ptr);
    std::vector<int> categories(count);
    int max_category = -1;
    for (std::size_t index = 0; index < count; ++index) {
        if (src[index] < 0) {
            throw std::runtime_error("visual prompt categories must be non-negative");
        }
        categories[index] = static_cast<int>(src[index]);
        max_category = std::max(max_category, categories[index]);
    }
    if (count == 0 || max_category < 0) {
        throw std::runtime_error("visual prompt categories must not be empty");
    }
    return categories;
}

inline void validate_visual_prompt_count(std::size_t prompt_count, std::vector<int> const& categories)
{
    if (prompt_count != categories.size()) {
        throw std::runtime_error("visual prompt categories must match the number of prompts");
    }
}

inline std::pair<float, float> compute_visual_prompt_letterbox_padding(
    int src_h,
    int src_w,
    int dst_h,
    int dst_w)
{
    float const gain = std::min(static_cast<float>(dst_h) / static_cast<float>(src_h), static_cast<float>(dst_w) / static_cast<float>(src_w));
    float const pad_x = std::round((static_cast<float>(dst_w) - std::round(static_cast<float>(src_w) * gain)) / 2.0f - 0.1f);
    float const pad_y = std::round((static_cast<float>(dst_h) - std::round(static_cast<float>(src_h) * gain)) / 2.0f - 0.1f);
    return { pad_x, pad_y };
}

inline VisualPromptHostBatch build_visual_prompt_batch_from_boxes(
    py::array_t<float, py::array::c_style | py::array::forcecast> const& boxes_py,
    std::vector<int> const& categories,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    int visual_stride)
{
    auto const info = boxes_py.request();
    if (info.ndim != 2 || info.shape[1] != 4) {
        throw std::runtime_error("visual prompt bboxes must have shape (N, 4)");
    }
    auto const prompt_count = static_cast<std::size_t>(info.shape[0]);
    validate_visual_prompt_count(prompt_count, categories);

    int const prompt_height = dst_h / visual_stride;
    int const prompt_width = dst_w / visual_stride;
    int const unique_prompt_count = *std::max_element(categories.begin(), categories.end()) + 1;

    VisualPromptHostBatch batch;
    batch.prompt_count = unique_prompt_count;
    batch.prompt_height = prompt_height;
    batch.prompt_width = prompt_width;
    batch.tensor.assign(
        static_cast<std::size_t>(unique_prompt_count) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width),
        0.0f);

    float const gain = std::min(static_cast<float>(dst_h) / static_cast<float>(src_h), static_cast<float>(dst_w) / static_cast<float>(src_w));
    auto const [pad_x, pad_y] = compute_visual_prompt_letterbox_padding(src_h, src_w, dst_h, dst_w);
    auto const* boxes = static_cast<float const*>(info.ptr);

    for (std::size_t index = 0; index < prompt_count; ++index) {
        float const x1 = ((boxes[(index * 4) + 0] * gain) + pad_x) / static_cast<float>(visual_stride);
        float const y1 = ((boxes[(index * 4) + 1] * gain) + pad_y) / static_cast<float>(visual_stride);
        float const x2 = ((boxes[(index * 4) + 2] * gain) + pad_x) / static_cast<float>(visual_stride);
        float const y2 = ((boxes[(index * 4) + 3] * gain) + pad_y) / static_cast<float>(visual_stride);

        int const left = std::max(0, std::min(prompt_width, static_cast<int>(std::ceil(x1))));
        int const top = std::max(0, std::min(prompt_height, static_cast<int>(std::ceil(y1))));
        int const right = std::max(0, std::min(prompt_width, static_cast<int>(std::ceil(x2))));
        int const bottom = std::max(0, std::min(prompt_height, static_cast<int>(std::ceil(y2))));
        if (left >= right || top >= bottom) {
            continue;
        }

        auto const plane_offset =
            static_cast<std::size_t>(categories[index]) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width);
        for (int y = top; y < bottom; ++y) {
            auto* row = batch.tensor.data() + plane_offset + static_cast<std::size_t>(y * prompt_width);
            std::fill(row + left, row + right, 1.0f);
        }
    }

    return batch;
}

inline cv::Mat letterbox_mask_nearest(cv::Mat const& src, int dst_h, int dst_w)
{
    float const gain = std::min(static_cast<float>(dst_h) / static_cast<float>(src.rows), static_cast<float>(dst_w) / static_cast<float>(src.cols));
    int const resized_w = std::max(1, static_cast<int>(std::round(static_cast<float>(src.cols) * gain)));
    int const resized_h = std::max(1, static_cast<int>(std::round(static_cast<float>(src.rows) * gain)));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_NEAREST);

    float const dw = static_cast<float>(dst_w - resized_w) / 2.0f;
    float const dh = static_cast<float>(dst_h - resized_h) / 2.0f;
    int const top = std::round(dh - 0.1f);
    int const bottom = std::round(dh + 0.1f);
    int const left = std::round(dw - 0.1f);
    int const right = std::round(dw + 0.1f);

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
    return output;
}

inline VisualPromptHostBatch build_visual_prompt_batch_from_masks(
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> const& masks_py,
    std::vector<int> const& categories,
    int dst_h,
    int dst_w,
    int visual_stride)
{
    auto const info = masks_py.request();
    if (info.ndim != 3) {
        throw std::runtime_error("visual prompt masks must have shape (N, H, W)");
    }
    auto const prompt_count = static_cast<std::size_t>(info.shape[0]);
    validate_visual_prompt_count(prompt_count, categories);

    int const src_h = static_cast<int>(info.shape[1]);
    int const src_w = static_cast<int>(info.shape[2]);
    int const prompt_height = dst_h / visual_stride;
    int const prompt_width = dst_w / visual_stride;
    int const unique_prompt_count = *std::max_element(categories.begin(), categories.end()) + 1;

    VisualPromptHostBatch batch;
    batch.prompt_count = unique_prompt_count;
    batch.prompt_height = prompt_height;
    batch.prompt_width = prompt_width;
    batch.tensor.assign(
        static_cast<std::size_t>(unique_prompt_count) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width),
        0.0f);

    auto const* src = static_cast<std::uint8_t const*>(info.ptr);
    auto const mask_plane = static_cast<std::size_t>(src_h) * static_cast<std::size_t>(src_w);
    for (std::size_t index = 0; index < prompt_count; ++index) {
        cv::Mat const mask(src_h, src_w, CV_8UC1, const_cast<std::uint8_t*>(src + (index * mask_plane)));
        cv::Mat const letterboxed = letterbox_mask_nearest(mask, dst_h, dst_w);
        cv::Mat resized;
        cv::resize(letterboxed, resized, cv::Size(prompt_width, prompt_height), 0.0, 0.0, cv::INTER_NEAREST);
        auto const plane_offset =
            static_cast<std::size_t>(categories[index]) * static_cast<std::size_t>(prompt_height) * static_cast<std::size_t>(prompt_width);
        for (int y = 0; y < prompt_height; ++y) {
            auto const* row = resized.ptr<std::uint8_t>(y);
            auto* dst_row = batch.tensor.data() + plane_offset + static_cast<std::size_t>(y * prompt_width);
            for (int x = 0; x < prompt_width; ++x) {
                dst_row[x] = (dst_row[x] != 0.0f || row[x] != 0U) ? 1.0f : 0.0f;
            }
        }
    }

    return batch;
}

} // namespace yoloe_native
