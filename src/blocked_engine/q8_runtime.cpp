#include "q8_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void quantize_blocked_fp32_to_i8(const BlockedTensorFP32& src, BlockedTensorI8& dst) {
    if (src.data.empty())
        throw std::runtime_error("quantize_blocked: empty src tensor");

    float max_abs = 0.0f;
    for (float v : src.data) max_abs = std::max(max_abs, std::abs(v));

    const float scale     = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
    const float inv_scale = 1.0f / scale;

    dst.reuse(src.CB, src.H, src.W, scale);

    for (size_t i = 0; i < src.data.size(); ++i) {
        int v = static_cast<int>(std::lrint(src.data[i] * inv_scale));
        v = std::max(-127, std::min(127, v));
        dst.data[i] = static_cast<int8_t>(v);
    }
}

void dequantize_blocked_i8_to_fp32(const BlockedTensorI8& src, BlockedTensorFP32& dst) {
    dst.reuse(src.CB, src.H, src.W);
    for (size_t i = 0; i < src.data.size(); ++i)
        dst.data[i] = static_cast<float>(src.data[i]) * src.scale;
}
