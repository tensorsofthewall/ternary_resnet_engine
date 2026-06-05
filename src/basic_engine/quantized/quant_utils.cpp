#include "quantized/quant_utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

float choose_symmetric_i8_scale(const Tensor& x) {
    if (x.data.empty()) {
        throw std::runtime_error("Cannot quantize empty Tensor");
    }

    float max_abs = 0.0f;

    for (float v : x.data) {
        max_abs = std::max(max_abs, std::abs(v));
    }

    if (max_abs == 0.0f) {
        return 1.0f;
    }

    return max_abs / 127.0f;
}

QuantTensorI8 quantize_tensor_symmetric_i8(const Tensor& x) {
    const float scale = choose_symmetric_i8_scale(x);
    const float inv_scale = 1.0f / scale;

    QuantTensorI8 q(x.c, x.h, x.w, scale);

    for (size_t i = 0; i < x.data.size(); ++i) {
        int v = static_cast<int>(std::lrint(x.data[i] * inv_scale));

        v = std::max(-127, std::min(127, v));

        q.data[i] = static_cast<int8_t>(v);
    }

    return q;
}

Tensor dequantize_tensor_i8(const QuantTensorI8& xq) {
    Tensor y(xq.c, xq.h, xq.w);

    for (size_t i = 0; i < xq.data.size(); ++i) {
        y.data[i] = static_cast<float>(xq.data[i]) * xq.scale;
    }

    return y;
}

QuantTensorI8 pad_quant_tensor_i8(
    const QuantTensorI8& x,
    int pad_h,
    int pad_w
) {
    QuantTensorI8 y(
        x.c,
        x.h + 2 * pad_h,
        x.w + 2 * pad_w,
        x.scale
    );

    for (int c = 0; c < x.c; ++c) {
        for (int h = 0; h < x.h; ++h) {
            for (int w = 0; w < x.w; ++w) {
                y(c, h + pad_h, w + pad_w) = x(c, h, w);
            }
        }
    }

    return y;
}

QuantTensorI8& quantize_pad_tensor_symmetric_i8_into(
    const Tensor& x,
    int pad_h,
    int pad_w,
    QuantTensorI8& out
) {
    if (x.data.empty()) {
        throw std::runtime_error("Cannot quantize empty Tensor");
    }

    float max_abs = 0.0f;

    for (float v : x.data) {
        max_abs = std::max(max_abs, std::abs(v));
    }

    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
    const float inv_scale = 1.0f / scale;

    const int padded_h = x.h + 2 * pad_h;
    const int padded_w = x.w + 2 * pad_w;

    out.c = x.c;
    out.h = padded_h;
    out.w = padded_w;
    out.scale = scale;

    const size_t required_size =
        static_cast<size_t>(out.c * out.h * out.w);

    out.data.assign(required_size, static_cast<int8_t>(0));

    for (int c = 0; c < x.c; ++c) {
        for (int h = 0; h < x.h; ++h) {
            const size_t src_base =
                static_cast<size_t>((c * x.h + h) * x.w);

            const size_t dst_base =
                static_cast<size_t>((c * padded_h + (h + pad_h)) * padded_w + pad_w);

            for (int w = 0; w < x.w; ++w) {
                int v = static_cast<int>(
                    std::lrint(x.data[src_base + static_cast<size_t>(w)] * inv_scale)
                );

                v = std::max(-127, std::min(127, v));

                out.data[dst_base + static_cast<size_t>(w)] =
                    static_cast<int8_t>(v);
            }
        }
    }

    return out;
}