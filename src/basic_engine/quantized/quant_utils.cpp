#include "quantized/quant_utils.hpp"

#include <algorithm>
#include <cmath>
#include <immintrin.h>
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

QuantTensorI8 pad_quant_tensor_i8_asymmetric(
    const QuantTensorI8& x,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right
) {
    QuantTensorI8 y(
        x.c,
        x.h + pad_top + pad_bottom,
        x.w + pad_left + pad_right,
        x.scale
    );

    for (int c = 0; c < x.c; ++c) {
        for (int h = 0; h < x.h; ++h) {
            for (int w = 0; w < x.w; ++w) {
                y(c, h + pad_top, w + pad_left) = x(c, h, w);
            }
        }
    }

    return y;
}

QuantTensorI8& quantize_pad_tensor_symmetric_i8_asymmetric_into(
    const Tensor& x,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    QuantTensorI8& out
) {
    if (x.data.empty()) {
        throw std::runtime_error("Cannot quantize empty Tensor");
    }

    // AVX2 vectorized max-abs scan
    const float* src = x.data.data();
    const size_t n = x.data.size();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 max_v = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        const __m256 v = _mm256_loadu_ps(src + i);
        max_v = _mm256_max_ps(max_v, _mm256_andnot_ps(sign_mask, v));
    }
    alignas(32) float max_arr[8];
    _mm256_store_ps(max_arr, max_v);
    float max_abs = 0.0f;
    for (int j = 0; j < 8; ++j) max_abs = std::max(max_abs, max_arr[j]);
    for (; i < n; ++i) max_abs = std::max(max_abs, std::abs(src[i]));

    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
    const float inv_scale = 1.0f / scale;

    const int padded_h = x.h + pad_top + pad_bottom;
    const int padded_w = x.w + pad_left + pad_right;

    out.c = x.c;
    out.h = padded_h;
    out.w = padded_w;
    out.scale = scale;

    out.data.assign(static_cast<size_t>(out.c * out.h * out.w), static_cast<int8_t>(0));

    const __m256 inv_scale_v = _mm256_set1_ps(inv_scale);
    const __m256 pos127_v    = _mm256_set1_ps(127.0f);
    const __m256 neg127_v    = _mm256_set1_ps(-127.0f);
    const __m128i zero128    = _mm_setzero_si128();

    for (int c = 0; c < x.c; ++c) {
        for (int h = 0; h < x.h; ++h) {
            const size_t src_base = static_cast<size_t>((c * x.h + h) * x.w);
            const size_t dst_base = static_cast<size_t>(
                (c * padded_h + (h + pad_top)) * padded_w + pad_left);

            int w = 0;
            for (; w + 7 < x.w; w += 8) {
                const __m256 vals   = _mm256_loadu_ps(src + src_base + w);
                const __m256 scaled = _mm256_mul_ps(vals, inv_scale_v);
                const __m256 clamped = _mm256_min_ps(_mm256_max_ps(scaled, neg127_v), pos127_v);
                const __m256i i32   = _mm256_cvtps_epi32(clamped);
                const __m128i lo    = _mm256_extracti128_si256(i32, 0);
                const __m128i hi    = _mm256_extracti128_si256(i32, 1);
                const __m128i i8    = _mm_packs_epi16(_mm_packs_epi32(lo, hi), zero128);
                _mm_storel_epi64(
                    reinterpret_cast<__m128i*>(out.data.data() + dst_base + w), i8);
            }
            for (; w < x.w; ++w) {
                int v = static_cast<int>(std::lrint(src[src_base + w] * inv_scale));
                v = std::max(-127, std::min(127, v));
                out.data[dst_base + w] = static_cast<int8_t>(v);
            }
        }
    }

    return out;
}