#include "quantized/conv3x3_q8_pattern_ocblock4_avx2.hpp"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

#include <omp.h>

bool can_use_q8_ocblock4_layer4_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 512 &&
           conv.out_channels == 512 &&
           conv.out_channels % 4 == 0;
}

bool can_use_q8_ocblock4_s2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 128 &&
           conv.out_channels == 256 &&
           conv.out_channels % 4 == 0;
}

namespace {

inline int32_t q8_at(
    const QuantTensorI8& xq,
    int c,
    int h,
    int w
) {
    return static_cast<int32_t>(xq(c, h, w));
}

inline __m256i make_tail_mask_i32(int valid_lanes) {
    alignas(32) int mask_values[8];

    for (int i = 0; i < 8; ++i) {
        mask_values[i] = (i < valid_lanes) ? -1 : 0;
    }

    return _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(mask_values)
    );
}

inline void store_width_masked_8(
    float* base,
    __m256 value,
    int valid_lanes
) {
    const __m256i mask = make_tail_mask_i32(valid_lanes);
    _mm256_maskstore_ps(base, mask, value);
}

inline __m256i load_i8_width7_to_i32_safe(const int8_t* base) {
    alignas(16) int8_t tmp[8];

    tmp[0] = base[0];
    tmp[1] = base[1];
    tmp[2] = base[2];
    tmp[3] = base[3];
    tmp[4] = base[4];
    tmp[5] = base[5];
    tmp[6] = base[6];
    tmp[7] = 0;

    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(tmp));

    return _mm256_cvtepi8_epi32(bytes);
}

inline __m256i load_i8_width8_stride2_to_i32_fast(const int8_t* base) {
    alignas(16) int8_t tmp[8];

    tmp[0] = base[0];
    tmp[1] = base[2];
    tmp[2] = base[4];
    tmp[3] = base[6];
    tmp[4] = base[8];
    tmp[5] = base[10];
    tmp[6] = base[12];
    tmp[7] = base[14];

    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(tmp));

    return _mm256_cvtepi8_epi32(bytes);
}

inline void apply_weight_i32(
    int32_t& acc,
    int32_t x,
    int8_t w
) {
    if (w == 1) {
        acc += x;
    } else if (w == -1) {
        acc -= x;
    }
}

Tensor conv3x3_q8_ocblock4_layer4_s1_bn_avx2_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu
) {
    if (!can_use_q8_ocblock4_layer4_s1_3x3_avx2(conv)) {
        throw std::runtime_error(
            "q8 OC-Block4 layer4 stride-1 kernel called on unsupported conv"
        );
    }

    const int out_h = (xq.h - conv.kh) / conv.stride_h + 1;

    const int out_w = (xq.w - conv.kw) / conv.stride_w + 1;

    if (out_w != 7) {
        throw std::runtime_error(
            "q8 OC-Block4 layer4 stride-1 kernel expected out_w == 7"
        );
    }

    Tensor y(conv.out_channels, out_h, out_w);

    #pragma omp parallel for schedule(static)
    for (int block_idx=0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
    // for (const OCBlock4PatternPlan& block : plan.oc_blocks) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float out_scale0 =
            xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float out_scale1 =
            xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float out_scale2 =
            xq.scale * bn.scale[static_cast<size_t>(oc + 2)];
        const float out_scale3 =
            xq.scale * bn.scale[static_cast<size_t>(oc + 3)];

        const float bias0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float bias1 = bn.bias[static_cast<size_t>(oc + 1)];
        const float bias2 = bn.bias[static_cast<size_t>(oc + 2)];
        const float bias3 = bn.bias[static_cast<size_t>(oc + 3)];

        constexpr int valid_lanes = 7;
        const __m256 zero_v = _mm256_setzero_ps();

        for (int oh = 0; oh < out_h; ++oh) {
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            for (const OCBlock4PatternBucket& bucket : block.buckets) {
                const int8_t w0 = bucket.w0;
                const int8_t w1 = bucket.w1;
                const int8_t w2 = bucket.w2;
                const int8_t w3 = bucket.w3;

                for (const OCBlock4PatternAction& action : bucket.actions) {
                    const int ih =
                        oh + static_cast<int>(action.ky);

                    const int iw =
                        static_cast<int>(action.kx);

                    const int base =
                        (static_cast<int>(action.ic) * xq.h + ih) * xq.w + iw;

                    const __m256i xvec = load_i8_width7_to_i32_safe(
                        xq.data.data() + static_cast<size_t>(base)
                    );

                    if (w0 == 1) {
                        acc0 = _mm256_add_epi32(acc0, xvec);
                    } else if (w0 == -1) {
                        acc0 = _mm256_sub_epi32(acc0, xvec);
                    }

                    if (w1 == 1) {
                        acc1 = _mm256_add_epi32(acc1, xvec);
                    } else if (w1 == -1) {
                        acc1 = _mm256_sub_epi32(acc1, xvec);
                    }

                    if (w2 == 1) {
                        acc2 = _mm256_add_epi32(acc2, xvec);
                    } else if (w2 == -1) {
                        acc2 = _mm256_sub_epi32(acc2, xvec);
                    }

                    if (w3 == 1) {
                        acc3 = _mm256_add_epi32(acc3, xvec);
                    } else if (w3 == -1) {
                        acc3 = _mm256_sub_epi32(acc3, xvec);
                    }
                }
            }

            __m256 y0 = _mm256_cvtepi32_ps(acc0);
            __m256 y1 = _mm256_cvtepi32_ps(acc1);
            __m256 y2 = _mm256_cvtepi32_ps(acc2);
            __m256 y3 = _mm256_cvtepi32_ps(acc3);

            y0 = _mm256_add_ps(
                _mm256_mul_ps(y0, _mm256_set1_ps(out_scale0)),
                _mm256_set1_ps(bias0)
            );

            y1 = _mm256_add_ps(
                _mm256_mul_ps(y1, _mm256_set1_ps(out_scale1)),
                _mm256_set1_ps(bias1)
            );

            y2 = _mm256_add_ps(
                _mm256_mul_ps(y2, _mm256_set1_ps(out_scale2)),
                _mm256_set1_ps(bias2)
            );

            y3 = _mm256_add_ps(
                _mm256_mul_ps(y3, _mm256_set1_ps(out_scale3)),
                _mm256_set1_ps(bias3)
            );

            if (apply_relu) {
                y0 = _mm256_max_ps(y0, zero_v);
                y1 = _mm256_max_ps(y1, zero_v);
                y2 = _mm256_max_ps(y2, zero_v);
                y3 = _mm256_max_ps(y3, zero_v);
            }

            store_width_masked_8(y.ptr(oc + 0, oh, 0), y0, valid_lanes);
            store_width_masked_8(y.ptr(oc + 1, oh, 0), y1, valid_lanes);
            store_width_masked_8(y.ptr(oc + 2, oh, 0), y2, valid_lanes);
            store_width_masked_8(y.ptr(oc + 3, oh, 0), y3, valid_lanes);
        }
    }

    return y;
}

Tensor conv3x3_q8_ocblock4_s2_bn_avx2_out_hw_impl(
    const QuantTensorI8& xq,
    int out_h,
    int out_w,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu
) {
    if (!can_use_q8_ocblock4_s2_3x3_avx2(conv)) {
        throw std::runtime_error(
            "q8 OC-Block4 stride-2 kernel called on unsupported conv"
        );
    }

    if (!(out_h == 14 && out_w == 14)) {
        throw std::runtime_error(
            "q8 OC-Block4 stride-2 layer3.0.conv1 expected 14x14 output"
        );
    }

    Tensor y(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();

    for (const OCBlock4PatternPlan& block : plan.oc_blocks) {
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(
            xq.scale * bn.scale[static_cast<size_t>(oc + 0)]
        );
        const __m256 scale1 = _mm256_set1_ps(
            xq.scale * bn.scale[static_cast<size_t>(oc + 1)]
        );
        const __m256 scale2 = _mm256_set1_ps(
            xq.scale * bn.scale[static_cast<size_t>(oc + 2)]
        );
        const __m256 scale3 = _mm256_set1_ps(
            xq.scale * bn.scale[static_cast<size_t>(oc + 3)]
        );

        const __m256 bias0 = _mm256_set1_ps(
            bn.bias[static_cast<size_t>(oc + 0)]
        );
        const __m256 bias1 = _mm256_set1_ps(
            bn.bias[static_cast<size_t>(oc + 1)]
        );
        const __m256 bias2 = _mm256_set1_ps(
            bn.bias[static_cast<size_t>(oc + 2)]
        );
        const __m256 bias3 = _mm256_set1_ps(
            bn.bias[static_cast<size_t>(oc + 3)]
        );

        for (int oh = 0; oh < out_h; ++oh) {
            const int input_row_base = oh * conv.stride_h;

            for (int ow = 0; ow < out_w; ow += 8) {
                const int valid_lanes = std::min(8, out_w - ow);
                const int input_col_base = ow * conv.stride_w;

                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                for (const OCBlock4PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;
                    const int8_t w2 = bucket.w2;
                    const int8_t w3 = bucket.w3;

                    for (const OCBlock4PatternAction& action : bucket.actions) {
                        const int ih =
                            input_row_base + static_cast<int>(action.ky);

                        const int iw =
                            input_col_base + static_cast<int>(action.kx);

                        const int base =
                            (static_cast<int>(action.ic) * xq.h + ih) * xq.w + iw;

                        const __m256i xvec =
                            load_i8_width8_stride2_to_i32_fast(
                                xq.data.data() + static_cast<size_t>(base)
                            );

                        if (w0 == 1) {
                            acc0 = _mm256_add_epi32(acc0, xvec);
                        } else if (w0 == -1) {
                            acc0 = _mm256_sub_epi32(acc0, xvec);
                        }

                        if (w1 == 1) {
                            acc1 = _mm256_add_epi32(acc1, xvec);
                        } else if (w1 == -1) {
                            acc1 = _mm256_sub_epi32(acc1, xvec);
                        }

                        if (w2 == 1) {
                            acc2 = _mm256_add_epi32(acc2, xvec);
                        } else if (w2 == -1) {
                            acc2 = _mm256_sub_epi32(acc2, xvec);
                        }

                        if (w3 == 1) {
                            acc3 = _mm256_add_epi32(acc3, xvec);
                        } else if (w3 == -1) {
                            acc3 = _mm256_sub_epi32(acc3, xvec);
                        }
                    }
                }

                __m256 y0 = _mm256_cvtepi32_ps(acc0);
                __m256 y1 = _mm256_cvtepi32_ps(acc1);
                __m256 y2 = _mm256_cvtepi32_ps(acc2);
                __m256 y3 = _mm256_cvtepi32_ps(acc3);

                y0 = _mm256_add_ps(_mm256_mul_ps(y0, scale0), bias0);
                y1 = _mm256_add_ps(_mm256_mul_ps(y1, scale1), bias1);
                y2 = _mm256_add_ps(_mm256_mul_ps(y2, scale2), bias2);
                y3 = _mm256_add_ps(_mm256_mul_ps(y3, scale3), bias3);

                if (apply_relu) {
                    y0 = _mm256_max_ps(y0, zero_v);
                    y1 = _mm256_max_ps(y1, zero_v);
                    y2 = _mm256_max_ps(y2, zero_v);
                    y3 = _mm256_max_ps(y3, zero_v);
                }

                store_width_masked_8(y.ptr(oc + 0, oh, ow), y0, valid_lanes);
                store_width_masked_8(y.ptr(oc + 1, oh, ow), y1, valid_lanes);
                store_width_masked_8(y.ptr(oc + 2, oh, ow), y2, valid_lanes);
                store_width_masked_8(y.ptr(oc + 3, oh, ow), y3, valid_lanes);
            }
        }
    }

    return y;
}

} // namespace

Tensor conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace&
) {
    return conv3x3_q8_ocblock4_layer4_s1_bn_avx2_impl(
        xq,
        conv,
        plan,
        bn,
        false
    );
}

Tensor conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace&
) {
    return conv3x3_q8_ocblock4_layer4_s1_bn_avx2_impl(
        xq,
        conv,
        plan,
        bn,
        true
    );
}

Tensor conv3x3_q8_ocblock4_s2_bn_avx2_out_hw(
    const QuantTensorI8& xq,
    int out_h,
    int out_w,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace&
) {
    return conv3x3_q8_ocblock4_s2_bn_avx2_out_hw_impl(
        xq,
        out_h,
        out_w,
        conv,
        plan,
        bn,
        false
    );
}

Tensor conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
    const QuantTensorI8& xq,
    int out_h,
    int out_w,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace&
) {
    return conv3x3_q8_ocblock4_s2_bn_avx2_out_hw_impl(
        xq,
        out_h,
        out_w,
        conv,
        plan,
        bn,
        true
    );
}