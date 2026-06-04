#include "conv3x3_pattern_ocblock4_avx2.hpp"

#include "ops.hpp"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

bool can_use_pattern_ocblock4_layer4_3x3_avx2(const Conv2DParams& conv) {
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

bool can_use_pattern_ocblock4_layer4_s2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 256 &&
           conv.out_channels == 512 &&
           conv.out_channels % 4 == 0;
}

namespace {

inline __m256i make_tail_mask(int valid_lanes) {
    alignas(32) int mask_values[8];

    for (int i = 0; i < 8; ++i) {
        mask_values[i] = (i < valid_lanes) ? -1 : 0;
    }

    return _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(mask_values)
    );
}

inline __m256 load_width_masked_stride1_8(
    const float* base,
    int valid_lanes
) {
    const __m256i mask = make_tail_mask(valid_lanes);
    return _mm256_maskload_ps(base, mask);
}

inline __m256 load_width7_stride2_8(const float* base) {
    return _mm256_set_ps(
        0.0f,
        base[12],
        base[10],
        base[8],
        base[6],
        base[4],
        base[2],
        base[0]
    );
}

inline void store_width_masked_8(
    float* base,
    __m256 value,
    int valid_lanes
) {
    const __m256i mask = make_tail_mask(valid_lanes);
    _mm256_maskstore_ps(base, mask, value);
}

template <int W0, int W1, int W2, int W3, typename LoadFn>
inline void run_ocblock4_bucket_fixed(
    const OCBlock4PatternBucket& bucket,
    int oh,
    int stride_h,
    int xin_h,
    int xin_w,
    const float* xdata,
    LoadFn&& load_fn,
    __m256& acc0,
    __m256& acc1,
    __m256& acc2,
    __m256& acc3
) {
    for (const OCBlock4PatternAction& action : bucket.actions) {
        const int ih =
            oh * stride_h + static_cast<int>(action.ky);

        const int iw =
            static_cast<int>(action.kx);

        const int base =
            (static_cast<int>(action.ic) * xin_h + ih) * xin_w + iw;

        const __m256 xvec =
            load_fn(xdata + static_cast<size_t>(base));

        if constexpr (W0 == 1) {
            acc0 = _mm256_add_ps(acc0, xvec);
        } else if constexpr (W0 == -1) {
            acc0 = _mm256_sub_ps(acc0, xvec);
        }

        if constexpr (W1 == 1) {
            acc1 = _mm256_add_ps(acc1, xvec);
        } else if constexpr (W1 == -1) {
            acc1 = _mm256_sub_ps(acc1, xvec);
        }

        if constexpr (W2 == 1) {
            acc2 = _mm256_add_ps(acc2, xvec);
        } else if constexpr (W2 == -1) {
            acc2 = _mm256_sub_ps(acc2, xvec);
        }

        if constexpr (W3 == 1) {
            acc3 = _mm256_add_ps(acc3, xvec);
        } else if constexpr (W3 == -1) {
            acc3 = _mm256_sub_ps(acc3, xvec);
        }
    }
}

#define RUN_OCB4_CASE(ID, W0, W1, W2, W3)                            \
    case ID:                                                         \
        return run_ocblock4_bucket_fixed<W0, W1, W2, W3>(             \
            bucket,                                                   \
            oh,                                                       \
            stride_h,                                                 \
            xin_h,                                                    \
            xin_w,                                                    \
            xdata,                                                    \
            load_fn,                                                  \
            acc0,                                                     \
            acc1,                                                     \
            acc2,                                                     \
            acc3                                                      \
        )

template <typename LoadFn>
inline void run_ocblock4_bucket_dispatch(
    const OCBlock4PatternBucket& bucket,
    int oh,
    int stride_h,
    int xin_h,
    int xin_w,
    const float* xdata,
    LoadFn&& load_fn,
    __m256& acc0,
    __m256& acc1,
    __m256& acc2,
    __m256& acc3
) {
    switch (bucket.pattern_id) {
        RUN_OCB4_CASE(0,  -1, -1, -1, -1);
        RUN_OCB4_CASE(1,  -1, -1, -1,  0);
        RUN_OCB4_CASE(2,  -1, -1, -1,  1);
        RUN_OCB4_CASE(3,  -1, -1,  0, -1);
        RUN_OCB4_CASE(4,  -1, -1,  0,  0);
        RUN_OCB4_CASE(5,  -1, -1,  0,  1);
        RUN_OCB4_CASE(6,  -1, -1,  1, -1);
        RUN_OCB4_CASE(7,  -1, -1,  1,  0);
        RUN_OCB4_CASE(8,  -1, -1,  1,  1);

        RUN_OCB4_CASE(9,  -1,  0, -1, -1);
        RUN_OCB4_CASE(10, -1,  0, -1,  0);
        RUN_OCB4_CASE(11, -1,  0, -1,  1);
        RUN_OCB4_CASE(12, -1,  0,  0, -1);
        RUN_OCB4_CASE(13, -1,  0,  0,  0);
        RUN_OCB4_CASE(14, -1,  0,  0,  1);
        RUN_OCB4_CASE(15, -1,  0,  1, -1);
        RUN_OCB4_CASE(16, -1,  0,  1,  0);
        RUN_OCB4_CASE(17, -1,  0,  1,  1);

        RUN_OCB4_CASE(18, -1,  1, -1, -1);
        RUN_OCB4_CASE(19, -1,  1, -1,  0);
        RUN_OCB4_CASE(20, -1,  1, -1,  1);
        RUN_OCB4_CASE(21, -1,  1,  0, -1);
        RUN_OCB4_CASE(22, -1,  1,  0,  0);
        RUN_OCB4_CASE(23, -1,  1,  0,  1);
        RUN_OCB4_CASE(24, -1,  1,  1, -1);
        RUN_OCB4_CASE(25, -1,  1,  1,  0);
        RUN_OCB4_CASE(26, -1,  1,  1,  1);

        RUN_OCB4_CASE(27,  0, -1, -1, -1);
        RUN_OCB4_CASE(28,  0, -1, -1,  0);
        RUN_OCB4_CASE(29,  0, -1, -1,  1);
        RUN_OCB4_CASE(30,  0, -1,  0, -1);
        RUN_OCB4_CASE(31,  0, -1,  0,  0);
        RUN_OCB4_CASE(32,  0, -1,  0,  1);
        RUN_OCB4_CASE(33,  0, -1,  1, -1);
        RUN_OCB4_CASE(34,  0, -1,  1,  0);
        RUN_OCB4_CASE(35,  0, -1,  1,  1);

        RUN_OCB4_CASE(36,  0,  0, -1, -1);
        RUN_OCB4_CASE(37,  0,  0, -1,  0);
        RUN_OCB4_CASE(38,  0,  0, -1,  1);
        RUN_OCB4_CASE(39,  0,  0,  0, -1);
        // 40 is all-zero: (0,0,0,0). It should not exist in the plan.
        case 40:
            return;
        RUN_OCB4_CASE(41,  0,  0,  0,  1);
        RUN_OCB4_CASE(42,  0,  0,  1, -1);
        RUN_OCB4_CASE(43,  0,  0,  1,  0);
        RUN_OCB4_CASE(44,  0,  0,  1,  1);

        RUN_OCB4_CASE(45,  0,  1, -1, -1);
        RUN_OCB4_CASE(46,  0,  1, -1,  0);
        RUN_OCB4_CASE(47,  0,  1, -1,  1);
        RUN_OCB4_CASE(48,  0,  1,  0, -1);
        RUN_OCB4_CASE(49,  0,  1,  0,  0);
        RUN_OCB4_CASE(50,  0,  1,  0,  1);
        RUN_OCB4_CASE(51,  0,  1,  1, -1);
        RUN_OCB4_CASE(52,  0,  1,  1,  0);
        RUN_OCB4_CASE(53,  0,  1,  1,  1);

        RUN_OCB4_CASE(54,  1, -1, -1, -1);
        RUN_OCB4_CASE(55,  1, -1, -1,  0);
        RUN_OCB4_CASE(56,  1, -1, -1,  1);
        RUN_OCB4_CASE(57,  1, -1,  0, -1);
        RUN_OCB4_CASE(58,  1, -1,  0,  0);
        RUN_OCB4_CASE(59,  1, -1,  0,  1);
        RUN_OCB4_CASE(60,  1, -1,  1, -1);
        RUN_OCB4_CASE(61,  1, -1,  1,  0);
        RUN_OCB4_CASE(62,  1, -1,  1,  1);

        RUN_OCB4_CASE(63,  1,  0, -1, -1);
        RUN_OCB4_CASE(64,  1,  0, -1,  0);
        RUN_OCB4_CASE(65,  1,  0, -1,  1);
        RUN_OCB4_CASE(66,  1,  0,  0, -1);
        RUN_OCB4_CASE(67,  1,  0,  0,  0);
        RUN_OCB4_CASE(68,  1,  0,  0,  1);
        RUN_OCB4_CASE(69,  1,  0,  1, -1);
        RUN_OCB4_CASE(70,  1,  0,  1,  0);
        RUN_OCB4_CASE(71,  1,  0,  1,  1);

        RUN_OCB4_CASE(72,  1,  1, -1, -1);
        RUN_OCB4_CASE(73,  1,  1, -1,  0);
        RUN_OCB4_CASE(74,  1,  1, -1,  1);
        RUN_OCB4_CASE(75,  1,  1,  0, -1);
        RUN_OCB4_CASE(76,  1,  1,  0,  0);
        RUN_OCB4_CASE(77,  1,  1,  0,  1);
        RUN_OCB4_CASE(78,  1,  1,  1, -1);
        RUN_OCB4_CASE(79,  1,  1,  1,  0);
        RUN_OCB4_CASE(80,  1,  1,  1,  1);

        default:
            throw std::runtime_error("Unexpected OC-Block4 pattern id");
    }
}

#undef RUN_OCB4_CASE

inline void apply_weight(
    __m256& acc,
    const __m256& x,
    int8_t w
) {
    if (w == 1) {
        acc = _mm256_add_ps(acc, x);
    } else if (w == -1) {
        acc = _mm256_sub_ps(acc, x);
    }
}

Tensor conv3x3_pattern_ocblock4_layer4_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu
) {
    if (!can_use_pattern_ocblock4_layer4_3x3_avx2(conv)) {
        throw std::runtime_error(
            "Pattern OC-block4 layer4 AVX2 kernel called on unsupported conv"
        );
    }

    const Tensor* input_ptr = &x;

    if (conv.pad_h != 0 || conv.pad_w != 0) {
        pad_tensor_into(x, conv.pad_h, conv.pad_w, workspace.pad_buffer);
        input_ptr = &workspace.pad_buffer;
    }

    const Tensor& xin = *input_ptr;

    const int out_h =
        (x.h + 2 * conv.pad_h - 3) / conv.stride_h + 1;

    const int out_w =
        (x.w + 2 * conv.pad_w - 3) / conv.stride_w + 1;

    Tensor y(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();

    // Layer4 output is 7x7, so use one masked vector block per output row.
    const int valid_lanes = out_w;
    if (valid_lanes <= 0 || valid_lanes > 8) {
        throw std::runtime_error("Layer4 OC-block4 kernel expected out_w <= 8");
    }

    for (const OCBlock4PatternPlan& block : plan.oc_blocks) {
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 0)]);
        const __m256 scale1 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 1)]);
        const __m256 scale2 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 2)]);
        const __m256 scale3 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 3)]);

        const __m256 bias0 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 0)]);
        const __m256 bias1 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 1)]);
        const __m256 bias2 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 2)]);
        const __m256 bias3 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 3)]);

        for (int oh = 0; oh < out_h; ++oh) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            auto load_fn = [&](const float* p) {
                return load_width_masked_stride1_8(p, valid_lanes);
            };

            for (const OCBlock4PatternBucket& bucket : block.buckets) {
                run_ocblock4_bucket_dispatch(
                    bucket,
                    oh,
                    conv.stride_h,
                    xin.h,
                    xin.w,
                    xin.data.data(),
                    load_fn,
                    acc0,
                    acc1,
                    acc2,
                    acc3
                );
            }

            acc0 = _mm256_add_ps(_mm256_mul_ps(acc0, scale0), bias0);
            acc1 = _mm256_add_ps(_mm256_mul_ps(acc1, scale1), bias1);
            acc2 = _mm256_add_ps(_mm256_mul_ps(acc2, scale2), bias2);
            acc3 = _mm256_add_ps(_mm256_mul_ps(acc3, scale3), bias3);

            if (apply_relu) {
                acc0 = _mm256_max_ps(acc0, zero_v);
                acc1 = _mm256_max_ps(acc1, zero_v);
                acc2 = _mm256_max_ps(acc2, zero_v);
                acc3 = _mm256_max_ps(acc3, zero_v);
            }

            store_width_masked_8(y.ptr(oc + 0, oh, 0), acc0, valid_lanes);
            store_width_masked_8(y.ptr(oc + 1, oh, 0), acc1, valid_lanes);
            store_width_masked_8(y.ptr(oc + 2, oh, 0), acc2, valid_lanes);
            store_width_masked_8(y.ptr(oc + 3, oh, 0), acc3, valid_lanes);
        }
    }

    return y;
}

Tensor conv3x3_pattern_ocblock4_layer4_s2_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu
) {
    if (!can_use_pattern_ocblock4_layer4_s2_3x3_avx2(conv)) {
        throw std::runtime_error(
            "Pattern OC-block4 layer4 stride-2 AVX2 kernel called on unsupported conv"
        );
    }

    const Tensor* input_ptr = &x;

    if (conv.pad_h != 0 || conv.pad_w != 0) {
        pad_tensor_into(x, conv.pad_h, conv.pad_w, workspace.pad_buffer);
        input_ptr = &workspace.pad_buffer;
    }

    const Tensor& xin = *input_ptr;

    const int out_h =
        (x.h + 2 * conv.pad_h - 3) / conv.stride_h + 1;

    const int out_w =
        (x.w + 2 * conv.pad_w - 3) / conv.stride_w + 1;

    if (out_w != 7) {
        throw std::runtime_error(
            "Layer4 stride-2 OC-block4 kernel expects out_w == 7"
        );
    }

    Tensor y(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();
    constexpr int valid_lanes = 7;

    for (const OCBlock4PatternPlan& block : plan.oc_blocks) {
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(
            bn.scale[static_cast<size_t>(oc + 0)]
        );
        const __m256 scale1 = _mm256_set1_ps(
            bn.scale[static_cast<size_t>(oc + 1)]
        );
        const __m256 scale2 = _mm256_set1_ps(
            bn.scale[static_cast<size_t>(oc + 2)]
        );
        const __m256 scale3 = _mm256_set1_ps(
            bn.scale[static_cast<size_t>(oc + 3)]
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
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            auto load_fn = [&](const float* p) {
                return load_width7_stride2_8(p);
            };

            for (const OCBlock4PatternBucket& bucket : block.buckets) {
                run_ocblock4_bucket_dispatch(
                    bucket,
                    oh,
                    conv.stride_h,
                    xin.h,
                    xin.w,
                    xin.data.data(),
                    load_fn,
                    acc0,
                    acc1,
                    acc2,
                    acc3
                );
            }

            acc0 = _mm256_add_ps(_mm256_mul_ps(acc0, scale0), bias0);
            acc1 = _mm256_add_ps(_mm256_mul_ps(acc1, scale1), bias1);
            acc2 = _mm256_add_ps(_mm256_mul_ps(acc2, scale2), bias2);
            acc3 = _mm256_add_ps(_mm256_mul_ps(acc3, scale3), bias3);

            if (apply_relu) {
                acc0 = _mm256_max_ps(acc0, zero_v);
                acc1 = _mm256_max_ps(acc1, zero_v);
                acc2 = _mm256_max_ps(acc2, zero_v);
                acc3 = _mm256_max_ps(acc3, zero_v);
            }

            store_width_masked_8(y.ptr(oc + 0, oh, 0), acc0, valid_lanes);
            store_width_masked_8(y.ptr(oc + 1, oh, 0), acc1, valid_lanes);
            store_width_masked_8(y.ptr(oc + 2, oh, 0), acc2, valid_lanes);
            store_width_masked_8(y.ptr(oc + 3, oh, 0), acc3, valid_lanes);
        }
    }

    return y;
}

} // namespace

Tensor conv3x3_pattern_ocblock4_layer4_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3_pattern_ocblock4_layer4_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        false
    );
}

Tensor conv3x3_pattern_ocblock4_layer4_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3_pattern_ocblock4_layer4_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        true
    );
}

Tensor conv3x3_pattern_ocblock4_layer4_s2_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3_pattern_ocblock4_layer4_s2_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        false
    );
}

Tensor conv3x3_pattern_ocblock4_layer4_s2_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3_pattern_ocblock4_layer4_s2_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        true
    );
}