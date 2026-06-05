#include "conv3x3_pattern_ocblock4_avx2.hpp"

#include "ops.hpp"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

#include <omp.h>

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

void conv3x3_pattern_ocblock4_layer4_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu,
    Tensor& y_out
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

    y_out.reuse(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();

    // Layer4 output is 7x7, so use one masked vector block per output row.
    const int valid_lanes = out_w;
    if (valid_lanes <= 0 || valid_lanes > 8) {
        throw std::runtime_error("Layer4 OC-block4 kernel expected out_w <= 8");
    }

    #pragma omp parallel
    for (int block_idx=0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
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

            store_width_masked_8(y_out.ptr(oc + 0, oh, 0), acc0, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 1, oh, 0), acc1, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 2, oh, 0), acc2, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 3, oh, 0), acc3, valid_lanes);
        }
    }

}

void conv3x3_pattern_ocblock4_layer4_s2_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu,
    Tensor& y_out
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

    y_out.reuse(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();
    constexpr int valid_lanes = 7;

    #pragma omp parallel for schedule(static)
    for (int block_idx=0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        // for (const OCBlock4PatternPlan& block : plan.oc_blocks) { --- IGNORE ---
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
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

            store_width_masked_8(y_out.ptr(oc + 0, oh, 0), acc0, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 1, oh, 0), acc1, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 2, oh, 0), acc2, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 3, oh, 0), acc3, valid_lanes);
        }
    }

}

// OC4+OH2 kernel: runtime per-weight conditionals to avoid 81-case template dispatch.
// Processes 4 output channels and 2 output rows per tile → 8 accumulators.
// Branch prediction is perfect: w0..w3 are constant within each bucket.
void conv3x3_pattern_ocblock4_s1_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu,
    Tensor& y_out
) {
    const Tensor* input_ptr = &x;

    if (conv.pad_h != 0 || conv.pad_w != 0) {
        pad_tensor_into(x, conv.pad_h, conv.pad_w, workspace.pad_buffer);
        input_ptr = &workspace.pad_buffer;
    }

    const Tensor& xin = *input_ptr;

    const int out_h = (x.h + 2 * conv.pad_h - 3) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - 3) / conv.stride_w + 1;

    y_out.reuse(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();
    const float* xdata = xin.data.data();
    const int xin_h = xin.h;
    const int xin_w = xin.w;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 0)]);
        const __m256 scale1 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 1)]);
        const __m256 scale2 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 2)]);
        const __m256 scale3 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 3)]);
        const __m256 bias0  = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 0)]);
        const __m256 bias1  = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 1)]);
        const __m256 bias2  = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 2)]);
        const __m256 bias3  = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 3)]);

        // Flat accumulation: single contiguous loop over flat_actions.
        // Actions are sorted by (w0,w1,w2,w3) pattern, so branches within each
        // pattern-run are perfectly predicted; only ~80 transitions cause mispredictions.
        // No per-bucket pointer chasing — all action data is in one cache-friendly block.
        auto accumulate = [&](
            int oh0, int oh1, int ow_start, int valid_lanes,
            __m256& a0r0, __m256& a0r1,
            __m256& a1r0, __m256& a1r1,
            __m256& a2r0, __m256& a2r1,
            __m256& a3r0, __m256& a3r1
        ) {
            for (const OCBlock4FlatAction& a : block.flat_actions) {
                const int ih0 = oh0 + static_cast<int>(a.ky);
                const int ih1 = oh1 + static_cast<int>(a.ky);
                const int iw  = ow_start + static_cast<int>(a.kx);
                const int b0  = (static_cast<int>(a.ic) * xin_h + ih0) * xin_w + iw;
                const int b1  = (static_cast<int>(a.ic) * xin_h + ih1) * xin_w + iw;

                __m256 x0, x1;
                if (valid_lanes == 8) {
                    x0 = _mm256_loadu_ps(xdata + b0);
                    x1 = _mm256_loadu_ps(xdata + b1);
                } else {
                    x0 = load_width_masked_stride1_8(xdata + b0, valid_lanes);
                    x1 = load_width_masked_stride1_8(xdata + b1, valid_lanes);
                }

                if (a.w0 == 1) { a0r0 = _mm256_add_ps(a0r0, x0); a0r1 = _mm256_add_ps(a0r1, x1); }
                else if (a.w0 == -1) { a0r0 = _mm256_sub_ps(a0r0, x0); a0r1 = _mm256_sub_ps(a0r1, x1); }

                if (a.w1 == 1) { a1r0 = _mm256_add_ps(a1r0, x0); a1r1 = _mm256_add_ps(a1r1, x1); }
                else if (a.w1 == -1) { a1r0 = _mm256_sub_ps(a1r0, x0); a1r1 = _mm256_sub_ps(a1r1, x1); }

                if (a.w2 == 1) { a2r0 = _mm256_add_ps(a2r0, x0); a2r1 = _mm256_add_ps(a2r1, x1); }
                else if (a.w2 == -1) { a2r0 = _mm256_sub_ps(a2r0, x0); a2r1 = _mm256_sub_ps(a2r1, x1); }

                if (a.w3 == 1) { a3r0 = _mm256_add_ps(a3r0, x0); a3r1 = _mm256_add_ps(a3r1, x1); }
                else if (a.w3 == -1) { a3r0 = _mm256_sub_ps(a3r0, x0); a3r1 = _mm256_sub_ps(a3r1, x1); }
            }
        };

        auto bn_relu_store2 = [&](
            int oh0, int oh1, int ow_start, int valid_lanes,
            __m256 a0r0, __m256 a0r1,
            __m256 a1r0, __m256 a1r1,
            __m256 a2r0, __m256 a2r1,
            __m256 a3r0, __m256 a3r1
        ) {
            a0r0 = _mm256_add_ps(_mm256_mul_ps(a0r0, scale0), bias0);
            a0r1 = _mm256_add_ps(_mm256_mul_ps(a0r1, scale0), bias0);
            a1r0 = _mm256_add_ps(_mm256_mul_ps(a1r0, scale1), bias1);
            a1r1 = _mm256_add_ps(_mm256_mul_ps(a1r1, scale1), bias1);
            a2r0 = _mm256_add_ps(_mm256_mul_ps(a2r0, scale2), bias2);
            a2r1 = _mm256_add_ps(_mm256_mul_ps(a2r1, scale2), bias2);
            a3r0 = _mm256_add_ps(_mm256_mul_ps(a3r0, scale3), bias3);
            a3r1 = _mm256_add_ps(_mm256_mul_ps(a3r1, scale3), bias3);

            if (apply_relu) {
                a0r0 = _mm256_max_ps(a0r0, zero_v); a0r1 = _mm256_max_ps(a0r1, zero_v);
                a1r0 = _mm256_max_ps(a1r0, zero_v); a1r1 = _mm256_max_ps(a1r1, zero_v);
                a2r0 = _mm256_max_ps(a2r0, zero_v); a2r1 = _mm256_max_ps(a2r1, zero_v);
                a3r0 = _mm256_max_ps(a3r0, zero_v); a3r1 = _mm256_max_ps(a3r1, zero_v);
            }

            if (valid_lanes == 8) {
                _mm256_storeu_ps(y_out.ptr(oc + 0, oh0, ow_start), a0r0);
                _mm256_storeu_ps(y_out.ptr(oc + 0, oh1, ow_start), a0r1);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh0, ow_start), a1r0);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh1, ow_start), a1r1);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh0, ow_start), a2r0);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh1, ow_start), a2r1);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh0, ow_start), a3r0);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh1, ow_start), a3r1);
            } else {
                store_width_masked_8(y_out.ptr(oc + 0, oh0, ow_start), a0r0, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 0, oh1, ow_start), a0r1, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh0, ow_start), a1r0, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh1, ow_start), a1r1, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh0, ow_start), a2r0, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh1, ow_start), a2r1, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh0, ow_start), a3r0, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh1, ow_start), a3r1, valid_lanes);
            }
        };

        int oh = 0;

        // OH_BLOCK=2: process 2 output rows per iteration
        for (; oh + 1 < out_h; oh += 2) {
            const int ih0_base = oh;
            const int ih1_base = oh + 1;
            int ow = 0;

            for (; ow + 7 < out_w; ow += 8) {
                __m256 a0r0=zero_v, a0r1=zero_v, a1r0=zero_v, a1r1=zero_v;
                __m256 a2r0=zero_v, a2r1=zero_v, a3r0=zero_v, a3r1=zero_v;
                accumulate(ih0_base, ih1_base, ow, 8,
                           a0r0, a0r1, a1r0, a1r1, a2r0, a2r1, a3r0, a3r1);
                bn_relu_store2(oh, oh + 1, ow, 8,
                               a0r0, a0r1, a1r0, a1r1, a2r0, a2r1, a3r0, a3r1);
            }

            if (ow < out_w) {
                const int valid = out_w - ow;
                __m256 a0r0=zero_v, a0r1=zero_v, a1r0=zero_v, a1r1=zero_v;
                __m256 a2r0=zero_v, a2r1=zero_v, a3r0=zero_v, a3r1=zero_v;
                accumulate(ih0_base, ih1_base, ow, valid,
                           a0r0, a0r1, a1r0, a1r1, a2r0, a2r1, a3r0, a3r1);
                bn_relu_store2(oh, oh + 1, ow, valid,
                               a0r0, a0r1, a1r0, a1r1, a2r0, a2r1, a3r0, a3r1);
            }
        }

        // Scalar fallback for odd last row (rare: ResNet-18 spatial dims are all even)
        for (; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                for (const OCBlock4FlatAction& a : block.flat_actions) {
                    const int ih = oh + static_cast<int>(a.ky);
                    const int iw = ow + static_cast<int>(a.kx);
                    const float xv = xdata[(static_cast<int>(a.ic) * xin_h + ih) * xin_w + iw];
                    if (a.w0 ==  1) acc0 += xv; else if (a.w0 == -1) acc0 -= xv;
                    if (a.w1 ==  1) acc1 += xv; else if (a.w1 == -1) acc1 -= xv;
                    if (a.w2 ==  1) acc2 += xv; else if (a.w2 == -1) acc2 -= xv;
                    if (a.w3 ==  1) acc3 += xv; else if (a.w3 == -1) acc3 -= xv;
                }
                acc0 = acc0 * bn.scale[static_cast<size_t>(oc+0)] + bn.bias[static_cast<size_t>(oc+0)];
                acc1 = acc1 * bn.scale[static_cast<size_t>(oc+1)] + bn.bias[static_cast<size_t>(oc+1)];
                acc2 = acc2 * bn.scale[static_cast<size_t>(oc+2)] + bn.bias[static_cast<size_t>(oc+2)];
                acc3 = acc3 * bn.scale[static_cast<size_t>(oc+3)] + bn.bias[static_cast<size_t>(oc+3)];
                if (apply_relu) {
                    acc0 = std::max(acc0, 0.f); acc1 = std::max(acc1, 0.f);
                    acc2 = std::max(acc2, 0.f); acc3 = std::max(acc3, 0.f);
                }
                y_out(oc+0, oh, ow) = acc0; y_out(oc+1, oh, ow) = acc1;
                y_out(oc+2, oh, ow) = acc2; y_out(oc+3, oh, ow) = acc3;
            }
        }
    }

}

} // namespace

bool can_use_pattern_ocblock4_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 4 == 0;
}

void conv3x3_pattern_ocblock4_s1_bn_avx2(
    const Tensor& x, const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv3x3_pattern_ocblock4_s1_bn_avx2_impl(x, conv, plan, bn, workspace, false, y_out);
}

void conv3x3_pattern_ocblock4_s1_bn_relu_avx2(
    const Tensor& x, const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv3x3_pattern_ocblock4_s1_bn_avx2_impl(x, conv, plan, bn, workspace, true, y_out);
}

void conv3x3_pattern_ocblock4_layer4_bn_avx2(
    const Tensor& x, const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv3x3_pattern_ocblock4_layer4_bn_avx2_impl(x, conv, plan, bn, workspace, false, y_out);
}

void conv3x3_pattern_ocblock4_layer4_bn_relu_avx2(
    const Tensor& x, const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv3x3_pattern_ocblock4_layer4_bn_avx2_impl(x, conv, plan, bn, workspace, true, y_out);
}

void conv3x3_pattern_ocblock4_layer4_s2_bn_avx2(
    const Tensor& x, const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv3x3_pattern_ocblock4_layer4_s2_bn_avx2_impl(x, conv, plan, bn, workspace, false, y_out);
}

void conv3x3_pattern_ocblock4_layer4_s2_bn_relu_avx2(
    const Tensor& x, const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv3x3_pattern_ocblock4_layer4_s2_bn_avx2_impl(x, conv, plan, bn, workspace, true, y_out);
}