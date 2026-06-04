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

inline void store_width_masked_8(
    float* base,
    __m256 value,
    int valid_lanes
) {
    const __m256i mask = make_tail_mask(valid_lanes);
    _mm256_maskstore_ps(base, mask, value);
}

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
                        (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                    const __m256 xvec = load_width_masked_stride1_8(
                        xin.data.data() + static_cast<size_t>(base),
                        valid_lanes
                    );

                    apply_weight(acc0, xvec, w0);
                    apply_weight(acc1, xvec, w1);
                    apply_weight(acc2, xvec, w2);
                    apply_weight(acc3, xvec, w3);
                }
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