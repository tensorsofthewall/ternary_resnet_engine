#include "conv1_ocblock4_avx2.hpp"

#include "ops.hpp"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

#include <omp.h>

bool can_use_conv1_ocblock4_7x7s2_avx2(const Conv2DParams& conv) {
    return conv.name == "conv1" &&
           conv.in_channels == 3 &&
           conv.out_channels == 64 &&
           conv.kh == 7 &&
           conv.kw == 7 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.pad_h == 3 &&
           conv.pad_w == 3 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 4 == 0;
}

namespace {

inline __m256 load_width_stride2_8(const float* base) {
    return _mm256_set_ps(
        base[14],
        base[12],
        base[10],
        base[8],
        base[6],
        base[4],
        base[2],
        base[0]
    );
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

Tensor conv1_ocblock4_7x7s2_bn_relu_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const Conv1OCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    if (!can_use_conv1_ocblock4_7x7s2_avx2(conv)) {
        throw std::runtime_error(
            "conv1 OC-Block4 AVX2 kernel called on unsupported conv"
        );
    }

    const Tensor* input_ptr = &x;

    if (conv.pad_h != 0 || conv.pad_w != 0) {
        pad_tensor_into(x, conv.pad_h, conv.pad_w, workspace.pad_buffer);
        input_ptr = &workspace.pad_buffer;
    }

    const Tensor& xin = *input_ptr;

    const int out_h =
        (x.h + 2 * conv.pad_h - conv.kh) / conv.stride_h + 1;

    const int out_w =
        (x.w + 2 * conv.pad_w - conv.kw) / conv.stride_w + 1;

    if (out_h != 112 || out_w != 112) {
        throw std::runtime_error(
            "conv1 OC-Block4 AVX2 kernel expected 112x112 output"
        );
    }

    Tensor y(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();

    // #pragma omp parallel for schedule(static)
    // for (int block_idx=0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
    //     const Conv1OCBlock4Plan& block = plan.oc_blocks[block_idx];
    for (const Conv1OCBlock4Plan& block : plan.oc_blocks) {
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
            for (int ow = 0; ow < out_w; ow += 8) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (const Conv1OCBlock4PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;
                    const int8_t w2 = bucket.w2;
                    const int8_t w3 = bucket.w3;

                    for (const Conv1OCBlock4Action& action : bucket.actions) {
                        const int ih =
                            oh * 2 + static_cast<int>(action.ky);

                        const int iw =
                            ow * 2 + static_cast<int>(action.kx);

                        const int base =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        const __m256 xvec = load_width_stride2_8(
                            xin.data.data() + static_cast<size_t>(base)
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

                acc0 = _mm256_max_ps(acc0, zero_v);
                acc1 = _mm256_max_ps(acc1, zero_v);
                acc2 = _mm256_max_ps(acc2, zero_v);
                acc3 = _mm256_max_ps(acc3, zero_v);

                _mm256_storeu_ps(y.ptr(oc + 0, oh, ow), acc0);
                _mm256_storeu_ps(y.ptr(oc + 1, oh, ow), acc1);
                _mm256_storeu_ps(y.ptr(oc + 2, oh, ow), acc2);
                _mm256_storeu_ps(y.ptr(oc + 3, oh, ow), acc3);
            }
        }
    }

    return y;
}

} // namespace

Tensor conv1_ocblock4_7x7s2_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const Conv1OCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv1_ocblock4_7x7s2_bn_relu_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace
    );
}