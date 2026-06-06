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

// Stride-2 gather: loads base[0], base[2], ..., base[14] as one AVX2 gather.
// Replaces 8 scalar loads. scale=4 → each index is multiplied by sizeof(float).
inline __m256 load_width_stride2_8(const float* base) {
    return _mm256_i32gather_ps(
        base,
        _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0),
        4
    );
}

void conv1_ocblock4_7x7s2_bn_relu_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const Conv1OCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    Tensor& y_out
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

    y_out.reuse(conv.out_channels, out_h, out_w);

    const float* const data = xin.data.data();
    const __m256 zero_v = _mm256_setzero_ps();

    // #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const Conv1OCBlock4Plan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 0)]);
        const __m256 scale1 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 1)]);
        const __m256 scale2 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 2)]);
        const __m256 scale3 = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc + 3)]);

        const __m256 bias0 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 0)]);
        const __m256 bias1 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 1)]);
        const __m256 bias2 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 2)]);
        const __m256 bias3 = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 3)]);

        // OH_BLOCK=4: process 4 output rows per pass.
        // out_h=112, 112/4=28 exact — no residual needed.
        // out_w=112, 112/8=14 exact — no tail needed.
        for (int oh = 0; oh < out_h; oh += 4) {
            // Input row bases for each of the 4 output rows (stride=2)
            const int ih0 = oh * 2;
            const int ih1 = oh * 2 + 2;
            const int ih2 = oh * 2 + 4;
            const int ih3 = oh * 2 + 6;

            for (int ow = 0; ow < out_w; ow += 8) {
                const int iw = ow * 2;

                // 16 accumulators: acc[row][oc_channel] — fills AVX2's 16 ymm registers
                __m256 acc00 = zero_v, acc01 = zero_v, acc02 = zero_v, acc03 = zero_v;
                __m256 acc10 = zero_v, acc11 = zero_v, acc12 = zero_v, acc13 = zero_v;
                __m256 acc20 = zero_v, acc21 = zero_v, acc22 = zero_v, acc23 = zero_v;
                __m256 acc30 = zero_v, acc31 = zero_v, acc32 = zero_v, acc33 = zero_v;

                for (const Conv1OCBlock4PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;
                    const int8_t w2 = bucket.w2;
                    const int8_t w3 = bucket.w3;

                    for (const Conv1OCBlock4Action& action : bucket.actions) {
                        const int ic_row = static_cast<int>(action.ic) * xin.h;
                        const int ky    = static_cast<int>(action.ky);
                        const int col   = iw + static_cast<int>(action.kx);

                        // 4 independent gather loads — CPU can issue all in parallel
                        const __m256 x0 = load_width_stride2_8(data + (ic_row + ih0 + ky) * xin.w + col);
                        const __m256 x1 = load_width_stride2_8(data + (ic_row + ih1 + ky) * xin.w + col);
                        const __m256 x2 = load_width_stride2_8(data + (ic_row + ih2 + ky) * xin.w + col);
                        const __m256 x3 = load_width_stride2_8(data + (ic_row + ih3 + ky) * xin.w + col);

                        if (w0 == 1) {
                            acc00 = _mm256_add_ps(acc00, x0);
                            acc10 = _mm256_add_ps(acc10, x1);
                            acc20 = _mm256_add_ps(acc20, x2);
                            acc30 = _mm256_add_ps(acc30, x3);
                        } else if (w0 == -1) {
                            acc00 = _mm256_sub_ps(acc00, x0);
                            acc10 = _mm256_sub_ps(acc10, x1);
                            acc20 = _mm256_sub_ps(acc20, x2);
                            acc30 = _mm256_sub_ps(acc30, x3);
                        }

                        if (w1 == 1) {
                            acc01 = _mm256_add_ps(acc01, x0);
                            acc11 = _mm256_add_ps(acc11, x1);
                            acc21 = _mm256_add_ps(acc21, x2);
                            acc31 = _mm256_add_ps(acc31, x3);
                        } else if (w1 == -1) {
                            acc01 = _mm256_sub_ps(acc01, x0);
                            acc11 = _mm256_sub_ps(acc11, x1);
                            acc21 = _mm256_sub_ps(acc21, x2);
                            acc31 = _mm256_sub_ps(acc31, x3);
                        }

                        if (w2 == 1) {
                            acc02 = _mm256_add_ps(acc02, x0);
                            acc12 = _mm256_add_ps(acc12, x1);
                            acc22 = _mm256_add_ps(acc22, x2);
                            acc32 = _mm256_add_ps(acc32, x3);
                        } else if (w2 == -1) {
                            acc02 = _mm256_sub_ps(acc02, x0);
                            acc12 = _mm256_sub_ps(acc12, x1);
                            acc22 = _mm256_sub_ps(acc22, x2);
                            acc32 = _mm256_sub_ps(acc32, x3);
                        }

                        if (w3 == 1) {
                            acc03 = _mm256_add_ps(acc03, x0);
                            acc13 = _mm256_add_ps(acc13, x1);
                            acc23 = _mm256_add_ps(acc23, x2);
                            acc33 = _mm256_add_ps(acc33, x3);
                        } else if (w3 == -1) {
                            acc03 = _mm256_sub_ps(acc03, x0);
                            acc13 = _mm256_sub_ps(acc13, x1);
                            acc23 = _mm256_sub_ps(acc23, x2);
                            acc33 = _mm256_sub_ps(acc33, x3);
                        }
                    }
                }

                // BN fuse + ReLU: y = max(acc * scale + bias, 0)
                acc00 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc00, scale0), bias0), zero_v);
                acc01 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc01, scale1), bias1), zero_v);
                acc02 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc02, scale2), bias2), zero_v);
                acc03 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc03, scale3), bias3), zero_v);

                acc10 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc10, scale0), bias0), zero_v);
                acc11 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc11, scale1), bias1), zero_v);
                acc12 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc12, scale2), bias2), zero_v);
                acc13 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc13, scale3), bias3), zero_v);

                acc20 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc20, scale0), bias0), zero_v);
                acc21 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc21, scale1), bias1), zero_v);
                acc22 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc22, scale2), bias2), zero_v);
                acc23 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc23, scale3), bias3), zero_v);

                acc30 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc30, scale0), bias0), zero_v);
                acc31 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc31, scale1), bias1), zero_v);
                acc32 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc32, scale2), bias2), zero_v);
                acc33 = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(acc33, scale3), bias3), zero_v);

                // 16 stores: 4 rows × 4 OC channels
                _mm256_storeu_ps(y_out.ptr(oc + 0, oh + 0, ow), acc00);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh + 0, ow), acc01);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh + 0, ow), acc02);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh + 0, ow), acc03);

                _mm256_storeu_ps(y_out.ptr(oc + 0, oh + 1, ow), acc10);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh + 1, ow), acc11);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh + 1, ow), acc12);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh + 1, ow), acc13);

                _mm256_storeu_ps(y_out.ptr(oc + 0, oh + 2, ow), acc20);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh + 2, ow), acc21);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh + 2, ow), acc22);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh + 2, ow), acc23);

                _mm256_storeu_ps(y_out.ptr(oc + 0, oh + 3, ow), acc30);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh + 3, ow), acc31);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh + 3, ow), acc32);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh + 3, ow), acc33);
            }
        }
    }
}

} // namespace

void conv1_ocblock4_7x7s2_bn_relu_avx2(
    const Tensor& x, const Conv2DParams& conv, const Conv1OCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out
) {
    conv1_ocblock4_7x7s2_bn_relu_avx2_impl(x, conv, plan, bn, workspace, y_out);
}
