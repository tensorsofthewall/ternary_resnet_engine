#include "conv3x3_sparse_ocblock_avx2.hpp"

#include "ops.hpp"

#include <immintrin.h>
#include <stdexcept>
#include <algorithm>

bool can_use_sparse_ocblock4_3x3s1_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 && conv.kw == 3 && conv.stride_h == 1 && conv.stride_w == 1 && conv.dilation_h == 1 && conv.dilation_w == 1 && conv.in_channels>=256 && conv.out_channels >=256 && conv.out_channels % 4 == 0;
}

namespace {
    inline void apply_ternary_weight(__m256& acc, const __m256& x, int8_t w) {
        if (w == 1) {
            acc = _mm256_add_ps(acc, x);
        } else if (w == -1) {
            acc = _mm256_sub_ps(acc, x);
        }
    }

    void conv3x3s1_sparse_ocblock4_bn_avx2_impl(const Tensor& x, const Conv2DParams& conv, const SparseOCBlock4ConvPlan& plan, const BNFusedParams& bn, EngineWorkspace& workspace, bool apply_relu, Tensor& y_out) {
        if (!can_use_sparse_ocblock4_3x3s1_avx2(conv)) {
            throw std::runtime_error("Sparse OC-Block4 AVX2 kernel called on unsupported layer: " + conv.name);
        }

        const Tensor* input_ptr = &x;

        if (conv.pad_h != 0 || conv.pad_w != 0) {
            pad_tensor_into(
                x,
                conv.pad_h,
                conv.pad_w,
                workspace.pad_buffer
            );
            input_ptr = &workspace.pad_buffer;
        }

        const Tensor& xin = *input_ptr;
        const int out_h = x.h + 2 * conv.pad_h - 3 + 1;
        const int out_w = x.w + 2 * conv.pad_w - 3 + 1;

        y_out.reuse(conv.out_channels, out_h, out_w);

        const __m256 zero_v = _mm256_setzero_ps();

        for (const OCBlock4Plan& block : plan.oc_blocks) {
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
                int ow = 0;

                for (; ow + 7 < out_w; ow += 8) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (const OCBlock4Action& action : block.actions) {
                        const int ih = oh + static_cast<int>(action.ky);
                        const int iw = ow + static_cast<int>(action.kx);

                        const int base =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        const float* p =
                            xin.data.data() + static_cast<size_t>(base);

                        const __m256 xvec = _mm256_loadu_ps(p);

                        apply_ternary_weight(acc0, xvec, action.w0);
                        apply_ternary_weight(acc1, xvec, action.w1);
                        apply_ternary_weight(acc2, xvec, action.w2);
                        apply_ternary_weight(acc3, xvec, action.w3);
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

                    _mm256_storeu_ps(y_out.ptr(oc + 0, oh, ow), acc0);
                    _mm256_storeu_ps(y_out.ptr(oc + 1, oh, ow), acc1);
                    _mm256_storeu_ps(y_out.ptr(oc + 2, oh, ow), acc2);
                    _mm256_storeu_ps(y_out.ptr(oc + 3, oh, ow), acc3);
                }

                for (; ow < out_w; ++ow) {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;

                    for (const OCBlock4Action& action : block.actions) {
                        const int ih = oh + static_cast<int>(action.ky);
                        const int iw = ow + static_cast<int>(action.kx);

                        const int idx =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        const float xv = xin.data[static_cast<size_t>(idx)];

                        if (action.w0 == 1) {
                            acc0 += xv;
                        } else if (action.w0 == -1) {
                            acc0 -= xv;
                        }

                        if (action.w1 == 1) {
                            acc1 += xv;
                        } else if (action.w1 == -1) {
                            acc1 -= xv;
                        }

                        if (action.w2 == 1) {
                            acc2 += xv;
                        } else if (action.w2 == -1) {
                            acc2 -= xv;
                        }

                        if (action.w3 == 1) {
                            acc3 += xv;
                        } else if (action.w3 == -1) {
                            acc3 -= xv;
                        }
                    }

                    acc0 = acc0 * bn.scale[static_cast<size_t>(oc + 0)] +
                        bn.bias[static_cast<size_t>(oc + 0)];
                    acc1 = acc1 * bn.scale[static_cast<size_t>(oc + 1)] +
                        bn.bias[static_cast<size_t>(oc + 1)];
                    acc2 = acc2 * bn.scale[static_cast<size_t>(oc + 2)] +
                        bn.bias[static_cast<size_t>(oc + 2)];
                    acc3 = acc3 * bn.scale[static_cast<size_t>(oc + 3)] +
                        bn.bias[static_cast<size_t>(oc + 3)];

                    if (apply_relu) {
                        acc0 = std::max(acc0, 0.0f);
                        acc1 = std::max(acc1, 0.0f);
                        acc2 = std::max(acc2, 0.0f);
                        acc3 = std::max(acc3, 0.0f);
                    }

                    y_out(oc + 0, oh, ow) = acc0;
                    y_out(oc + 1, oh, ow) = acc1;
                    y_out(oc + 2, oh, ow) = acc2;
                    y_out(oc + 3, oh, ow) = acc3;
                }
            }
        }
    }
}

void conv3x3s1_sparse_ocblock4_bn_avx2(const Tensor& x, const Conv2DParams& conv, const SparseOCBlock4ConvPlan& plan, const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out) {
    conv3x3s1_sparse_ocblock4_bn_avx2_impl(x, conv, plan, bn, workspace, false, y_out);
}

void conv3x3s1_sparse_ocblock4_bn_relu_avx2(const Tensor& x, const Conv2DParams& conv, const SparseOCBlock4ConvPlan& plan, const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out) {
    conv3x3s1_sparse_ocblock4_bn_avx2_impl(x, conv, plan, bn, workspace, true, y_out);
}
