#include "conv_avx2.hpp"
#include "ops.hpp"

#include <immintrin.h>
#include <stdexcept>

bool can_use_avx2_stride1(const Conv2DParams& conv) {
    return conv.stride_h == 1 && conv.stride_w == 1 && conv.dilation_h == 1 && conv.dilation_w == 1;
}

namespace {
    Tensor conv2d_ternary_sparse_padded_bn_avx2_stride1_impl(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, bool apply_relu) {
        if (!can_use_avx2_stride1(conv)) {
            throw std::runtime_error("AVX2 stride-1 conv called on unsupported layer: " + conv.name);
        }
        Tensor xpad;
        const Tensor* input_ptr = &x;

        if (conv.pad_h != 0 || conv.pad_w != 0) {
            xpad = pad_tensor(x, conv.pad_h, conv.pad_w);
            input_ptr = &xpad;
        }

        const Tensor& xin = *input_ptr;
        const int out_h = x.h + 2 * conv.pad_h - conv.kh + 1;
        const int out_w = x.w + 2 * conv.pad_w - conv.kw + 1;

        Tensor y(conv.out_channels, out_h, out_w);

        const __m256 zero_v = _mm256_setzero_ps();
        
        for (int oc = 0; oc < conv.out_channels; ++oc) {
            const ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

            const __m256 scale_v = _mm256_set1_ps(bn.scale[static_cast<size_t>(oc)]);
            const __m256 bias_v = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc)]);

            for (int oh=0; oh < out_h; ++oh) {
                int ow = 0;
                for (; ow + 7 < out_w; ow += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (const WeightIndex& wi : oc_plan.pos) {
                        const float* p = xin.ptr(
                            static_cast<int>(wi.ic),
                            oh + static_cast<int>(wi.ky),
                            ow + static_cast<int>(wi.kx)
                        );
                        const __m256 xv = _mm256_loadu_ps(p);
                        acc = _mm256_add_ps(acc, xv);
                    }

                    for (const WeightIndex& wi : oc_plan.neg) {
                        const float* p = xin.ptr(
                            static_cast<int>(wi.ic),
                            oh + static_cast<int>(wi.ky),
                            ow + static_cast<int>(wi.kx)
                        );
                        const __m256 xv = _mm256_loadu_ps(p);
                        acc = _mm256_sub_ps(acc, xv);
                    }

                    // acc = _mm256_add_ps(_mm256_mul_ps(acc, scale_v), bias_v);
                    acc = _mm256_fmadd_ps(acc, scale_v, bias_v); // fused multiply add - acc*scale + bias 

                    if (apply_relu) {
                        acc = _mm256_max_ps(acc, zero_v);
                    }
                    _mm256_storeu_ps(y.ptr(oc,oh,ow), acc);
                }

                //scalar tail
                for (; ow < out_w; ++ow) {
                    float acc = 0.0f;
                    for (const WeightIndex& wi : oc_plan.pos) {
                        acc += xin(static_cast<int>(wi.ic), oh + static_cast<int>(wi.ky), ow + static_cast<int>(wi.kx));
                    }
                    for (const WeightIndex& wi : oc_plan.neg) {
                        acc -= xin(static_cast<int>(wi.ic), oh + static_cast<int>(wi.ky), ow + static_cast<int>(wi.kx));
                    }

                    acc = acc*bn.scale[static_cast<size_t>(oc)] + bn.bias[static_cast<size_t>(oc)];

                    if (apply_relu && acc < 0.0f) {
                        acc = 0.0f;
                    }
                    y(oc, oh, ow) = acc;
                }
            }
        }
        return y;
    }
}

Tensor conv2d_ternary_sparse_padded_bn_avx2_stride1(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn) {
    return conv2d_ternary_sparse_padded_bn_avx2_stride1_impl(x,conv,plan,bn,false);
}

Tensor conv2d_ternary_sparse_padded_bn_relu_avx2_stride1(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn) {
    return conv2d_ternary_sparse_padded_bn_avx2_stride1_impl(x,conv,plan,bn,true);
}