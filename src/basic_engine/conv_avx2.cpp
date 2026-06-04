#include "conv_avx2.hpp"
#include "ops.hpp"

#include <immintrin.h>
#include <stdexcept>

bool can_use_avx2_width_kernel(const Conv2DParams& conv) {
    if (conv.dilation_h != 1 || conv.dilation_w != 1) {
        return false;
    }
    if (conv.stride_h != conv.stride_w) {
        return false;
    }

    return conv.stride_h == 1 || conv.stride_h == 2;
}

namespace {
    static inline __m256 load_width_stride1_8(const float* base) {
        return _mm256_loadu_ps(base);
    }

    static inline __m256 load_width_stride2_8(const float* base) {
        return _mm256_set_ps(
            base[14], base[12], base[10], base[8],
            base[6], base[4], base[2], base[0]
        );
    }

    static inline __m256 load_width_8(const float* base, int stride_w) {
        if (stride_w == 1) {
            return load_width_stride1_8(base);
        } 
        if (stride_w == 2) {
            return load_width_stride2_8(base);
        }
        throw std::runtime_error("Unsupported stride_w in load_width_8: " + std::to_string(stride_w));
    }

    Tensor conv2d_ternary_sparse_padded_bn_avx2_width_impl(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, bool apply_relu) {
        if (!can_use_avx2_width_kernel(conv)) {
            throw std::runtime_error("AVX2 width-kernel conv called on unsupported layer: " + conv.name);
        }
        Tensor xpad;
        const Tensor* input_ptr = &x;

        if (conv.pad_h != 0 || conv.pad_w != 0) {
            xpad = pad_tensor(x, conv.pad_h, conv.pad_w);
            input_ptr = &xpad;
        }

        const Tensor& xin = *input_ptr;
        const int out_h = (x.h + 2 * conv.pad_h - conv.kh)/conv.stride_h + 1;
        const int out_w = (x.w + 2 * conv.pad_w - conv.kw)/conv.stride_w + 1;

        Tensor y(conv.out_channels, out_h, out_w);

        const __m256 zero_v = _mm256_setzero_ps();
        
        for (int oc = 0; oc < conv.out_channels; ++oc) {
            const ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

            const float scale = bn.scale[static_cast<size_t>(oc)];
            const float bias = bn.bias[static_cast<size_t>(oc)];
            const __m256 scale_v = _mm256_set1_ps(scale);
            const __m256 bias_v = _mm256_set1_ps(bias);

            for (int oh=0; oh < out_h; ++oh) {
                int ow = 0;
                for (; ow + 7 < out_w; ow += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (const WeightIndex& wi : oc_plan.pos) {
                        const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                        const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);
                        const float* p = xin.ptr(static_cast<int>(wi.ic), ih, iw);
                        const __m256 xv = load_width_8(p, conv.stride_w);
                        acc = _mm256_add_ps(acc, xv);
                    }
                    for (const WeightIndex& wi: oc_plan.neg) {
                        const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                        const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);
                        const float* p = xin.ptr(static_cast<int>(wi.ic), ih, iw);
                        const __m256 xv = load_width_8(p, conv.stride_w);
                        acc = _mm256_sub_ps(acc, xv);
                    }
                    acc = _mm256_fmadd_ps(acc, scale_v, bias_v); // acc*scale + bias
                    if (apply_relu) {
                        acc = _mm256_max_ps(acc, zero_v);
                    }
                    _mm256_storeu_ps(y.ptr(oc,oh,ow), acc);
                }
                for (; ow < out_w; ++ow) {
                    float acc = 0.0f;
                    for (const WeightIndex& wi : oc_plan.pos) {
                        const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                        const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);
                        acc += xin(static_cast<int>(wi.ic), ih, iw);
                    }
                    for (const WeightIndex& wi : oc_plan.neg) {
                        const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                        const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);
                        acc -= xin(static_cast<int>(wi.ic), ih, iw);
                    }
                    acc = acc*scale + bias;
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

Tensor conv2d_ternary_sparse_padded_bn_avx2_width(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn) {
    return conv2d_ternary_sparse_padded_bn_avx2_width_impl(x,conv,plan,bn,false);
}

Tensor conv2d_ternary_sparse_padded_bn_relu_avx2_width(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn) {
    return conv2d_ternary_sparse_padded_bn_avx2_width_impl(x,conv,plan,bn,true);
}