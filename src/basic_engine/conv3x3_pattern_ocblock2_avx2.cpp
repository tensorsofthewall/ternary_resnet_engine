#include "conv3x3_pattern_ocblock2_avx2.hpp"

#include "ops.hpp"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

bool can_use_pattern_ocblock2_3x3s1_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 2 == 0;
}

namespace {

constexpr int PATTERN_NEG_NEG=0;
constexpr int PATTERN_NEG_ZERO=1;
constexpr int PATTERN_NEG_POS=2;
constexpr int PATTERN_ZERO_NEG=3;
constexpr int PATTERN_ZERO_ZERO=4;
constexpr int PATTERN_ZERO_POS=5;
constexpr int PATTERN_POS_NEG=6;
constexpr int PATTERN_POS_ZERO=7;
constexpr int PATTERN_POS_POS=8;

// inline void apply_pattern(
//     __m256& acc0,
//     __m256& acc1,
//     const __m256& x,
//     int8_t w0,
//     int8_t w1
// ) {
//     if (w0 == 1) {
//         acc0 = _mm256_add_ps(acc0, x);
//     } else if (w0 == -1) {
//         acc0 = _mm256_sub_ps(acc0, x);
//     }

//     if (w1 == 1) {
//         acc1 = _mm256_add_ps(acc1, x);
//     } else if (w1 == -1) {
//         acc1 = _mm256_sub_ps(acc1, x);
//     }
// }

Tensor conv3x3s1_pattern_ocblock2_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu
) {
    if (!can_use_pattern_ocblock2_3x3s1_avx2(conv)) {
        throw std::runtime_error(
            "Pattern OC-block2 AVX2 kernel called on unsupported conv"
        );
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

    Tensor y(conv.out_channels, out_h, out_w);

    const __m256 zero_v = _mm256_setzero_ps();

    for (const OCBlock2PatternPlan& block : plan.oc_blocks) {
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(
            bn.scale[static_cast<size_t>(oc + 0)]
        );
        const __m256 scale1 = _mm256_set1_ps(
            bn.scale[static_cast<size_t>(oc + 1)]
        );

        const __m256 bias0 = _mm256_set1_ps(
            bn.bias[static_cast<size_t>(oc + 0)]
        );
        const __m256 bias1 = _mm256_set1_ps(
            bn.bias[static_cast<size_t>(oc + 1)]
        );

        for (int oh = 0; oh < out_h; ++oh) {
            int ow = 0;

            for (; ow + 7 < out_w; ow += 8) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                // for (const OCBlock2PatternBucket& bucket : block.buckets) {
                //     const int8_t w0 = bucket.w0;
                //     const int8_t w1 = bucket.w1;

                //     for (const OCBlock2Action& action : bucket.actions) {
                //         const int ih = oh + static_cast<int>(action.ky);
                //         const int iw = ow + static_cast<int>(action.kx);

                //         const int base =
                //             (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                //         const float* p =
                //             xin.data.data() + static_cast<size_t>(base);

                //         const __m256 xvec = _mm256_loadu_ps(p);

                //         apply_pattern(acc0, acc1, xvec, w0, w1);
                //     }
                // }

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    switch (bucket.pattern_id) {
                        case PATTERN_NEG_NEG: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc0 = _mm256_sub_ps(acc0, xvec);
                                acc1 = _mm256_sub_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_NEG_ZERO: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc0 = _mm256_sub_ps(acc0, xvec);
                            }
                            break;
                        }

                        case PATTERN_NEG_POS: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc0 = _mm256_sub_ps(acc0, xvec);
                                acc1 = _mm256_add_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_ZERO_NEG: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc1 = _mm256_sub_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_ZERO_POS: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc1 = _mm256_add_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_POS_NEG: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc0 = _mm256_add_ps(acc0, xvec);
                                acc1 = _mm256_sub_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_POS_ZERO: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc0 = _mm256_add_ps(acc0, xvec);
                            }
                            break;
                        }

                        case PATTERN_POS_POS: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh + static_cast<int>(action.ky);
                                const int iw = ow + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = _mm256_loadu_ps(
                                    xin.data.data() + static_cast<size_t>(base)
                                );

                                acc0 = _mm256_add_ps(acc0, xvec);
                                acc1 = _mm256_add_ps(acc1, xvec);
                            }
                            break;
                        }

                        default:
                            throw std::runtime_error("Unexpected OC-block2 pattern id");
                    }
                }

                acc0 = _mm256_add_ps(_mm256_mul_ps(acc0, scale0), bias0);
                acc1 = _mm256_add_ps(_mm256_mul_ps(acc1, scale1), bias1);

                if (apply_relu) {
                    acc0 = _mm256_max_ps(acc0, zero_v);
                    acc1 = _mm256_max_ps(acc1, zero_v);
                }

                _mm256_storeu_ps(y.ptr(oc + 0, oh, ow), acc0);
                _mm256_storeu_ps(y.ptr(oc + 1, oh, ow), acc1);
            }

            for (; ow < out_w; ++ow) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;

                    for (const OCBlock2Action& action : bucket.actions) {
                        const int ih = oh + static_cast<int>(action.ky);
                        const int iw = ow + static_cast<int>(action.kx);

                        const int idx =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        const float xv =
                            xin.data[static_cast<size_t>(idx)];

                        if (w0 == 1) {
                            acc0 += xv;
                        } else if (w0 == -1) {
                            acc0 -= xv;
                        }

                        if (w1 == 1) {
                            acc1 += xv;
                        } else if (w1 == -1) {
                            acc1 -= xv;
                        }
                    }
                }

                acc0 = acc0 * bn.scale[static_cast<size_t>(oc + 0)] +
                       bn.bias[static_cast<size_t>(oc + 0)];
                acc1 = acc1 * bn.scale[static_cast<size_t>(oc + 1)] +
                       bn.bias[static_cast<size_t>(oc + 1)];

                if (apply_relu) {
                    acc0 = std::max(acc0, 0.0f);
                    acc1 = std::max(acc1, 0.0f);
                }

                y(oc + 0, oh, ow) = acc0;
                y(oc + 1, oh, ow) = acc1;
            }
        }
    }

    return y;
}

} // namespace

Tensor conv3x3s1_pattern_ocblock2_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3s1_pattern_ocblock2_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        false
    );
}

Tensor conv3x3s1_pattern_ocblock2_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3s1_pattern_ocblock2_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        true
    );
}