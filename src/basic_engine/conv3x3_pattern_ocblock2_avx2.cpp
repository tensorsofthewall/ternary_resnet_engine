#include "conv3x3_pattern_ocblock2_avx2.hpp"

#include "ops.hpp"

#include <algorithm>
#include <immintrin.h>
#include <stdexcept>

bool can_use_pattern_ocblock2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == conv.stride_w &&
           (conv.stride_h == 1 || conv.stride_h == 2) &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 2 == 0;
}

bool can_use_pattern_ocblock2_spatial2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.stride_h == conv.stride_w &&
           (conv.stride_h == 1 || conv.stride_h == 2) &&
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

inline __m256 load_width_stride1_8(const float* base) {
    return _mm256_loadu_ps(base);
}

inline __m256 load_width_stride2_8(const float* base) {
    return _mm256_set_ps(
        base[14], base[12], base[10], base[8],
        base[6], base[4], base[2], base[0]
    );
}

inline __m256 load_width_8(const float* base, int stride_w) {
    if (stride_w == 1) {
        return load_width_stride1_8(base);
    } 
    if (stride_w == 2) {
        return load_width_stride2_8(base);
    }
    throw std::runtime_error("Unsupported stride_w in pattern OC-block2 kernel: " + std::to_string(stride_w));
}

inline void apply_pattern_spatial2(
    int pattern_id,
    __m256& acc0_r0,
    __m256& acc1_r0,
    __m256& acc0_r1,
    __m256& acc1_r1,
    const __m256& x0,
    const __m256& x1
) {
    switch (pattern_id) {
        case PATTERN_NEG_NEG:
            acc0_r0 = _mm256_sub_ps(acc0_r0, x0);
            acc1_r0 = _mm256_sub_ps(acc1_r0, x0);
            acc0_r1 = _mm256_sub_ps(acc0_r1, x1);
            acc1_r1 = _mm256_sub_ps(acc1_r1, x1);
            break;

        case PATTERN_NEG_ZERO:
            acc0_r0 = _mm256_sub_ps(acc0_r0, x0);
            acc0_r1 = _mm256_sub_ps(acc0_r1, x1);
            break;

        case PATTERN_NEG_POS:
            acc0_r0 = _mm256_sub_ps(acc0_r0, x0);
            acc1_r0 = _mm256_add_ps(acc1_r0, x0);
            acc0_r1 = _mm256_sub_ps(acc0_r1, x1);
            acc1_r1 = _mm256_add_ps(acc1_r1, x1);
            break;

        case PATTERN_ZERO_NEG:
            acc1_r0 = _mm256_sub_ps(acc1_r0, x0);
            acc1_r1 = _mm256_sub_ps(acc1_r1, x1);
            break;

        case PATTERN_ZERO_POS:
            acc1_r0 = _mm256_add_ps(acc1_r0, x0);
            acc1_r1 = _mm256_add_ps(acc1_r1, x1);
            break;

        case PATTERN_POS_NEG:
            acc0_r0 = _mm256_add_ps(acc0_r0, x0);
            acc1_r0 = _mm256_sub_ps(acc1_r0, x0);
            acc0_r1 = _mm256_add_ps(acc0_r1, x1);
            acc1_r1 = _mm256_sub_ps(acc1_r1, x1);
            break;

        case PATTERN_POS_ZERO:
            acc0_r0 = _mm256_add_ps(acc0_r0, x0);
            acc0_r1 = _mm256_add_ps(acc0_r1, x1);
            break;

        case PATTERN_POS_POS:
            acc0_r0 = _mm256_add_ps(acc0_r0, x0);
            acc1_r0 = _mm256_add_ps(acc1_r0, x0);
            acc0_r1 = _mm256_add_ps(acc0_r1, x1);
            acc1_r1 = _mm256_add_ps(acc1_r1, x1);
            break;

        default:
            throw std::runtime_error("Unexpected OC-block2 pattern id");
    }
}

inline __m256i make_tail_mask(int valid_lanes) {
    alignas(32) int mask_values[8];

    for (int i = 0; i < 8; ++i) {
        mask_values[i] = (i < valid_lanes) ? -1 : 0;
    }
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask_values));
}

inline __m256 load_width_masked_stride1_8 (const float* base, int valid_lanes) {
    const __m256i mask = make_tail_mask(valid_lanes);
    return _mm256_maskload_ps(base, mask);
}

inline void store_width_masked_8(float* base, __m256 value, int valid_lanes) {
    const __m256i mask = make_tail_mask(valid_lanes);
    _mm256_maskstore_ps(base, mask, value);
}

inline __m256 load_width_tail_8(const float* base, int stride_w, int valid_lanes) {
    if (stride_w == 1) {
        return load_width_masked_stride1_8(base, valid_lanes);
    }
    throw std::runtime_error("Unsupported stride_w in load_width_8 with masking: " + std::to_string(stride_w));
}

Tensor conv3x3s1_pattern_ocblock2_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu
) {
    if (!can_use_pattern_ocblock2_3x3_avx2(conv)) {
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

    const int out_h = (x.h + 2 * conv.pad_h - 3) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - 3) / conv.stride_w + 1;

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
                                const int ih = oh * conv.stride_h+ static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc0 = _mm256_sub_ps(acc0, xvec);
                                acc1 = _mm256_sub_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_NEG_ZERO: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc0 = _mm256_sub_ps(acc0, xvec);
                            }
                            break;
                        }

                        case PATTERN_NEG_POS: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc0 = _mm256_sub_ps(acc0, xvec);
                                acc1 = _mm256_add_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_ZERO_NEG: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc1 = _mm256_sub_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_ZERO_POS: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc1 = _mm256_add_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_POS_NEG: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc0 = _mm256_add_ps(acc0, xvec);
                                acc1 = _mm256_sub_ps(acc1, xvec);
                            }
                            break;
                        }

                        case PATTERN_POS_ZERO: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
                                );

                                acc0 = _mm256_add_ps(acc0, xvec);
                            }
                            break;
                        }

                        case PATTERN_POS_POS: {
                            for (const OCBlock2Action& action : bucket.actions) {
                                const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                                const int iw = ow * conv.stride_w + static_cast<int>(action.kx);
                                const int base =
                                    (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                                const __m256 xvec = load_width_8(
                                    xin.data.data() + static_cast<size_t>(base),
                                    conv.stride_w
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
                        const int ih = oh * conv.stride_h + static_cast<int>(action.ky);
                        const int iw = ow * conv.stride_w + static_cast<int>(action.kx);

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

Tensor conv3x3_pattern_ocblock2_spatial2_bn_avx2_impl(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace,
    bool apply_relu
) {
    if (!can_use_pattern_ocblock2_spatial2_3x3_avx2(conv)) {
        throw std::runtime_error(
            "Pattern OC-block2 spatial2 AVX2 kernel called on unsupported conv"
        );
    }

    const Tensor* input_ptr = &x;

    if (conv.pad_h != 0 || conv.pad_w != 0) {
        pad_tensor_into(x, conv.pad_h, conv.pad_w, workspace.pad_buffer);
        input_ptr = &workspace.pad_buffer;
    }

    const Tensor& xin = *input_ptr;

    const int out_h = (x.h + 2 * conv.pad_h - 3) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - 3) / conv.stride_w + 1;

    const bool use_masked_tail = conv.stride_w == 1 && out_w == 14;

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

        int oh = 0;

        for (; oh + 1 < out_h; oh += 2) {
            auto compute_store_rowpair_block = [&](int ow_cur, int valid_lanes) {
                __m256 acc0_r0 = _mm256_setzero_ps();
                __m256 acc1_r0 = _mm256_setzero_ps();
                __m256 acc0_r1 = _mm256_setzero_ps();
                __m256 acc1_r1 = _mm256_setzero_ps();

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int ih0 =
                            oh * conv.stride_h + static_cast<int>(action.ky);
                        const int ih1 =
                            (oh + 1) * conv.stride_h + static_cast<int>(action.ky);

                        const int iw =
                            ow_cur * conv.stride_w + static_cast<int>(action.kx);

                        const int base0 =
                            (static_cast<int>(action.ic) * xin.h + ih0) * xin.w + iw;
                        const int base1 =
                            (static_cast<int>(action.ic) * xin.h + ih1) * xin.w + iw;

                        __m256 x0;
                        __m256 x1;

                        if (valid_lanes == 8) {
                            x0 = load_width_8(
                                xin.data.data() + static_cast<size_t>(base0),
                                conv.stride_w
                            );
                            x1 = load_width_8(
                                xin.data.data() + static_cast<size_t>(base1),
                                conv.stride_w
                            );
                        } else {
                            x0 = load_width_tail_8(
                                xin.data.data() + static_cast<size_t>(base0),
                                conv.stride_w,
                                valid_lanes
                            );
                            x1 = load_width_tail_8(
                                xin.data.data() + static_cast<size_t>(base1),
                                conv.stride_w,
                                valid_lanes
                            );
                        }

                        apply_pattern_spatial2(
                            bucket.pattern_id,
                            acc0_r0,
                            acc1_r0,
                            acc0_r1,
                            acc1_r1,
                            x0,
                            x1
                        );
                    }
                }

                acc0_r0 = _mm256_add_ps(_mm256_mul_ps(acc0_r0, scale0), bias0);
                acc1_r0 = _mm256_add_ps(_mm256_mul_ps(acc1_r0, scale1), bias1);
                acc0_r1 = _mm256_add_ps(_mm256_mul_ps(acc0_r1, scale0), bias0);
                acc1_r1 = _mm256_add_ps(_mm256_mul_ps(acc1_r1, scale1), bias1);

                if (apply_relu) {
                    acc0_r0 = _mm256_max_ps(acc0_r0, zero_v);
                    acc1_r0 = _mm256_max_ps(acc1_r0, zero_v);
                    acc0_r1 = _mm256_max_ps(acc0_r1, zero_v);
                    acc1_r1 = _mm256_max_ps(acc1_r1, zero_v);
                }

                if (valid_lanes == 8) {
                    _mm256_storeu_ps(y.ptr(oc + 0, oh,     ow_cur), acc0_r0);
                    _mm256_storeu_ps(y.ptr(oc + 1, oh,     ow_cur), acc1_r0);
                    _mm256_storeu_ps(y.ptr(oc + 0, oh + 1, ow_cur), acc0_r1);
                    _mm256_storeu_ps(y.ptr(oc + 1, oh + 1, ow_cur), acc1_r1);
                } else {
                    store_width_masked_8(y.ptr(oc + 0, oh,     ow_cur), acc0_r0, valid_lanes);
                    store_width_masked_8(y.ptr(oc + 1, oh,     ow_cur), acc1_r0, valid_lanes);
                    store_width_masked_8(y.ptr(oc + 0, oh + 1, ow_cur), acc0_r1, valid_lanes);
                    store_width_masked_8(y.ptr(oc + 1, oh + 1, ow_cur), acc1_r1, valid_lanes);
                }
            };

            auto compute_scalar_rowpair_cell = [&](int ow_cur, int r) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;

                const int oh_cur = oh + r;

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;

                    for (const OCBlock2Action& action : bucket.actions) {
                        const int ih =
                            oh_cur * conv.stride_h + static_cast<int>(action.ky);
                        const int iw =
                            ow_cur * conv.stride_w + static_cast<int>(action.kx);

                        const int idx =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        const float xv = xin.data[static_cast<size_t>(idx)];

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

                y(oc + 0, oh_cur, ow_cur) = acc0;
                y(oc + 1, oh_cur, ow_cur) = acc1;
            };

            int ow = 0;

            for (; ow + 7 < out_w; ow += 8) {
                compute_store_rowpair_block(ow, 8);
            }

            if (ow < out_w) {
                if (use_masked_tail) {
                    const int valid_lanes = out_w - ow;
                    compute_store_rowpair_block(ow, valid_lanes);
                } else {
                    for (; ow < out_w; ++ow) {
                        compute_scalar_rowpair_cell(ow, 0);
                        compute_scalar_rowpair_cell(ow, 1);
                    }
                }
            }
        }
        
        // Tail row if out_h is odd. Use the existing non-spatial kernel logic.
        for (; oh < out_h; ++oh) {
            auto compute_store_single_row_block = [&](int ow_cur, int valid_lanes) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int ih =
                            oh * conv.stride_h + static_cast<int>(action.ky);
                        const int iw =
                            ow_cur * conv.stride_w + static_cast<int>(action.kx);

                        const int base =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        __m256 xvec;

                        if (valid_lanes == 8) {
                            xvec = load_width_8(
                                xin.data.data() + static_cast<size_t>(base),
                                conv.stride_w
                            );
                        } else {
                            xvec = load_width_tail_8(
                                xin.data.data() + static_cast<size_t>(base),
                                conv.stride_w,
                                valid_lanes
                            );
                        }

                        __m256 dummy0 = _mm256_setzero_ps();
                        __m256 dummy1 = _mm256_setzero_ps();

                        apply_pattern_spatial2(
                            bucket.pattern_id,
                            acc0,
                            acc1,
                            dummy0,
                            dummy1,
                            xvec,
                            _mm256_setzero_ps()
                        );
                    }
                }

                acc0 = _mm256_add_ps(_mm256_mul_ps(acc0, scale0), bias0);
                acc1 = _mm256_add_ps(_mm256_mul_ps(acc1, scale1), bias1);

                if (apply_relu) {
                    acc0 = _mm256_max_ps(acc0, zero_v);
                    acc1 = _mm256_max_ps(acc1, zero_v);
                }

                if (valid_lanes == 8) {
                    _mm256_storeu_ps(y.ptr(oc + 0, oh, ow_cur), acc0);
                    _mm256_storeu_ps(y.ptr(oc + 1, oh, ow_cur), acc1);
                } else {
                    store_width_masked_8(y.ptr(oc + 0, oh, ow_cur), acc0, valid_lanes);
                    store_width_masked_8(y.ptr(oc + 1, oh, ow_cur), acc1, valid_lanes);
                }
            };

            auto compute_scalar_single_row_cell = [&](int ow_cur) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;

                    for (const OCBlock2Action& action : bucket.actions) {
                        const int ih =
                            oh * conv.stride_h + static_cast<int>(action.ky);
                        const int iw =
                            ow_cur * conv.stride_w + static_cast<int>(action.kx);

                        const int idx =
                            (static_cast<int>(action.ic) * xin.h + ih) * xin.w + iw;

                        const float xv = xin.data[static_cast<size_t>(idx)];

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

                y(oc + 0, oh, ow_cur) = acc0;
                y(oc + 1, oh, ow_cur) = acc1;
            };

            int ow = 0;

            for (; ow + 7 < out_w; ow += 8) {
                compute_store_single_row_block(ow, 8);
            }

            if (ow < out_w) {
                if (use_masked_tail) {
                    const int valid_lanes = out_w - ow;
                    compute_store_single_row_block(ow, valid_lanes);
                } else {
                    for (; ow < out_w; ++ow) {
                        compute_scalar_single_row_cell(ow);
                    }
                }
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

Tensor conv3x3_pattern_ocblock2_spatial2_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3_pattern_ocblock2_spatial2_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        false
    );
}

Tensor conv3x3_pattern_ocblock2_spatial2_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
) {
    return conv3x3_pattern_ocblock2_spatial2_bn_avx2_impl(
        x,
        conv,
        plan,
        bn,
        workspace,
        true
    );
}