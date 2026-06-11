#include "block32_conv1.hpp"

#include "nhwc_q8_dot_avx2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace nhwc_block32 {

namespace {

size_t conv_weight_index(const Conv2DParams& conv, int oc, int ic, int ky, int kx) {
    return static_cast<size_t>(((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx);
}

} // namespace

Conv1Block32Plan build_conv1_block32_plan(const Conv2DParams& conv) {
    const int K = conv.in_channels * conv.kh * conv.kw;
    if (K > CONV1_K_PADDED) {
        throw std::runtime_error(
            "build_conv1_block32_plan: in_channels*kh*kw exceeds " +
            std::to_string(CONV1_K_PADDED) + " for conv: " + conv.name);
    }

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );
    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error("Conv weight size mismatch while building Conv1Block32Plan: " + conv.name);
    }

    Conv1Block32Plan plan;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;

    plan.w.assign(static_cast<size_t>(conv.out_channels), {});
    for (int oc = 0; oc < conv.out_channels; ++oc) {
        for (int ky = 0; ky < conv.kh; ++ky) {
            for (int kx = 0; kx < conv.kw; ++kx) {
                for (int ic = 0; ic < conv.in_channels; ++ic) {
                    const int8_t wv = conv.weight[conv_weight_index(conv, oc, ic, ky, kx)];
                    if (wv < -1 || wv > 1) {
                        throw std::runtime_error(
                            "Non-ternary weight value while building Conv1Block32Plan: " + conv.name);
                    }
                    const int idx = (ky * conv.kw + kx) * conv.in_channels + ic;
                    plan.w[static_cast<size_t>(oc)]
                          [static_cast<size_t>(idx / BLOCK32)]
                          [static_cast<size_t>(idx % BLOCK32)] = wv;
                }
            }
        }
    }

    return plan;
}

void conv_block32_conv1_oc4_avx2(
    const Conv1Block32Plan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    if (plan.out_channels % 4 != 0) {
        throw std::runtime_error("conv_block32_conv1_oc4_avx2: out_channels must be a multiple of 4");
    }

    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            alignas(32) uint8_t buf[CONV1_K_PADDED] = {0};
            for (int ky = 0; ky < plan.kh; ++ky) {
                const int ih = oh * plan.stride_h + ky + read_offset_h;
                for (int kx = 0; kx < plan.kw; ++kx) {
                    const int iw = ow * plan.stride_w + kx + read_offset_w;
                    const uint8_t* src = &in.data[
                        (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw))
                        * static_cast<size_t>(C)];
                    const int idx = (ky * plan.kw + kx) * C;
                    for (int ic = 0; ic < C; ++ic) buf[idx + ic] = src[ic];
                }
            }

            uint8_t* out_px = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W)
                    + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C));

            for (int oc0 = 0; oc0 < plan.out_channels; oc0 += 4) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                for (int b = 0; b < CONV1_K_BLOCKS; ++b) {
                    const uint8_t* a = buf + b * BLOCK32;
                    acc0 = _mm256_add_epi32(acc0, dot_u8_i8_32_partial_avx2(a, plan.w[static_cast<size_t>(oc0 + 0)][static_cast<size_t>(b)].data()));
                    acc1 = _mm256_add_epi32(acc1, dot_u8_i8_32_partial_avx2(a, plan.w[static_cast<size_t>(oc0 + 1)][static_cast<size_t>(b)].data()));
                    acc2 = _mm256_add_epi32(acc2, dot_u8_i8_32_partial_avx2(a, plan.w[static_cast<size_t>(oc0 + 2)][static_cast<size_t>(b)].data()));
                    acc3 = _mm256_add_epi32(acc3, dot_u8_i8_32_partial_avx2(a, plan.w[static_cast<size_t>(oc0 + 3)][static_cast<size_t>(b)].data()));
                }

                const int32_t d[4] = {
                    hsum_epi32(acc0), hsum_epi32(acc1), hsum_epi32(acc2), hsum_epi32(acc3),
                };

                for (int k = 0; k < 4; ++k) {
                    const int oc = oc0 + k;
                    float v = static_cast<float>(d[k]) * (bn.scale[static_cast<size_t>(oc)] * dq) + bn.bias[static_cast<size_t>(oc)];
                    if (relu) v = std::max(v, 0.0f);
                    out_px[oc] = requantize_u8(v, inv_out_scale);
                }
            }
        }
    }
}

void conv1_scalar(
    const Conv2DParams& conv, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int read_offset_h = in.border - conv.pad_h;
    const int read_offset_w = in.border - conv.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oc = 0; oc < conv.out_channels; ++oc) {
        const float s = bn.scale[static_cast<size_t>(oc)] * dq;
        const float b = bn.bias[static_cast<size_t>(oc)];

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc = 0;
                for (int ky = 0; ky < conv.kh; ++ky) {
                    const int ih = oh * conv.stride_h + ky + read_offset_h;
                    for (int kx = 0; kx < conv.kw; ++kx) {
                        const int iw = ow * conv.stride_w + kx + read_offset_w;
                        const size_t base = (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw))
                            * static_cast<size_t>(conv.in_channels);
                        for (int ic = 0; ic < conv.in_channels; ++ic) {
                            const uint8_t a = in.data[base + static_cast<size_t>(ic)];
                            const int8_t w = conv.weight[conv_weight_index(conv, oc, ic, ky, kx)];
                            acc += static_cast<int32_t>(a) * static_cast<int32_t>(w);
                        }
                    }
                }
                float v = static_cast<float>(acc) * s + b;
                if (relu) v = std::max(v, 0.0f);
                const size_t out_idx = (static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W)
                    + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C) + static_cast<size_t>(oc);
                out.data[out_idx] = requantize_u8(v, inv_out_scale);
            }
        }
    }
}

} // namespace nhwc_block32
