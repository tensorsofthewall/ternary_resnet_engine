#include "block32_conv_avx2.hpp"

#include "nhwc_common.hpp"
#include "nhwc_q8_dot_avx2.hpp"

#include <immintrin.h>
#include <algorithm>

namespace nhwc_block32 {

void conv_block32_oc4_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    require_uniform_block_layout(plan);

    const int OC = plan.out_channels;
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oc0 = 0; oc0 < OC; oc0 += 4) {
        const auto& b0 = plan.oc_plans[static_cast<size_t>(oc0 + 0)].blocks;
        const auto& b1 = plan.oc_plans[static_cast<size_t>(oc0 + 1)].blocks;
        const auto& b2 = plan.oc_plans[static_cast<size_t>(oc0 + 2)].blocks;
        const auto& b3 = plan.oc_plans[static_cast<size_t>(oc0 + 3)].blocks;
        const size_t n = b0.size();

        float s[4], bb[4];
        for (int k = 0; k < 4; ++k) {
            s[k] = bn.scale[static_cast<size_t>(oc0 + k)] * dq;
            bb[k] = bn.bias[static_cast<size_t>(oc0 + k)];
        }

        for (int oh = 0; oh < OH; ++oh) {
            uint8_t* out_row = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * out.W + static_cast<size_t>(out.border)) * out.C)
                + static_cast<size_t>(oc0);

            for (int ow = 0; ow < OW; ++ow) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                for (size_t i = 0; i < n; ++i) {
                    const Block32& blk = b0[i];
                    const int ih = oh * plan.stride_h + blk.ky + read_offset_h;
                    const int iw = ow * plan.stride_w + blk.kx + read_offset_w;
                    const size_t base =
                        (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
                        + static_cast<size_t>(blk.ic_block) * BLOCK32;
                    const uint8_t* a = &in.data[base];

                    acc0 = _mm256_add_epi32(acc0, dot_u8_i8_32_partial_avx2(a, b0[i].w));
                    acc1 = _mm256_add_epi32(acc1, dot_u8_i8_32_partial_avx2(a, b1[i].w));
                    acc2 = _mm256_add_epi32(acc2, dot_u8_i8_32_partial_avx2(a, b2[i].w));
                    acc3 = _mm256_add_epi32(acc3, dot_u8_i8_32_partial_avx2(a, b3[i].w));
                }

                const int32_t d[4] = {
                    hsum_epi32(acc0), hsum_epi32(acc1), hsum_epi32(acc2), hsum_epi32(acc3),
                };

                uint8_t* out_px = out_row + static_cast<size_t>(ow) * static_cast<size_t>(out.C);
                epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
            }
        }
    }
}

void conv_block32_scalar(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int OC = plan.out_channels;
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oc = 0; oc < OC; ++oc) {
        const auto& blocks = plan.oc_plans[static_cast<size_t>(oc)].blocks;
        const float s = bn.scale[static_cast<size_t>(oc)] * dq;
        const float b = bn.bias[static_cast<size_t>(oc)];

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc = 0;
                for (const Block32& blk : blocks) {
                    const int ih = oh * plan.stride_h + blk.ky + read_offset_h;
                    const int iw = ow * plan.stride_w + blk.kx + read_offset_w;
                    const size_t base =
                        (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
                        + static_cast<size_t>(blk.ic_block) * BLOCK32;
                    for (int ci = 0; ci < BLOCK32; ++ci) {
                        acc += static_cast<int32_t>(in.data[base + static_cast<size_t>(ci)])
                             * static_cast<int32_t>(blk.w[ci]);
                    }
                }
                float v = static_cast<float>(acc) * s + b;
                if (relu) v = std::max(v, 0.0f);
                const size_t out_idx =
                    (static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C)
                    + static_cast<size_t>(oc);
                out.data[out_idx] = requantize_u8(v, inv_out_scale);
            }
        }
    }
}

} // namespace nhwc_block32
