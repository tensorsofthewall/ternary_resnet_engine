#include "block32_layer1_opt.hpp"

#include "nhwc_common.hpp"
#include "nhwc_q8_dot_avx2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

namespace nhwc_block32 {

namespace {

void check_3x3_s1_p1(const Block32ConvPlan& plan, const char* who) {
    if (plan.kh != 3 || plan.kw != 3 || plan.stride_h != 1 || plan.stride_w != 1) {
        throw std::runtime_error(std::string(who) + ": requires 3x3 stride1 conv");
    }
}

// Computes one (oh, ow..ow+1) x (oc0..oc0+3) tile: loads each (ky,kx,ic_block)
// weight block once and dots it against the activation blocks for both ow
// and ow+1, with 8 running accumulators (4 channels x 2 pixels) hsummed once
// each at the end.
inline void n2_oc4_tile(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out,
    int oh, int ow, int oc0, int read_offset_h, int read_offset_w,
    int C, float dq, float inv_out_scale)
{
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

    __m256i acc00 = _mm256_setzero_si256(), acc01 = _mm256_setzero_si256();
    __m256i acc02 = _mm256_setzero_si256(), acc03 = _mm256_setzero_si256();
    __m256i acc10 = _mm256_setzero_si256(), acc11 = _mm256_setzero_si256();
    __m256i acc12 = _mm256_setzero_si256(), acc13 = _mm256_setzero_si256();

    for (size_t i = 0; i < n; ++i) {
        const Block32& blk = b0[i];
        const int ih = oh + blk.ky + read_offset_h;
        const int iw = ow + blk.kx + read_offset_w;
        const size_t base =
            (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
            + static_cast<size_t>(blk.ic_block) * BLOCK32;
        const uint8_t* a0 = &in.data[base];
        const uint8_t* a1 = a0 + C;

        const __m256i vw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0[i].w));
        const __m256i vw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1[i].w));
        const __m256i vw2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2[i].w));
        const __m256i vw3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3[i].w));

        acc00 = _mm256_add_epi32(acc00, dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw0));
        acc01 = _mm256_add_epi32(acc01, dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw1));
        acc02 = _mm256_add_epi32(acc02, dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw2));
        acc03 = _mm256_add_epi32(acc03, dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw3));
        acc10 = _mm256_add_epi32(acc10, dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw0));
        acc11 = _mm256_add_epi32(acc11, dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw1));
        acc12 = _mm256_add_epi32(acc12, dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw2));
        acc13 = _mm256_add_epi32(acc13, dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw3));
    }

    const int32_t d0[4] = { hsum_epi32(acc00), hsum_epi32(acc01), hsum_epi32(acc02), hsum_epi32(acc03) };
    const int32_t d1[4] = { hsum_epi32(acc10), hsum_epi32(acc11), hsum_epi32(acc12), hsum_epi32(acc13) };

    uint8_t* out_px0 = out.data.data()
        + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C))
        + static_cast<size_t>(oc0);
    uint8_t* out_px1 = out_px0 + static_cast<size_t>(out.C);
    epilogue_oc4_avx2(d0, s, bb, relu, inv_out_scale, out_px0);
    epilogue_oc4_avx2(d1, s, bb, relu, inv_out_scale, out_px1);
}

// Computes one (oh,ow) x (oc0..oc0+3) single-pixel tile -- same per-pixel
// compute as conv_block32_oc4_avx2's inner body, used for the border ring in
// conv_block32_n2_split_avx2.
inline void oc4_single_tile(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out,
    int oh, int ow, int oc0, int read_offset_h, int read_offset_w,
    int C, float dq, float inv_out_scale)
{
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

    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (size_t i = 0; i < n; ++i) {
        const Block32& blk = b0[i];
        const int ih = oh + blk.ky + read_offset_h;
        const int iw = ow + blk.kx + read_offset_w;
        const size_t base =
            (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
            + static_cast<size_t>(blk.ic_block) * BLOCK32;
        const uint8_t* a = &in.data[base];

        acc0 = _mm256_add_epi32(acc0, dot_u8_i8_32_partial_avx2(a, b0[i].w));
        acc1 = _mm256_add_epi32(acc1, dot_u8_i8_32_partial_avx2(a, b1[i].w));
        acc2 = _mm256_add_epi32(acc2, dot_u8_i8_32_partial_avx2(a, b2[i].w));
        acc3 = _mm256_add_epi32(acc3, dot_u8_i8_32_partial_avx2(a, b3[i].w));
    }

    const int32_t d[4] = { hsum_epi32(acc0), hsum_epi32(acc1), hsum_epi32(acc2), hsum_epi32(acc3) };

    uint8_t* out_px = out.data.data()
        + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C))
        + static_cast<size_t>(oc0);
    epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
}

} // namespace

void conv_block32_n2_oc_major_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    require_uniform_block_layout(plan);
    check_3x3_s1_p1(plan, "conv_block32_n2_oc_major_avx2");
    if (OW % 2 != 0) throw std::runtime_error("conv_block32_n2_oc_major_avx2: OW must be even");

    const int OC = plan.out_channels;
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oc0 = 0; oc0 < OC; oc0 += 4) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ow += 2) {
                n2_oc4_tile(plan, bn, in, relu, out, oh, ow, oc0, read_offset_h, read_offset_w, C, dq, inv_out_scale);
            }
        }
    }
}

void conv_block32_n2_pixel_major_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    require_uniform_block_layout(plan);
    check_3x3_s1_p1(plan, "conv_block32_n2_pixel_major_avx2");
    if (OW % 2 != 0) throw std::runtime_error("conv_block32_n2_pixel_major_avx2: OW must be even");

    const int OC = plan.out_channels;
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ow += 2) {
            for (int oc0 = 0; oc0 < OC; oc0 += 4) {
                n2_oc4_tile(plan, bn, in, relu, out, oh, ow, oc0, read_offset_h, read_offset_w, C, dq, inv_out_scale);
            }
        }
    }
}

void conv_block32_n2_split_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    require_uniform_block_layout(plan);
    check_3x3_s1_p1(plan, "conv_block32_n2_split_avx2");
    if (OH < 3 || OW < 3) throw std::runtime_error("conv_block32_n2_split_avx2: OH and OW must be >= 3");
    if ((OW - 2) % 2 != 0) throw std::runtime_error("conv_block32_n2_split_avx2: OW-2 must be even");

    const int OC = plan.out_channels;
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    // Border ring (oh in {0,OH-1} or ow in {0,OW-1}): single-pixel OC4.
    for (int oh = 0; oh < OH; ++oh) {
        const bool border_row = (oh == 0 || oh == OH - 1);
        for (int ow = 0; ow < OW; ++ow) {
            if (!border_row && ow != 0 && ow != OW - 1) continue;
            for (int oc0 = 0; oc0 < OC; oc0 += 4) {
                oc4_single_tile(plan, bn, in, relu, out, oh, ow, oc0, read_offset_h, read_offset_w, C, dq, inv_out_scale);
            }
        }
    }

    // Interior (oh in [1,OH-2], ow in [1,OW-2]): N2 ow-pairs, OC-major.
    for (int oc0 = 0; oc0 < OC; oc0 += 4) {
        for (int oh = 1; oh <= OH - 2; ++oh) {
            for (int ow = 1; ow <= OW - 2; ow += 2) {
                n2_oc4_tile(plan, bn, in, relu, out, oh, ow, oc0, read_offset_h, read_offset_w, C, dq, inv_out_scale);
            }
        }
    }
}

void conv_block32_n4_oc_major_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    require_uniform_block_layout(plan);
    check_3x3_s1_p1(plan, "conv_block32_n4_oc_major_avx2");
    if (OW % 4 != 0) throw std::runtime_error("conv_block32_n4_oc_major_avx2: OW must be a multiple of 4");

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
            for (int ow = 0; ow < OW; ow += 4) {
                __m256i acc[4][4];
                for (int p = 0; p < 4; ++p)
                    for (int k = 0; k < 4; ++k)
                        acc[p][k] = _mm256_setzero_si256();

                for (size_t i = 0; i < n; ++i) {
                    const Block32& blk = b0[i];
                    const int ih = oh + blk.ky + read_offset_h;
                    const int iw = ow + blk.kx + read_offset_w;
                    const size_t base =
                        (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
                        + static_cast<size_t>(blk.ic_block) * BLOCK32;
                    const uint8_t* a0 = &in.data[base];
                    const uint8_t* a1 = a0 + C;
                    const uint8_t* a2 = a1 + C;
                    const uint8_t* a3 = a2 + C;

                    const __m256i vw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0[i].w));
                    const __m256i vw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1[i].w));
                    const __m256i vw2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2[i].w));
                    const __m256i vw3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3[i].w));

                    acc[0][0] = _mm256_add_epi32(acc[0][0], dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw0));
                    acc[0][1] = _mm256_add_epi32(acc[0][1], dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw1));
                    acc[0][2] = _mm256_add_epi32(acc[0][2], dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw2));
                    acc[0][3] = _mm256_add_epi32(acc[0][3], dot_u8_i8_32_partial_w_preloaded_avx2(a0, vw3));

                    acc[1][0] = _mm256_add_epi32(acc[1][0], dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw0));
                    acc[1][1] = _mm256_add_epi32(acc[1][1], dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw1));
                    acc[1][2] = _mm256_add_epi32(acc[1][2], dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw2));
                    acc[1][3] = _mm256_add_epi32(acc[1][3], dot_u8_i8_32_partial_w_preloaded_avx2(a1, vw3));

                    acc[2][0] = _mm256_add_epi32(acc[2][0], dot_u8_i8_32_partial_w_preloaded_avx2(a2, vw0));
                    acc[2][1] = _mm256_add_epi32(acc[2][1], dot_u8_i8_32_partial_w_preloaded_avx2(a2, vw1));
                    acc[2][2] = _mm256_add_epi32(acc[2][2], dot_u8_i8_32_partial_w_preloaded_avx2(a2, vw2));
                    acc[2][3] = _mm256_add_epi32(acc[2][3], dot_u8_i8_32_partial_w_preloaded_avx2(a2, vw3));

                    acc[3][0] = _mm256_add_epi32(acc[3][0], dot_u8_i8_32_partial_w_preloaded_avx2(a3, vw0));
                    acc[3][1] = _mm256_add_epi32(acc[3][1], dot_u8_i8_32_partial_w_preloaded_avx2(a3, vw1));
                    acc[3][2] = _mm256_add_epi32(acc[3][2], dot_u8_i8_32_partial_w_preloaded_avx2(a3, vw2));
                    acc[3][3] = _mm256_add_epi32(acc[3][3], dot_u8_i8_32_partial_w_preloaded_avx2(a3, vw3));
                }

                uint8_t* out_row = out.data.data()
                    + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C))
                    + static_cast<size_t>(oc0);

                for (int p = 0; p < 4; ++p) {
                    const int32_t d[4] = {
                        hsum_epi32(acc[p][0]), hsum_epi32(acc[p][1]), hsum_epi32(acc[p][2]), hsum_epi32(acc[p][3]),
                    };
                    epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_row + static_cast<size_t>(p) * static_cast<size_t>(out.C));
                }
            }
        }
    }
}

} // namespace nhwc_block32
