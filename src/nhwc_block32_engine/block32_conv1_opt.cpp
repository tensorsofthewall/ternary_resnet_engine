#include "block32_conv1_opt.hpp"

#include "nhwc_q8_dot_avx2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace nhwc_block32 {

namespace {

size_t conv_weight_index(const Conv2DParams& conv, int oc, int ic, int ky, int kx) {
    return static_cast<size_t>(((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx);
}

// Loads the CONV1_K_BLOCKS 32-byte blocks of `patch` once, then dots them
// against every oc_tile's weight blocks, four output channels at a time,
// hsumming once per output channel and applying the fused BN+ReLU+requant
// epilogue. Stores NHWC q8 into out_px[0..plan.out_channels).
inline void conv1_compute_pixel_v2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const uint8_t* patch, bool relu, float dq, float inv_out_scale,
    uint8_t* out_px)
{
    __m256i va[CONV1_K_BLOCKS];
    for (int b = 0; b < CONV1_K_BLOCKS; ++b)
        va[b] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(patch + b * BLOCK32));

    const int n_tiles = plan.out_channels / 4;
    for (int t = 0; t < n_tiles; ++t) {
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        for (int b = 0; b < CONV1_K_BLOCKS; ++b) {
            const auto& wb = plan.w[static_cast<size_t>(t)][static_cast<size_t>(b)];
            acc0 = _mm256_add_epi32(acc0, dot_u8_i8_32_partial_preloaded_avx2(va[b], wb[0].data()));
            acc1 = _mm256_add_epi32(acc1, dot_u8_i8_32_partial_preloaded_avx2(va[b], wb[1].data()));
            acc2 = _mm256_add_epi32(acc2, dot_u8_i8_32_partial_preloaded_avx2(va[b], wb[2].data()));
            acc3 = _mm256_add_epi32(acc3, dot_u8_i8_32_partial_preloaded_avx2(va[b], wb[3].data()));
        }

        const int32_t d[4] = {
            hsum_epi32(acc0), hsum_epi32(acc1), hsum_epi32(acc2), hsum_epi32(acc3),
        };

        for (int k = 0; k < 4; ++k) {
            const int oc = t * 4 + k;
            float v = static_cast<float>(d[k]) * (bn.scale[static_cast<size_t>(oc)] * dq) + bn.bias[static_cast<size_t>(oc)];
            if (relu) v = std::max(v, 0.0f);
            out_px[oc] = requantize_u8(v, inv_out_scale);
        }
    }
}

} // namespace

Conv1Block32PlanV2 build_conv1_block32_plan_v2(const Conv2DParams& conv) {
    const int K = conv.in_channels * conv.kh * conv.kw;
    if (K > CONV1_K_PADDED) {
        throw std::runtime_error(
            "build_conv1_block32_plan_v2: in_channels*kh*kw exceeds " +
            std::to_string(CONV1_K_PADDED) + " for conv: " + conv.name);
    }
    if (conv.out_channels % 4 != 0) {
        throw std::runtime_error(
            "build_conv1_block32_plan_v2: out_channels must be a multiple of 4 for conv: " + conv.name);
    }

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );
    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error("Conv weight size mismatch while building Conv1Block32PlanV2: " + conv.name);
    }

    Conv1Block32PlanV2 plan;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;

    plan.w.assign(static_cast<size_t>(conv.out_channels / 4), {});
    for (int oc = 0; oc < conv.out_channels; ++oc) {
        const int tile = oc / 4;
        const int lane = oc % 4;
        for (int ky = 0; ky < conv.kh; ++ky) {
            for (int kx = 0; kx < conv.kw; ++kx) {
                for (int ic = 0; ic < conv.in_channels; ++ic) {
                    const int8_t wv = conv.weight[conv_weight_index(conv, oc, ic, ky, kx)];
                    if (wv < -1 || wv > 1) {
                        throw std::runtime_error(
                            "Non-ternary weight value while building Conv1Block32PlanV2: " + conv.name);
                    }
                    const int idx = (ky * conv.kw + kx) * conv.in_channels + ic;
                    plan.w[static_cast<size_t>(tile)]
                          [static_cast<size_t>(idx / BLOCK32)]
                          [static_cast<size_t>(lane)]
                          [static_cast<size_t>(idx % BLOCK32)] = wv;
                }
            }
        }
    }

    return plan;
}

void conv_block32_conv1_oc4_v2_generic_avx2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * plan.stride_h + read_offset_h;
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * plan.stride_w + read_offset_w;

            alignas(32) uint8_t patch[CONV1_K_PADDED];
            build_conv1_patch_generic(in, C, plan.kh, plan.kw, ih0, iw0, patch);

            uint8_t* out_px = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W)
                    + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C));

            conv1_compute_pixel_v2(plan, bn, patch, relu, dq, inv_out_scale, out_px);
        }
    }
}

void conv_block32_conv1_oc4_v2_split_avx2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * plan.stride_h + read_offset_h;
        const bool row_interior = (ih0 >= 0) && (ih0 + plan.kh <= in.H);
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * plan.stride_w + read_offset_w;
            const bool col_interior = (iw0 >= 0) && (iw0 + plan.kw <= in.W);

            alignas(32) uint8_t patch[CONV1_K_PADDED];
            if (row_interior && col_interior) {
                build_conv1_patch_interior(in, C, plan.kh, plan.kw, ih0, iw0, patch);
            } else {
                build_conv1_patch_generic(in, C, plan.kh, plan.kw, ih0, iw0, patch);
            }

            uint8_t* out_px = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W)
                    + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C));

            conv1_compute_pixel_v2(plan, bn, patch, relu, dq, inv_out_scale, out_px);
        }
    }
}

void conv_block32_conv1_oc4_v2_interior_avx2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * plan.stride_h + read_offset_h;
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * plan.stride_w + read_offset_w;

            alignas(32) uint8_t patch[CONV1_K_PADDED];
            build_conv1_patch_interior(in, C, plan.kh, plan.kw, ih0, iw0, patch);

            uint8_t* out_px = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W)
                    + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C));

            conv1_compute_pixel_v2(plan, bn, patch, relu, dq, inv_out_scale, out_px);
        }
    }
}

PackedBPanel build_packed_b_panel_conv1(const Conv1Block32PlanV2& plan) {
    PackedBPanel out;
    out.out_channels = plan.out_channels;
    out.K_pad = CONV1_K_PADDED;
    const int n_oc_tiles = plan.out_channels / 4;
    out.data = AlignedBuffer32<int8_t>(static_cast<size_t>(n_oc_tiles) * CONV1_K_BLOCKS * 4 * BLOCK32);

    for (int t = 0; t < n_oc_tiles; ++t) {
        for (int kb = 0; kb < CONV1_K_BLOCKS; ++kb) {
            for (int lane = 0; lane < 4; ++lane) {
                std::memcpy(const_cast<int8_t*>(out.block(t, kb, lane)),
                            plan.w[static_cast<size_t>(t)][static_cast<size_t>(kb)][static_cast<size_t>(lane)].data(),
                            BLOCK32);
            }
        }
    }
    return out;
}

namespace {

// Phase 16 Step 2: panel-GEMM conv1 microkernel. For each tile of N_TILE
// adjacent output columns, gathers each pixel's CONV1_K_PADDED-wide patch
// (same interior/generic builders as the v2 kernels) into
// panel[N_TILE][CONV1_K_PADDED], then runs an oc_tile-major GEMM: each
// oc_tile's CONV1_K_BLOCKS weight blocks are loaded once and reused across
// all N_TILE pixels, with vector accumulators kept across all k_blocks and
// hsum'd once per pixel.
template <int N_TILE>
void conv1_panel_gemm_oc4_avx2_impl(
    const PackedBPanel& B, const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int C = plan.in_channels;
    const int read_offset_h = in.border - plan.pad_h;
    const int read_offset_w = in.border - plan.pad_w;
    const int k_blocks = B.K_pad / 32;
    const int n_oc_tiles = B.out_channels / 4;
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;
    const __m256i ones16 = _mm256_set1_epi16(1);

    AlignedBuffer32<uint8_t> panel(static_cast<size_t>(N_TILE) * static_cast<size_t>(B.K_pad));

    auto process_tile = [&](int oh, int ow0, int tile_width) {
        const int ih0 = oh * plan.stride_h + read_offset_h;
        const bool row_interior = (ih0 >= 0) && (ih0 + plan.kh <= in.H);
        for (int p = 0; p < tile_width; ++p) {
            const int iw0 = (ow0 + p) * plan.stride_w + read_offset_w;
            const bool col_interior = (iw0 >= 0) && (iw0 + plan.kw <= in.W);
            uint8_t* patch = panel.data() + static_cast<size_t>(p) * static_cast<size_t>(B.K_pad);
            if (row_interior && col_interior) {
                build_conv1_patch_interior(in, C, plan.kh, plan.kw, ih0, iw0, patch);
            } else {
                build_conv1_patch_generic(in, C, plan.kh, plan.kw, ih0, iw0, patch);
            }
        }

        uint8_t* out_row = out.data.data()
            + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow0 + out.border)) * static_cast<size_t>(out.C));

        for (int t = 0; t < n_oc_tiles; ++t) {
            float s[4], bb[4];
            for (int k = 0; k < 4; ++k) {
                s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * dq;
                bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
            }

            __m256i acc[4][N_TILE];
            for (int lane = 0; lane < 4; ++lane)
                for (int p = 0; p < tile_width; ++p) acc[lane][p] = _mm256_setzero_si256();

            for (int kb = 0; kb < k_blocks; ++kb) {
                __m256i va[N_TILE];
                for (int p = 0; p < tile_width; ++p) {
                    va[p] = _mm256_load_si256(reinterpret_cast<const __m256i*>(
                        panel.data() + static_cast<size_t>(p) * static_cast<size_t>(B.K_pad) + kb * 32));
                }
                for (int lane = 0; lane < 4; ++lane) {
                    const __m256i vw = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.block(t, kb, lane)));
                    for (int p = 0; p < tile_width; ++p) {
                        const __m256i prod16 = _mm256_maddubs_epi16(va[p], vw);
                        acc[lane][p] = _mm256_add_epi32(acc[lane][p], _mm256_madd_epi16(prod16, ones16));
                    }
                }
            }

            for (int p = 0; p < tile_width; ++p) {
                int32_t d[4];
                for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane][p]);
                uint8_t* out_px = out_row + static_cast<size_t>(p) * static_cast<size_t>(out.C) + static_cast<size_t>(t * 4);
                epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
            }
        }
    };

    for (int oh = 0; oh < OH; ++oh) {
        int ow0 = 0;
        for (; ow0 + N_TILE <= OW; ow0 += N_TILE) process_tile(oh, ow0, N_TILE);
        for (; ow0 < OW; ++ow0) process_tile(oh, ow0, 1);
    }
}

} // namespace

void conv1_panel_gemm_oc4_n8_avx2(
    const PackedBPanel& B, const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv1_panel_gemm_oc4_avx2_impl<8>(B, plan, bn, in, relu, out, OH, OW);
}

void conv1_panel_gemm_oc4_n16_avx2(
    const PackedBPanel& B, const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv1_panel_gemm_oc4_avx2_impl<16>(B, plan, bn, in, relu, out, OH, OW);
}

} // namespace nhwc_block32
