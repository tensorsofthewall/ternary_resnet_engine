#pragma once

#include "block32_conv1.hpp"
#include "block32_plan.hpp"
#include "im2col_gemm.hpp"
#include "model.hpp"
#include "nhwc_common.hpp"
#include "q2q_tensor.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nhwc_block32 {

// Gathers the receptive field for output pixel (oh,ow) into `patch[160]`,
// byte-by-byte, in flatten order ((ky*kw)+kx)*C+ic. Zero-fills
// patch[kh*kw*C .. CONV1_K_PADDED).
inline void build_conv1_patch_generic(
    const q2q::QTensorNHWC& in, int C, int kh, int kw,
    int ih0, int iw0, uint8_t* patch)
{
    for (int ky = 0; ky < kh; ++ky) {
        const int ih = ih0 + ky;
        for (int kx = 0; kx < kw; ++kx) {
            const int iw = iw0 + kx;
            const uint8_t* src = &in.data[
                (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw))
                * static_cast<size_t>(C)];
            const int idx = (ky * kw + kx) * C;
            for (int ic = 0; ic < C; ++ic) patch[idx + ic] = src[ic];
        }
    }
    const int valid = kh * kw * C;
    std::memset(patch + valid, 0, static_cast<size_t>(CONV1_K_PADDED - valid));
}

// Gathers the receptive field for output pixel (oh,ow) into `patch[160]` via
// kh row-memcpys of (kw*C) bytes each -- valid only when, for every ky in
// [0,kh), the row [ih0+ky] has (kw*C) contiguous valid bytes starting at
// column iw0 (i.e. iw0 >= 0 and iw0+kw <= in.W). Zero-fills
// patch[kh*kw*C .. CONV1_K_PADDED).
inline void build_conv1_patch_interior(
    const q2q::QTensorNHWC& in, int C, int kh, int kw,
    int ih0, int iw0, uint8_t* patch)
{
    const int row_bytes = kw * C;
    for (int ky = 0; ky < kh; ++ky) {
        const uint8_t* src = &in.data[
            (static_cast<size_t>(ih0 + ky) * static_cast<size_t>(in.W) + static_cast<size_t>(iw0))
            * static_cast<size_t>(C)];
        std::memcpy(patch + ky * row_bytes, src, static_cast<size_t>(row_bytes));
    }
    const int valid = kh * kw * C;
    std::memset(patch + valid, 0, static_cast<size_t>(CONV1_K_PADDED - valid));
}

// Conv1 weight plan, repacked for the OC4 kernels below: w[oc_tile][k_block]
// holds the (oc_lane, 32-lane) weight blocks for the four output channels
// oc_tile*4 .. oc_tile*4+3, i.e. w[oc_tile][k_block][oc_lane][0..31]. This
// keeps the four weight blocks dotted against the same preloaded activation
// block contiguous in memory, vs. Conv1Block32Plan's w[oc][block][32] where
// they are CONV1_K_BLOCKS*32 bytes apart.
struct Conv1Block32PlanV2 {
    int in_channels = 0;
    int out_channels = 0;
    int kh = 0, kw = 0;
    int stride_h = 0, stride_w = 0;
    int pad_h = 0, pad_w = 0;

    // w[oc_tile][k_block][oc_lane][32], oc_tile in [0,out_channels/4),
    // k_block in [0,CONV1_K_BLOCKS), oc_lane in [0,4).
    std::vector<std::array<std::array<std::array<int8_t, BLOCK32>, 4>, CONV1_K_BLOCKS>> w;
};

// Builds a Conv1Block32PlanV2 from `conv`. Same validation as
// build_conv1_block32_plan (K <= CONV1_K_PADDED, weight tensor size,
// ternary values), but out_channels must additionally be a multiple of 4.
Conv1Block32PlanV2 build_conv1_block32_plan_v2(const Conv2DParams& conv);

// OC4 conv1, "generic patch" path: for each output pixel, gathers the
// receptive field byte-by-byte into an alignas(32) uint8_t patch[160]
// (flatten order ((ky*kw)+kx)*in_channels+ic, zero-padded for
// idx in [kh*kw*in_channels, CONV1_K_PADDED)), loads the patch's
// CONV1_K_BLOCKS 32-byte blocks into registers once, and reuses them across
// every oc_tile via dot_u8_i8_32_partial_preloaded_avx2. Fused BN + optional
// ReLU + requant, NHWC q8 output. plan.out_channels must be a multiple of 4.
void conv_block32_conv1_oc4_v2_generic_avx2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC4 conv1, "interior/border split" path: for each output pixel, builds the
// patch via a 7x21-byte row-memcpy fast path when the whole receptive field's
// rows are contiguous in `in.data` (the common case for 224x224 input), and
// falls back to the byte-by-byte generic gather (with zero padding for
// idx >= kh*kw*in_channels) otherwise. Same compute/epilogue as
// conv_block32_conv1_oc4_v2_generic_avx2.
void conv_block32_conv1_oc4_v2_split_avx2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC4 conv1, "interior row-memcpy" path: like
// conv_block32_conv1_oc4_v2_split_avx2 but unconditionally uses the
// 7x21-byte row-memcpy patch builder for every output pixel (no
// interior/border branch). Only valid when every output pixel's receptive
// field rows are contiguous in `in.data` -- true for the standard
// 224x224/7x7/stride2/pad3 conv1 with in.border == pad_h == pad_w, but NOT a
// general-purpose kernel. Same compute/epilogue as the other v2 kernels.
void conv_block32_conv1_oc4_v2_interior_avx2(
    const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Phase 16 Step 2: flattens a Conv1Block32PlanV2 (already in
// w[oc_tile][k_block][oc_lane][32] order) into a PackedBPanel with
// out_channels=plan.out_channels, K_pad=CONV1_K_PADDED (160, 5 k_blocks).
PackedBPanel build_packed_b_panel_conv1(const Conv1Block32PlanV2& plan);

// Phase 16 Step 2: panel-GEMM conv1. For each tile of N_TILE adjacent output
// columns at a given row, gathers each pixel's CONV1_K_PADDED-wide patch
// (build_conv1_patch_interior/generic, same as the v2 kernels) into
// panel[N_TILE][160], then runs the OC4 GEMM microkernel (oc_tile-major,
// weight blocks loaded once per oc_tile and reused across all N_TILE pixels)
// against B for every oc_tile. Falls back to an N_TILE=1 tail for OW not a
// multiple of N_TILE. plan.out_channels must be a multiple of 4.
void conv1_panel_gemm_oc4_n8_avx2(
    const PackedBPanel& B, const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void conv1_panel_gemm_oc4_n16_avx2(
    const PackedBPanel& B, const Conv1Block32PlanV2& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

} // namespace nhwc_block32
