#pragma once

#include "block32_plan.hpp"
#include "model.hpp"
#include "nhwc_common.hpp"
#include "q2q_tensor.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace nhwc_block32 {

// conv1 (7x7/stride2/pad3, in_channels==3) is too narrow for the
// (ky,kx,ic_block32) Block32ConvPlan used elsewhere (in_channels==3 is not a
// multiple of BLOCK32). Instead, gather each output pixel's whole receptive
// field (kh*kw*in_channels = 147 for the standard conv1) into a single
// zero-padded K=160 (5*BLOCK32) u8 vector, in (ky,kx,ic) order:
//   idx = (ky*kw + kx) * in_channels + ic,  idx in [0, kh*kw*in_channels)
// padded with zeros for idx in [kh*kw*in_channels, CONV1_K_PADDED). Weights
// are pre-gathered into the same layout (zero in the padding region), so the
// per-pixel dot product reuses dot_u8_i8_32_partial_avx2 across
// CONV1_K_BLOCKS=5 blocks.
constexpr int CONV1_K_PADDED = 160;
constexpr int CONV1_K_BLOCKS = CONV1_K_PADDED / BLOCK32;

struct Conv1Block32Plan {
    int in_channels = 0;
    int out_channels = 0;
    int kh = 0, kw = 0;
    int stride_h = 0, stride_w = 0;
    int pad_h = 0, pad_w = 0;

    // w[oc][block][i], block in [0,CONV1_K_BLOCKS), i in [0,BLOCK32).
    std::vector<std::array<std::array<int8_t, BLOCK32>, CONV1_K_BLOCKS>> w;
};

// Builds a Conv1Block32Plan from `conv`. Throws if in_channels*kh*kw exceeds
// CONV1_K_PADDED, if the weight tensor size doesn't match the declared
// shape, or if any weight value is outside {-1,0,+1}.
Conv1Block32Plan build_conv1_block32_plan(const Conv2DParams& conv);

// AVX2 OC4 conv1: for each output pixel, gathers the receptive field into a
// CONV1_K_PADDED-wide u8 buffer (see layout above), then dots it against
// CONV1_K_BLOCKS 32-wide weight blocks per output channel, four channels at a
// time, with fused BN + optional ReLU + requant. plan.out_channels must be a
// multiple of 4. `in.border` must be >= plan.pad_h/pad_w (true for
// in.border == plan.pad_h, the convention used by build_block32_resnet18_context).
void conv_block32_conv1_oc4_avx2(
    const Conv1Block32Plan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Independent scalar reference for conv1, computed directly from
// Conv2DParams::weight (not the plan) -- verifies the plan-building and the
// AVX2 gather/dot logic together.
void conv1_scalar(
    const Conv2DParams& conv, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

} // namespace nhwc_block32
