#pragma once

#include "block32_plan.hpp"
#include "model.hpp"
#include "q2q_tensor.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace nhwc_block32 {

// Flat oc_tile-major weight panel for 3x3/stride1/pad1 block32 ternary convs:
// w[oc_tile][k_idx][oc_lane][32], where oc_tile in [0, out_channels/4),
// k_idx = (ky*3 + kx) * ic_blocks + ic_block in [0, 9*ic_blocks), and
// oc_lane in [0,4). Built from a Block32ConvPlan whose per-output-channel
// block lists are uniform (require_uniform_block_layout) and contain exactly
// 9*ic_blocks entries in canonical (ky,kx,ic_block) order -- true in practice
// for ternary weights at width 32 (see Block32 in block32_plan.hpp).
struct Conv3x3S1P1PanelPlan {
    int out_channels = 0;
    int ic_blocks = 0;
    std::vector<std::vector<std::array<std::array<int8_t, BLOCK32>, 4>>> w;  // [oc_tile][k_idx][lane][32]
};

// Builds a Conv3x3S1P1PanelPlan from `plan`. Throws unless plan.kh==plan.kw==3,
// plan.stride_h==plan.stride_w==1, plan.pad_h==plan.pad_w==1,
// plan.out_channels % 4 == 0, and every output channel's block list has
// exactly 9*plan.ic_blocks entries in canonical (ky,kx,ic_block) order.
Conv3x3S1P1PanelPlan build_conv3x3s1p1_panel_plan(const Block32ConvPlan& plan);

// Like build_conv3x3s1p1_panel_plan but allows any stride (still requires
// plan.kh==plan.kw==3, plan.pad_h==plan.pad_w==1, plan.out_channels % 4 == 0).
// Used for the layer2/3/4 3x3/stride2/pad1 transition convs.
Conv3x3S1P1PanelPlan build_conv3x3_panel_plan_strided(const Block32ConvPlan& plan);

// Specialized OC4_N1 (single output pixel per iteration), flat weight-panel
// addressing, hsum once per channel at the end of the K=9*ic_blocks
// reduction. ic_blocks must be 4 (128-channel layer2 3x3/s1/p1 convs,
// 28x28) -- throws otherwise.
void conv3x3s1p1_128x128_28x28_oc4_n1_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC-major, N_TILE=2 (two adjacent output columns per iteration, 8
// accumulators), flat weight-panel addressing, hsum once per accumulator at
// the end of the K=9*ic_blocks reduction. Requires OW % 2 == 0. ic_blocks
// must be 4 (128-channel layer2 3x3/s1/p1 convs, 28x28).
void conv3x3s1p1_128x128_28x28_oc4_n2_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC-major, N_TILE=4 (16 accumulators), flat weight-panel addressing.
// Requires OW % 4 == 0. ic_blocks must be 4 (128-channel layer2 3x3/s1/p1
// convs, 28x28).
void conv3x3s1p1_128x128_28x28_oc4_n4_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC-major, N_TILE=2, flat weight-panel addressing. ic_blocks must be 8
// (256-channel layer3 3x3/s1/p1 convs, 14x14).
void conv3x3s1p1_256x256_14x14_oc4_n2_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC-major, single output column per iteration (4 accumulators), flat
// weight-panel addressing. ic_blocks must be 16 (512-channel layer4 3x3/s1/p1
// convs, 7x7).
void conv3x3s1p1_512x512_7x7_oc4_n1_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC-major, N_TILE=2 main loop with a single-column tail for odd OW (e.g.
// OW=7: three N2 iterations + one N1 tail), flat weight-panel addressing.
// ic_blocks must be 16 (512-channel layer4 3x3/s1/p1 convs, 7x7).
void conv3x3s1p1_512x512_7x7_oc4_n2_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Flat oc_tile-major weight panel for 1x1/stride2/pad0 block32 downsample
// convs: w[oc_tile][ic_block][oc_lane][32], oc_tile in [0, out_channels/4),
// ic_block in [0, ic_blocks), oc_lane in [0,4).
struct Conv1x1PanelPlan {
    int out_channels = 0;
    int ic_blocks = 0;
    std::vector<std::vector<std::array<std::array<int8_t, BLOCK32>, 4>>> w;  // [oc_tile][ic_block][lane][32]
};

// Builds a Conv1x1PanelPlan from `plan`. Throws unless plan.kh==plan.kw==1,
// plan.stride_h==plan.stride_w==2, plan.pad_h==plan.pad_w==0, and
// plan.out_channels % 4 == 0.
Conv1x1PanelPlan build_conv1x1_panel_plan(const Block32ConvPlan& plan);

// OC4 1x1/stride2/pad0 block32 conv, flat weight-panel addressing, hsum once
// per output channel. ic_blocks is read at runtime from `plan`, so this
// supports every downsample shape in the model (64->128: ic_blocks=2,
// 128->256: ic_blocks=4, 256->512: ic_blocks=8).
void conv1x1s2_block32_oc4_avx2(
    const Conv1x1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

} // namespace nhwc_block32
