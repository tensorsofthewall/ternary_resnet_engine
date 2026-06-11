#pragma once

#include "block32_plan.hpp"
#include "model.hpp"
#include "q2q_tensor.hpp"

namespace nhwc_block32 {

// OC4_N2, OC-major: outer loop over oc0 (groups of 4 output channels), then
// (oh, ow-pairs); each (ky,kx,ic_block) weight block is loaded once and
// dotted against the activation blocks for two adjacent ow positions, with
// 8 running __m256i accumulators (4 channels x 2 pixels) hsummed once each
// at the end of the block loop. 3x3/stride1/pad1 only; OW must be even.
void conv_block32_n2_oc_major_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC4_N2, pixel-major: outer loop over (oh, ow-pairs), then oc0. Same
// per-(oc0,ow-pair) compute as conv_block32_n2_oc_major_avx2, with the
// spatial and output-channel loops swapped. 3x3/stride1/pad1 only; OW must
// be even.
void conv_block32_n2_pixel_major_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Interior/border split: the interior (oh in [1,OH-2], ow in [1,OW-2]) is
// processed in N2 ow-pairs via the OC-major N2 compute; the 1-pixel-wide
// border ring (oh in {0,OH-1} or ow in {0,OW-1}) is processed pixel-by-pixel
// via the same per-pixel OC4 compute as conv_block32_oc4_avx2. Requires
// OH>=3, OW>=3, and (OW-2) even (true for layer1's 56x56, OH=OW=56).
void conv_block32_n2_split_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// OC4_N4, OC-major: like conv_block32_n2_oc_major_avx2 but processes 4
// adjacent ow positions per iteration (16 running __m256i accumulators: 4
// channels x 4 pixels). 3x3/stride1/pad1 only; OW must be a multiple of 4.
void conv_block32_n4_oc_major_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

} // namespace nhwc_block32
