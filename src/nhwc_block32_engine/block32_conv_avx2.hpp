#pragma once

#include "block32_plan.hpp"
#include "model.hpp"
#include "q2q_tensor.hpp"

namespace nhwc_block32 {

// Generic AVX2 OC4 block32 ternary conv + fused BN + optional ReLU + requant,
// reading from and writing to q2q::QTensorNHWC (NHWC q8, zero-filled border
// halo). Generalizes conv3x3_s1_block32_oc4_avx2 to arbitrary kernel size,
// stride, and padding (3x3 stride1/2 pad1, 1x1 stride2 pad0): for each
// (ky,kx,ic_block) block, the input row/col read at output position (oh,ow)
// is `oh*stride_h + ky + (in.border - plan.pad_h)` /
// `ow*stride_w + kx + (in.border - plan.pad_w)`. `in.border` must be large
// enough that this never indexes outside `in.data` for any (oh,ow) in
// [0,OH)x[0,OW) -- border=1 suffices for every conv in this model (3x3/pad1
// and 1x1/pad0). plan.out_channels must be a multiple of 4; throws via
// require_uniform_block_layout if the per-output-channel block layouts
// don't line up (true for every conv in this model).
void conv_block32_oc4_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Scalar reference for conv_block32_oc4_avx2 (same indexing, no SIMD). Used
// to verify the AVX2 kernel across the various conv shapes (3x3 s1/s2 pad1,
// 1x1 s2 pad0) that appear in layer2-4.
void conv_block32_scalar(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

} // namespace nhwc_block32
