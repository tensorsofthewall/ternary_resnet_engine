#pragma once

#include "q2q_tensor.hpp"
#include "tensor.hpp"

#include <cmath>
#include <cstdint>

namespace q2q {

// Scales a dequantized FP32 value back to u8 by `v * inv_out_scale`
// (inv_out_scale = 1/out_scale), rounds to nearest, and clamps to [0, 255].
static inline uint8_t requantize_u8(float v, float inv_out_scale) {
    int32_t q = static_cast<int32_t>(std::lround(v * inv_out_scale));
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    return static_cast<uint8_t>(q);
}

// Quantizes an FP32 NCHW image (H*W*C contiguous floats, channel-major) into
// `dst`'s padded q8 NHWC layout in place, using the fixed ACT_SCALE constant
// as the quantization scale (`dst.scale` is left untouched -- the caller is
// expected to have set it to ACT_SCALE when allocating). The existing zero
// halo (dst.border) is preserved.
void quantize_nchw_fp32_to_q8_nhwc(const float* src, int H, int W, int C, QTensorNHWC& dst);

// Dequantizes a (logically H x W x C) region of a padded q8 NHWC tensor back
// to an FP32 NCHW Tensor, using `src.scale` and skipping `src.border`.
void dequantize_q8_nhwc_to_fp32_nchw(const QTensorNHWC& src, int H, int W, int C, Tensor& dst);

} // namespace q2q
