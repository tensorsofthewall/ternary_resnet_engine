#pragma once

#include "q2q_tensor.hpp"

namespace q2q {

// 3x3/stride=2/pad=1 max-pool directly in q8 (per-channel max of u8 lanes --
// monotonic under the shared affine quantization, so no dequant/requant needed).
// `x` must carry a >=1-wide border; OH/OW are the output's logical extents.
void q8_maxpool_3x3_s2_p1(const QTensorNHWC& x, int OH, int OW, QTensorNHWC& y);

} // namespace q2q
