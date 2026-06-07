#pragma once

#include "q2q_tensor.hpp"

namespace q2q {

void q2q_residual_add_relu(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, QTensorNHWC& y);

void q2q_residual_add_relu_avx2(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, QTensorNHWC& y);

} // namespace q2q
