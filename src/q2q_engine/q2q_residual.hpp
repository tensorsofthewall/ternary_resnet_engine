#pragma once

#include "q2q_tensor.hpp"

namespace q2q {

// q8-to-q8 residual block: dequantizes `a`/`b` (each via its own per-tensor
// scale) to FP32, adds, applies ReLU, and requantizes to u8 with `out_scale`.
// Scalar reference implementation -- never a materialized FP32 tensor (the
// dequant/add/relu/requant chain runs per element).
void q2q_residual_add_relu(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, QTensorNHWC& y);

// AVX2 8-wide vectorized version of q2q_residual_add_relu, numerically
// equivalent modulo float-rounding-mode differences (round-to-nearest-even
// vs. std::lround's round-half-away-from-zero). C must be a multiple of 8
// (true for every q8 tensor in this engine, which are multiples of 32).
void q2q_residual_add_relu_avx2(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, QTensorNHWC& y);

} // namespace q2q
