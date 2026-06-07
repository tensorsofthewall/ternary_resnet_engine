#pragma once

#include "blocked_tensor.hpp"
#include "blocked_model.hpp"
#include "bn_fusion.hpp"

// Reference scalar conv implementations (correctness-first, not optimized).
// Each function: FP32 input → per-tensor q8 → int32 accumulate → BN → FP32 output.

void blocked_conv3x3_s1_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output);

void blocked_conv3x3_s2_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output);

// 1×1 stride-2 downsample conv (pad=0).
void blocked_conv1x1_s2_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output);

// Phase 5 — AVX2 sign_epi32 mullo kernel.
// Loop order: (oc_cb, oh, ow) outer → (ic_cb, ky, kx, ci_in) inner.
// acc_lo/acc_hi stay in YMM registers. Skips zero activations.
void blocked_conv3x3_s1_avx2(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output);

void blocked_conv3x3_s2_avx2(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output);
