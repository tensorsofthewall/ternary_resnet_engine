#pragma once

#include <cstdint>
#include <vector>
#include "model.hpp"
#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

// Pre-built sparse index plan for the packed activation-panel kernel.
// pos_idx[oc] and neg_idx[oc] hold K-indices into the A_panel K dimension.
// K = IC * KH * KW (e.g. 2304 for layer3, 4608 for layer4).
struct PackedSparseQ8Plan {
    int K  = 0;
    int OC = 0;
    std::vector<std::vector<uint16_t>> pos_idx;
    std::vector<std::vector<uint16_t>> neg_idx;
};

PackedSparseQ8Plan build_packed_sparse_q8_plan(const Conv2DParams& conv);

// ---------------------------------------------------------------------------
// Layer3: IC=256, OH=14, OW=14, K=2304, N_TILE=14, N_PAD=16
// A_panel: caller-provided buffer of K * 16 = 36864 bytes.
// pack_a_panel_layer3 must be called for each oh before the compute call.
// ---------------------------------------------------------------------------

void pack_a_panel_layer3(const QuantTensorI8& xq, int oh, int8_t* A_panel);

void packed_q8_3x3_s1_layer3_bn_avx2(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out);

void packed_q8_3x3_s1_layer3_bn_relu_avx2(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out);

// ---------------------------------------------------------------------------
// Layer4: IC=512, OH=7, OW=7, K=4608, N_TILE=7, N_PAD=16
// A_panel: caller-provided buffer of K * 16 = 73728 bytes.
// ---------------------------------------------------------------------------

void pack_a_panel_layer4(const QuantTensorI8& xq, int oh, int8_t* A_panel);

void packed_q8_3x3_s1_layer4_bn_avx2(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out);

void packed_q8_3x3_s1_layer4_bn_relu_avx2(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out);
