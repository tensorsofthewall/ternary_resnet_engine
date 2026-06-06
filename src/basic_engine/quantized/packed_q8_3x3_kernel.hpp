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

// ---------------------------------------------------------------------------
// Layer2: IC=128, OH=28, OW=28, K=1152, N_TILE=28, N_PAD=32
// A_panel: caller-provided buffer of K * 32 = 36864 bytes.
// ---------------------------------------------------------------------------

void pack_a_panel_layer2(const QuantTensorI8& xq, int oh, int8_t* A_panel);

void packed_q8_3x3_s1_layer2_bn_avx2(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out);

void packed_q8_3x3_s1_layer2_bn_relu_avx2(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out);

// ---------------------------------------------------------------------------
// OC_TILE=2 (Phase 3): merged pair plan — one A_panel load per k serves both OCs.
//
// PairAction: for each k where at least one of w0/w1 is nonzero, stores both
// weights. Inner loop uses sign_epi32 for branch-free weighting (w in {-1,0,+1}).
// Four independent int32 accumulator chains for better OOO pipeline utilisation.
//
// Layer3 only (layer4 ruled out in Phase 2: panel 72KB exceeds L1).
// ---------------------------------------------------------------------------

struct PairAction {
    uint16_t k;
    int8_t   w0;   // weight for oc*2+0; may be 0 when only w1 is nonzero at this k
    int8_t   w1;   // weight for oc*2+1; may be 0 when only w0 is nonzero at this k
};

struct PackedSparsePairPlan {
    int K  = 0;
    int OC = 0;
    std::vector<std::vector<PairAction>> pairs;  // [OC/2]; oc = pair_idx*2
};

PackedSparsePairPlan build_packed_sparse_pair_plan(const Conv2DParams& conv);

void packed_q8_3x3_s1_layer3_oc2_bn_avx2(
    const int8_t*               A_panel,
    const PackedSparsePairPlan& plan,
    const BNFusedParams&        bn,
    float                       xq_scale,
    int                         oh,
    Tensor&                     y_out);

void packed_q8_3x3_s1_layer3_oc2_bn_relu_avx2(
    const int8_t*               A_panel,
    const PackedSparsePairPlan& plan,
    const BNFusedParams&        bn,
    float                       xq_scale,
    int                         oh,
    Tensor&                     y_out);

// ---------------------------------------------------------------------------
// OC_TILE=4 (Phase 3): merged quad plan — one A_panel load per k serves 4 OCs.
//
// Eight independent int32 accumulator chains.  AVX2 has 16 YMM registers;
// the quad kernel uses at most 13 simultaneously (8 accumulators + raw/lo/hi +
// one sign broadcast), so no spilling expected.
// ---------------------------------------------------------------------------

struct QuadAction {
    uint16_t k;
    int8_t   w0, w1, w2, w3;
    int8_t   _pad[2];  // pad to 8 bytes: 8 QuadActions per cache line
};

struct PackedSparseQuadPlan {
    int K  = 0;
    int OC = 0;
    std::vector<std::vector<QuadAction>> quads;  // [OC/4]; oc = quad_idx*4
};

PackedSparseQuadPlan build_packed_sparse_quad_plan(const Conv2DParams& conv);

void packed_q8_3x3_s1_layer3_oc4_bn_avx2(
    const int8_t*               A_panel,
    const PackedSparseQuadPlan& plan,
    const BNFusedParams&        bn,
    float                       xq_scale,
    int                         oh,
    Tensor&                     y_out);

void packed_q8_3x3_s1_layer3_oc4_bn_relu_avx2(
    const int8_t*               A_panel,
    const PackedSparseQuadPlan& plan,
    const BNFusedParams&        bn,
    float                       xq_scale,
    int                         oh,
    Tensor&                     y_out);
