#pragma once

#include "bn_fusion.hpp"
#include "model.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "q2q_common.hpp"
#include "q2q_conv1_vnni.hpp"
#include "q2q_pool.hpp"
#include "q2q_quant.hpp"
#include "q2q_tensor.hpp"
#include "q2q_weight_panel.hpp"

#include <vector>

namespace q2q {

// Holds every per-layer Conv2DParams/BNFusedParams (as non-owning pointers
// into the Model -- the Model must outlive this context, exactly as the
// monolith's main() held references into its local `model`), every prebuilt
// weight panel, every scratch QTensorNHWC, and the basic_engine Tensor
// outputs (dequant/avgpool/fc) used by the dequant+head tail.
struct Q2QResnet18Context {
    const Conv2DParams* conv1 = nullptr;
    const BNFusedParams* bn1 = nullptr;

    const Conv2DParams* c10_conv1 = nullptr;
    const Conv2DParams* c10_conv2 = nullptr;
    const BNFusedParams* bn10_1 = nullptr;
    const BNFusedParams* bn10_2 = nullptr;
    const Conv2DParams* c11_conv1 = nullptr;
    const Conv2DParams* c11_conv2 = nullptr;
    const BNFusedParams* bn11_1 = nullptr;
    const BNFusedParams* bn11_2 = nullptr;

    const Conv2DParams* c20_conv1 = nullptr;
    const Conv2DParams* c20_conv2 = nullptr;
    const Conv2DParams* c20_ds = nullptr;
    const BNFusedParams* bn20_1 = nullptr;
    const BNFusedParams* bn20_2 = nullptr;
    const BNFusedParams* bn20_ds = nullptr;
    const Conv2DParams* c21_conv1 = nullptr;
    const Conv2DParams* c21_conv2 = nullptr;
    const BNFusedParams* bn21_1 = nullptr;
    const BNFusedParams* bn21_2 = nullptr;

    const Conv2DParams* c30_conv1 = nullptr;
    const Conv2DParams* c30_conv2 = nullptr;
    const Conv2DParams* c30_ds = nullptr;
    const BNFusedParams* bn30_1 = nullptr;
    const BNFusedParams* bn30_2 = nullptr;
    const BNFusedParams* bn30_ds = nullptr;
    const Conv2DParams* c31_conv1 = nullptr;
    const Conv2DParams* c31_conv2 = nullptr;
    const BNFusedParams* bn31_1 = nullptr;
    const BNFusedParams* bn31_2 = nullptr;

    const Conv2DParams* c40_conv1 = nullptr;
    const Conv2DParams* c40_conv2 = nullptr;
    const Conv2DParams* c40_ds = nullptr;
    const BNFusedParams* bn40_1 = nullptr;
    const BNFusedParams* bn40_2 = nullptr;
    const BNFusedParams* bn40_ds = nullptr;
    const Conv2DParams* c41_conv1 = nullptr;
    const Conv2DParams* c41_conv2 = nullptr;
    const BNFusedParams* bn41_1 = nullptr;
    const BNFusedParams* bn41_2 = nullptr;

    const LinearParams* fc = nullptr;

    // ---- derived dimensions ----
    int IC_C1 = 0, OC_C1 = 0, H_IN = 0, W_IN = 0;
    int OH1 = 0, OW1 = 0, OHp = 0, OWp = 0;
    int OC1 = 0, OC2 = 0, OC3 = 0, OC4 = 0;
    int H1 = 0, W1 = 0, H2 = 0, W2 = 0, H3 = 0, W3 = 0, H4 = 0, W4 = 0;

    // ---- panels ----
    Conv1Panel p_c1;
    std::vector<int8_t> p10_c1, p10_c2, p11_c1, p11_c2;
    std::vector<int8_t> p20_c1, p20_c2, p20_ds, p21_c1, p21_c2;
    std::vector<int8_t> p30_c1, p30_c2, p30_ds, p31_c1, p31_c2;
    std::vector<int8_t> p40_c1, p40_c2, p40_ds, p41_c1, p41_c2;

    // ---- shared scratch tensors (safe to share across the two sequential,
    // self-contained phases -- no cross-phase dependency) ----
    QTensorNHWC input_q8, conv1_out, pool_out;
    QTensorNHWC l10_mid, l10_conv2_out, l10_out;
    QTensorNHWC l11_mid, l11_conv2_out, l11_out;
    QTensorNHWC l20_mid, l20_conv2_out, l20_identity, l20_out;
    QTensorNHWC l21_mid, l21_conv2_out, l21_out;
    QTensorNHWC l30_mid, l30_conv2_out, l30_identity, l30_out;
    QTensorNHWC l31_mid, l31_conv2_out, l31_out;
    QTensorNHWC l40_mid, l40_conv2_out, l40_identity, l40_out;
    QTensorNHWC l41_mid, l41_conv2_out, l41_out;

    Tensor dequant_out, avgpool_out, fc_out;
};

// Looks up every Conv2DParams/BNFusedParams, derives dimensions, builds every
// weight panel, and allocates every scratch tensor -- everything that runs
// once before the timing loops (verbatim port of the monolith's pre-loop
// main() body).
Q2QResnet18Context build_q2q_resnet18_context(const Model& model);

// ---- generic orchestration: parameterized over (conv_fn, res_fn). Defined
// as templates in this header (rather than dispatched via function pointers)
// so the compiler instantiates one direct-call specialization per (conv_fn,
// res_fn) pair -- the same zero-overhead property the monolith's C++14
// generic lambdas (auto&& params) provided. Verbatim ports of run_layer1..4 /
// run_full, rewritten to operate on ctx.* members instead of captured locals. ----

template <class ConvFn, class ResFn>
void run_q2q_layer1(Q2QResnet18Context& ctx, ConvFn&& conv_fn, ResFn&& res_fn, const QTensorNHWC& in) {
    conv_fn(in,          ctx.p10_c1, *ctx.c10_conv1, *ctx.bn10_1, true,  ACT_SCALE, ctx.l10_mid, ctx.H1, ctx.W1);
    conv_fn(ctx.l10_mid, ctx.p10_c2, *ctx.c10_conv2, *ctx.bn10_2, false, ACT_SCALE, ctx.l10_conv2_out, ctx.H1, ctx.W1);
    res_fn(ctx.l10_conv2_out, in, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, ctx.l10_out);
    conv_fn(ctx.l10_out, ctx.p11_c1, *ctx.c11_conv1, *ctx.bn11_1, true,  ACT_SCALE, ctx.l11_mid, ctx.H1, ctx.W1);
    conv_fn(ctx.l11_mid, ctx.p11_c2, *ctx.c11_conv2, *ctx.bn11_2, false, ACT_SCALE, ctx.l11_conv2_out, ctx.H1, ctx.W1);
    res_fn(ctx.l11_conv2_out, ctx.l10_out, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, ctx.l11_out);
}

template <class ConvFn, class ResFn>
void run_q2q_layer2(Q2QResnet18Context& ctx, ConvFn&& conv_fn, ResFn&& res_fn, const QTensorNHWC& in) {
    conv_fn(in,          ctx.p20_c1, *ctx.c20_conv1, *ctx.bn20_1, true,  ACT_SCALE, ctx.l20_mid, ctx.H2, ctx.W2);
    conv_fn(ctx.l20_mid, ctx.p20_c2, *ctx.c20_conv2, *ctx.bn20_2, false, ACT_SCALE, ctx.l20_conv2_out, ctx.H2, ctx.W2);
    conv_fn(in,          ctx.p20_ds, *ctx.c20_ds,    *ctx.bn20_ds,false, ACT_SCALE, ctx.l20_identity, ctx.H2, ctx.W2);
    res_fn(ctx.l20_conv2_out, ctx.l20_identity, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, ctx.l20_out);
    conv_fn(ctx.l20_out, ctx.p21_c1, *ctx.c21_conv1, *ctx.bn21_1, true,  ACT_SCALE, ctx.l21_mid, ctx.H2, ctx.W2);
    conv_fn(ctx.l21_mid, ctx.p21_c2, *ctx.c21_conv2, *ctx.bn21_2, false, ACT_SCALE, ctx.l21_conv2_out, ctx.H2, ctx.W2);
    res_fn(ctx.l21_conv2_out, ctx.l20_out, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, ctx.l21_out);
}

template <class ConvFn, class ResFn>
void run_q2q_layer3(Q2QResnet18Context& ctx, ConvFn&& conv_fn, ResFn&& res_fn, const QTensorNHWC& in) {
    conv_fn(in,          ctx.p30_c1, *ctx.c30_conv1, *ctx.bn30_1, true,  ACT_SCALE, ctx.l30_mid, ctx.H3, ctx.W3);
    conv_fn(ctx.l30_mid, ctx.p30_c2, *ctx.c30_conv2, *ctx.bn30_2, false, ACT_SCALE, ctx.l30_conv2_out, ctx.H3, ctx.W3);
    conv_fn(in,          ctx.p30_ds, *ctx.c30_ds,    *ctx.bn30_ds,false, ACT_SCALE, ctx.l30_identity, ctx.H3, ctx.W3);
    res_fn(ctx.l30_conv2_out, ctx.l30_identity, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, ctx.l30_out);
    conv_fn(ctx.l30_out, ctx.p31_c1, *ctx.c31_conv1, *ctx.bn31_1, true,  ACT_SCALE, ctx.l31_mid, ctx.H3, ctx.W3);
    conv_fn(ctx.l31_mid, ctx.p31_c2, *ctx.c31_conv2, *ctx.bn31_2, false, ACT_SCALE, ctx.l31_conv2_out, ctx.H3, ctx.W3);
    res_fn(ctx.l31_conv2_out, ctx.l30_out, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, ctx.l31_out);
}

template <class ConvFn, class ResFn>
void run_q2q_layer4(Q2QResnet18Context& ctx, ConvFn&& conv_fn, ResFn&& res_fn, const QTensorNHWC& in) {
    conv_fn(in,          ctx.p40_c1, *ctx.c40_conv1, *ctx.bn40_1, true,  ACT_SCALE, ctx.l40_mid, ctx.H4, ctx.W4);
    conv_fn(ctx.l40_mid, ctx.p40_c2, *ctx.c40_conv2, *ctx.bn40_2, false, ACT_SCALE, ctx.l40_conv2_out, ctx.H4, ctx.W4);
    conv_fn(in,          ctx.p40_ds, *ctx.c40_ds,    *ctx.bn40_ds,false, ACT_SCALE, ctx.l40_identity, ctx.H4, ctx.W4);
    res_fn(ctx.l40_conv2_out, ctx.l40_identity, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, ctx.l40_out);
    conv_fn(ctx.l40_out, ctx.p41_c1, *ctx.c41_conv1, *ctx.bn41_1, true,  ACT_SCALE, ctx.l41_mid, ctx.H4, ctx.W4);
    conv_fn(ctx.l41_mid, ctx.p41_c2, *ctx.c41_conv2, *ctx.bn41_2, false, ACT_SCALE, ctx.l41_conv2_out, ctx.H4, ctx.W4);
    res_fn(ctx.l41_conv2_out, ctx.l40_out, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, ctx.l41_out);
}

template <class ConvFn, class ResFn>
void run_q2q_full(Q2QResnet18Context& ctx, ConvFn&& conv_fn, ResFn&& res_fn, const Tensor& golden_input) {
    quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.input_q8);
    q2q_conv1_im2col_bn_relu_requant(ctx.input_q8, ctx.p_c1, *ctx.conv1, *ctx.bn1, ACT_SCALE, ctx.conv1_out, ctx.OH1, ctx.OW1);
    q8_maxpool_3x3_s2_p1(ctx.conv1_out, ctx.OHp, ctx.OWp, ctx.pool_out);
    run_q2q_layer1(ctx, conv_fn, res_fn, ctx.pool_out);
    run_q2q_layer2(ctx, conv_fn, res_fn, ctx.l11_out);
    run_q2q_layer3(ctx, conv_fn, res_fn, ctx.l21_out);
    run_q2q_layer4(ctx, conv_fn, res_fn, ctx.l31_out);
    dequantize_q8_nhwc_to_fp32_nchw(ctx.l41_out, ctx.H4, ctx.W4, ctx.OC4, ctx.dequant_out);
    adaptive_avgpool_1x1_into(ctx.dequant_out, ctx.avgpool_out);
    linear_into(ctx.avgpool_out, *ctx.fc, ctx.fc_out);
}

} // namespace q2q
