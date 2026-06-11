#pragma once

#include "block32_conv1.hpp"
#include "block32_plan.hpp"
#include "model.hpp"
#include "ops.hpp"
#include "q2q_tensor.hpp"
#include "tensor.hpp"

namespace nhwc_block32 {

// Holds every per-layer Conv2DParams/BNFusedParams (as non-owning pointers
// into the Model -- the Model must outlive this context), every prebuilt
// block32 plan (including the conv1 gather plan), every scratch
// q2q::QTensorNHWC, and the basic_engine Tensor outputs (dequant/avgpool/fc)
// used by the dequant+head tail. Unlike q2q::Q2QResnet18Context, there is
// only ONE kernel pair (conv_block32_oc4_avx2 + block32_residual_add_relu_avx2,
// plus conv_block32_conv1_oc4_avx2 for conv1), so the orchestration functions
// below are plain (non-template) functions.
struct Block32Resnet18Context {
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

    // ---- block32 plans ----
    Conv1Block32Plan p_conv1;
    Block32ConvPlan p10_c1, p10_c2, p11_c1, p11_c2;
    Block32ConvPlan p20_c1, p20_c2, p20_ds, p21_c1, p21_c2;
    Block32ConvPlan p30_c1, p30_c2, p30_ds, p31_c1, p31_c2;
    Block32ConvPlan p40_c1, p40_c2, p40_ds, p41_c1, p41_c2;

    // ---- scratch tensors ----
    q2q::QTensorNHWC input_q8, conv1_out, pool_out;
    q2q::QTensorNHWC l10_mid, l10_conv2_out, l10_out;
    q2q::QTensorNHWC l11_mid, l11_conv2_out, l11_out;
    q2q::QTensorNHWC l20_mid, l20_conv2_out, l20_identity, l20_out;
    q2q::QTensorNHWC l21_mid, l21_conv2_out, l21_out;
    q2q::QTensorNHWC l30_mid, l30_conv2_out, l30_identity, l30_out;
    q2q::QTensorNHWC l31_mid, l31_conv2_out, l31_out;
    q2q::QTensorNHWC l40_mid, l40_conv2_out, l40_identity, l40_out;
    q2q::QTensorNHWC l41_mid, l41_conv2_out, l41_out;

    Tensor dequant_out, avgpool_out, fc_out;
};

// Looks up every Conv2DParams/BNFusedParams, derives dimensions, builds every
// block32 plan (including the conv1 gather plan), and allocates every scratch
// tensor -- everything that runs once before timing loops.
Block32Resnet18Context build_block32_resnet18_context(const Model& model);

// Runs the layer1 pair of basic residual blocks (10, 11; stride 1, no
// downsample) on `in`, writing the final activation to ctx.l11_out.
void run_block32_layer1(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in);

// Runs the layer2 pair of residual blocks (20: stride-2 with a downsample
// projection; 21: stride 1) on `in`, writing the final activation to
// ctx.l21_out.
void run_block32_layer2(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in);

// Runs the layer3 pair of residual blocks (30: stride-2 with a downsample
// projection; 31: stride 1) on `in`, writing the final activation to
// ctx.l31_out.
void run_block32_layer3(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in);

// Runs the layer4 pair of residual blocks (40: stride-2 with a downsample
// projection; 41: stride 1) on `in`, writing the final activation to
// ctx.l41_out.
void run_block32_layer4(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in);

// End-to-end forward pass: quantizes `golden_input` (FP32 NCHW) to q8 NHWC,
// runs conv1 + maxpool + all four residual stacks, then dequantizes the final
// activation and runs the avgpool+FC head, leaving logits in ctx.fc_out.
void run_block32_full(Block32Resnet18Context& ctx, const Tensor& golden_input);

} // namespace nhwc_block32
