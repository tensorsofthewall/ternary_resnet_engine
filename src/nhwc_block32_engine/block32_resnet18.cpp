#include "block32_resnet18.hpp"

#include "block32_conv_avx2.hpp"
#include "block32_residual.hpp"
#include "nhwc_common.hpp"
#include "q2q_pool.hpp"
#include "q2q_quant.hpp"

namespace nhwc_block32 {

namespace {
constexpr int BORDER = 1;
} // namespace

Block32Resnet18Context build_block32_resnet18_context(const Model& model) {
    Block32Resnet18Context ctx;

    ctx.conv1 = &model.convs.at("conv1");
    ctx.bn1   = &model.fused_bns.at("bn1");
    ctx.c10_conv1 = &model.convs.at("layer1.0.conv1");
    ctx.c10_conv2 = &model.convs.at("layer1.0.conv2");
    ctx.bn10_1    = &model.fused_bns.at("layer1.0.bn1");
    ctx.bn10_2    = &model.fused_bns.at("layer1.0.bn2");
    ctx.c11_conv1 = &model.convs.at("layer1.1.conv1");
    ctx.c11_conv2 = &model.convs.at("layer1.1.conv2");
    ctx.bn11_1    = &model.fused_bns.at("layer1.1.bn1");
    ctx.bn11_2    = &model.fused_bns.at("layer1.1.bn2");
    ctx.c20_conv1 = &model.convs.at("layer2.0.conv1");
    ctx.c20_conv2 = &model.convs.at("layer2.0.conv2");
    ctx.c20_ds    = &model.convs.at("layer2.0.downsample.0");
    ctx.bn20_1    = &model.fused_bns.at("layer2.0.bn1");
    ctx.bn20_2    = &model.fused_bns.at("layer2.0.bn2");
    ctx.bn20_ds   = &model.fused_bns.at("layer2.0.downsample.1");
    ctx.c21_conv1 = &model.convs.at("layer2.1.conv1");
    ctx.c21_conv2 = &model.convs.at("layer2.1.conv2");
    ctx.bn21_1    = &model.fused_bns.at("layer2.1.bn1");
    ctx.bn21_2    = &model.fused_bns.at("layer2.1.bn2");
    ctx.c30_conv1 = &model.convs.at("layer3.0.conv1");
    ctx.c30_conv2 = &model.convs.at("layer3.0.conv2");
    ctx.c30_ds    = &model.convs.at("layer3.0.downsample.0");
    ctx.bn30_1    = &model.fused_bns.at("layer3.0.bn1");
    ctx.bn30_2    = &model.fused_bns.at("layer3.0.bn2");
    ctx.bn30_ds   = &model.fused_bns.at("layer3.0.downsample.1");
    ctx.c31_conv1 = &model.convs.at("layer3.1.conv1");
    ctx.c31_conv2 = &model.convs.at("layer3.1.conv2");
    ctx.bn31_1    = &model.fused_bns.at("layer3.1.bn1");
    ctx.bn31_2    = &model.fused_bns.at("layer3.1.bn2");
    ctx.c40_conv1 = &model.convs.at("layer4.0.conv1");
    ctx.c40_conv2 = &model.convs.at("layer4.0.conv2");
    ctx.c40_ds    = &model.convs.at("layer4.0.downsample.0");
    ctx.bn40_1    = &model.fused_bns.at("layer4.0.bn1");
    ctx.bn40_2    = &model.fused_bns.at("layer4.0.bn2");
    ctx.bn40_ds   = &model.fused_bns.at("layer4.0.downsample.1");
    ctx.c41_conv1 = &model.convs.at("layer4.1.conv1");
    ctx.c41_conv2 = &model.convs.at("layer4.1.conv2");
    ctx.bn41_1    = &model.fused_bns.at("layer4.1.bn1");
    ctx.bn41_2    = &model.fused_bns.at("layer4.1.bn2");
    ctx.fc        = &model.linears.at("fc");

    ctx.IC_C1 = ctx.conv1->in_channels;
    ctx.OC_C1 = ctx.conv1->out_channels;
    ctx.H_IN = 224;
    ctx.W_IN = 224;
    ctx.OH1 = (ctx.H_IN + 2 * ctx.conv1->pad_h - ctx.conv1->kh) / ctx.conv1->stride_h + 1;
    ctx.OW1 = (ctx.W_IN + 2 * ctx.conv1->pad_w - ctx.conv1->kw) / ctx.conv1->stride_w + 1;
    ctx.OHp = (ctx.OH1 + 2 - 3) / 2 + 1;
    ctx.OWp = (ctx.OW1 + 2 - 3) / 2 + 1;
    ctx.OC1 = ctx.c10_conv1->out_channels;
    ctx.OC2 = ctx.c20_conv1->out_channels;
    ctx.OC3 = ctx.c30_conv1->out_channels;
    ctx.OC4 = ctx.c40_conv1->out_channels;
    ctx.H1 = ctx.OHp; ctx.W1 = ctx.OWp;
    ctx.H2 = 28; ctx.W2 = 28;
    ctx.H3 = 14; ctx.W3 = 14;
    ctx.H4 = 7;  ctx.W4 = 7;

    // ---- block32 plans ----
    ctx.p_conv1 = build_conv1_block32_plan(*ctx.conv1);
    ctx.p10_c1 = build_block32_plan(*ctx.c10_conv1);
    ctx.p10_c2 = build_block32_plan(*ctx.c10_conv2);
    ctx.p11_c1 = build_block32_plan(*ctx.c11_conv1);
    ctx.p11_c2 = build_block32_plan(*ctx.c11_conv2);
    ctx.p20_c1 = build_block32_plan(*ctx.c20_conv1);
    ctx.p20_c2 = build_block32_plan(*ctx.c20_conv2);
    ctx.p20_ds = build_block32_plan(*ctx.c20_ds);
    ctx.p21_c1 = build_block32_plan(*ctx.c21_conv1);
    ctx.p21_c2 = build_block32_plan(*ctx.c21_conv2);
    ctx.p30_c1 = build_block32_plan(*ctx.c30_conv1);
    ctx.p30_c2 = build_block32_plan(*ctx.c30_conv2);
    ctx.p30_ds = build_block32_plan(*ctx.c30_ds);
    ctx.p31_c1 = build_block32_plan(*ctx.c31_conv1);
    ctx.p31_c2 = build_block32_plan(*ctx.c31_conv2);
    ctx.p40_c1 = build_block32_plan(*ctx.c40_conv1);
    ctx.p40_c2 = build_block32_plan(*ctx.c40_conv2);
    ctx.p40_ds = build_block32_plan(*ctx.c40_ds);
    ctx.p41_c1 = build_block32_plan(*ctx.c41_conv1);
    ctx.p41_c2 = build_block32_plan(*ctx.c41_conv2);

    // ---- scratch tensors ----
    ctx.input_q8  = q2q::make_q8_tensor(ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.conv1->pad_h, ACT_SCALE);
    ctx.conv1_out = q2q::make_q8_tensor(ctx.OH1, ctx.OW1, ctx.OC_C1, BORDER, ACT_SCALE);
    ctx.pool_out  = q2q::make_q8_tensor(ctx.OHp, ctx.OWp, ctx.OC_C1, BORDER, ACT_SCALE);

    ctx.l10_mid       = q2q::make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
    ctx.l10_conv2_out = q2q::make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
    ctx.l10_out       = q2q::make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
    ctx.l11_mid       = q2q::make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
    ctx.l11_conv2_out = q2q::make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
    ctx.l11_out       = q2q::make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);

    ctx.l20_mid       = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
    ctx.l20_conv2_out = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
    ctx.l20_identity  = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
    ctx.l20_out       = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
    ctx.l21_mid       = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
    ctx.l21_conv2_out = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
    ctx.l21_out       = q2q::make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);

    ctx.l30_mid       = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
    ctx.l30_conv2_out = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
    ctx.l30_identity  = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
    ctx.l30_out       = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
    ctx.l31_mid       = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
    ctx.l31_conv2_out = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
    ctx.l31_out       = q2q::make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);

    ctx.l40_mid       = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
    ctx.l40_conv2_out = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
    ctx.l40_identity  = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
    ctx.l40_out       = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
    ctx.l41_mid       = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
    ctx.l41_conv2_out = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
    ctx.l41_out       = q2q::make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);

    ctx.dequant_out = Tensor(ctx.OC4, ctx.H4, ctx.W4);
    ctx.avgpool_out = Tensor(ctx.OC4, 1, 1);
    ctx.fc_out      = Tensor(1000, 1, 1);

    return ctx;
}

void run_block32_layer1(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in) {
    conv_block32_oc4_avx2(ctx.p10_c1, *ctx.bn10_1, in,          true,  ctx.l10_mid, ctx.H1, ctx.W1);
    conv_block32_oc4_avx2(ctx.p10_c2, *ctx.bn10_2, ctx.l10_mid, false, ctx.l10_conv2_out, ctx.H1, ctx.W1);
    block32_residual_add_relu_avx2(ctx.l10_conv2_out, in, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, ctx.l10_out);
    conv_block32_oc4_avx2(ctx.p11_c1, *ctx.bn11_1, ctx.l10_out, true,  ctx.l11_mid, ctx.H1, ctx.W1);
    conv_block32_oc4_avx2(ctx.p11_c2, *ctx.bn11_2, ctx.l11_mid, false, ctx.l11_conv2_out, ctx.H1, ctx.W1);
    block32_residual_add_relu_avx2(ctx.l11_conv2_out, ctx.l10_out, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, ctx.l11_out);
}

void run_block32_layer2(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in) {
    conv_block32_oc4_avx2(ctx.p20_c1, *ctx.bn20_1, in,          true,  ctx.l20_mid, ctx.H2, ctx.W2);
    conv_block32_oc4_avx2(ctx.p20_c2, *ctx.bn20_2, ctx.l20_mid, false, ctx.l20_conv2_out, ctx.H2, ctx.W2);
    conv_block32_oc4_avx2(ctx.p20_ds, *ctx.bn20_ds, in,         false, ctx.l20_identity, ctx.H2, ctx.W2);
    block32_residual_add_relu_avx2(ctx.l20_conv2_out, ctx.l20_identity, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, ctx.l20_out);
    conv_block32_oc4_avx2(ctx.p21_c1, *ctx.bn21_1, ctx.l20_out, true,  ctx.l21_mid, ctx.H2, ctx.W2);
    conv_block32_oc4_avx2(ctx.p21_c2, *ctx.bn21_2, ctx.l21_mid, false, ctx.l21_conv2_out, ctx.H2, ctx.W2);
    block32_residual_add_relu_avx2(ctx.l21_conv2_out, ctx.l20_out, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, ctx.l21_out);
}

void run_block32_layer3(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in) {
    conv_block32_oc4_avx2(ctx.p30_c1, *ctx.bn30_1, in,          true,  ctx.l30_mid, ctx.H3, ctx.W3);
    conv_block32_oc4_avx2(ctx.p30_c2, *ctx.bn30_2, ctx.l30_mid, false, ctx.l30_conv2_out, ctx.H3, ctx.W3);
    conv_block32_oc4_avx2(ctx.p30_ds, *ctx.bn30_ds, in,         false, ctx.l30_identity, ctx.H3, ctx.W3);
    block32_residual_add_relu_avx2(ctx.l30_conv2_out, ctx.l30_identity, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, ctx.l30_out);
    conv_block32_oc4_avx2(ctx.p31_c1, *ctx.bn31_1, ctx.l30_out, true,  ctx.l31_mid, ctx.H3, ctx.W3);
    conv_block32_oc4_avx2(ctx.p31_c2, *ctx.bn31_2, ctx.l31_mid, false, ctx.l31_conv2_out, ctx.H3, ctx.W3);
    block32_residual_add_relu_avx2(ctx.l31_conv2_out, ctx.l30_out, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, ctx.l31_out);
}

void run_block32_layer4(Block32Resnet18Context& ctx, const q2q::QTensorNHWC& in) {
    conv_block32_oc4_avx2(ctx.p40_c1, *ctx.bn40_1, in,          true,  ctx.l40_mid, ctx.H4, ctx.W4);
    conv_block32_oc4_avx2(ctx.p40_c2, *ctx.bn40_2, ctx.l40_mid, false, ctx.l40_conv2_out, ctx.H4, ctx.W4);
    conv_block32_oc4_avx2(ctx.p40_ds, *ctx.bn40_ds, in,         false, ctx.l40_identity, ctx.H4, ctx.W4);
    block32_residual_add_relu_avx2(ctx.l40_conv2_out, ctx.l40_identity, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, ctx.l40_out);
    conv_block32_oc4_avx2(ctx.p41_c1, *ctx.bn41_1, ctx.l40_out, true,  ctx.l41_mid, ctx.H4, ctx.W4);
    conv_block32_oc4_avx2(ctx.p41_c2, *ctx.bn41_2, ctx.l41_mid, false, ctx.l41_conv2_out, ctx.H4, ctx.W4);
    block32_residual_add_relu_avx2(ctx.l41_conv2_out, ctx.l40_out, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, ctx.l41_out);
}

void run_block32_full(Block32Resnet18Context& ctx, const Tensor& golden_input) {
    q2q::quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.input_q8);
    conv_block32_conv1_oc4_avx2(ctx.p_conv1, *ctx.bn1, ctx.input_q8, true, ctx.conv1_out, ctx.OH1, ctx.OW1);
    q2q::q8_maxpool_3x3_s2_p1(ctx.conv1_out, ctx.OHp, ctx.OWp, ctx.pool_out);
    run_block32_layer1(ctx, ctx.pool_out);
    run_block32_layer2(ctx, ctx.l11_out);
    run_block32_layer3(ctx, ctx.l21_out);
    run_block32_layer4(ctx, ctx.l31_out);
    q2q::dequantize_q8_nhwc_to_fp32_nchw(ctx.l41_out, ctx.H4, ctx.W4, ctx.OC4, ctx.dequant_out);
    adaptive_avgpool_1x1_into(ctx.dequant_out, ctx.avgpool_out);
    linear_into(ctx.avgpool_out, *ctx.fc, ctx.fc_out);
}

} // namespace nhwc_block32
