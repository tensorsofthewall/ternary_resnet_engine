#include "q2q_resnet18.hpp"

namespace q2q {

Q2QResnet18Context build_q2q_resnet18_context(const Model& model) {
    Q2QResnet18Context ctx;

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

    // ---- panels ----
    ctx.p_c1  = build_conv1_im2col_panel(*ctx.conv1);
    ctx.p10_c1 = build_oc4_k32_panel(*ctx.c10_conv1);
    ctx.p10_c2 = build_oc4_k32_panel(*ctx.c10_conv2);
    ctx.p11_c1 = build_oc4_k32_panel(*ctx.c11_conv1);
    ctx.p11_c2 = build_oc4_k32_panel(*ctx.c11_conv2);
    ctx.p20_c1 = build_oc4_k32_panel(*ctx.c20_conv1);
    ctx.p20_c2 = build_oc4_k32_panel(*ctx.c20_conv2);
    ctx.p20_ds = build_oc4_k32_panel(*ctx.c20_ds);
    ctx.p21_c1 = build_oc4_k32_panel(*ctx.c21_conv1);
    ctx.p21_c2 = build_oc4_k32_panel(*ctx.c21_conv2);
    ctx.p30_c1 = build_oc4_k32_panel(*ctx.c30_conv1);
    ctx.p30_c2 = build_oc4_k32_panel(*ctx.c30_conv2);
    ctx.p30_ds = build_oc4_k32_panel(*ctx.c30_ds);
    ctx.p31_c1 = build_oc4_k32_panel(*ctx.c31_conv1);
    ctx.p31_c2 = build_oc4_k32_panel(*ctx.c31_conv2);
    ctx.p40_c1 = build_oc4_k32_panel(*ctx.c40_conv1);
    ctx.p40_c2 = build_oc4_k32_panel(*ctx.c40_conv2);
    ctx.p40_ds = build_oc4_k32_panel(*ctx.c40_ds);
    ctx.p41_c1 = build_oc4_k32_panel(*ctx.c41_conv1);
    ctx.p41_c2 = build_oc4_k32_panel(*ctx.c41_conv2);

    // ---- shared scratch tensors (safe to share across the two sequential,
    // self-contained phases -- no cross-phase dependency) ----
    ctx.input_q8  = make_q8_tensor(ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.conv1->pad_h, ACT_SCALE);
    ctx.conv1_out = make_q8_tensor(ctx.OH1, ctx.OW1, ctx.OC_C1, PAD_Q8, ACT_SCALE);
    ctx.pool_out  = make_q8_tensor(ctx.OHp, ctx.OWp, ctx.OC_C1, PAD_Q8, ACT_SCALE);
    ctx.l10_mid       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, PAD_Q8, ACT_SCALE);
    ctx.l10_conv2_out = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, PAD_Q8, ACT_SCALE);
    ctx.l10_out       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, PAD_Q8, ACT_SCALE);
    ctx.l11_mid       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, PAD_Q8, ACT_SCALE);
    ctx.l11_conv2_out = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, PAD_Q8, ACT_SCALE);
    ctx.l11_out       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, PAD_Q8, ACT_SCALE);
    ctx.l20_mid       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l20_conv2_out = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l20_identity  = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l20_out       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l21_mid       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l21_conv2_out = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l21_out       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, PAD_Q8, ACT_SCALE);
    ctx.l30_mid       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l30_conv2_out = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l30_identity  = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l30_out       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l31_mid       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l31_conv2_out = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l31_out       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, PAD_Q8, ACT_SCALE);
    ctx.l40_mid       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);
    ctx.l40_conv2_out = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);
    ctx.l40_identity  = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);
    ctx.l40_out       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);
    ctx.l41_mid       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);
    ctx.l41_conv2_out = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);
    ctx.l41_out       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, PAD_Q8, ACT_SCALE);

    ctx.dequant_out = Tensor(ctx.OC4, ctx.H4, ctx.W4);
    ctx.avgpool_out = Tensor(ctx.OC4, 1, 1);
    ctx.fc_out      = Tensor(1000, 1, 1);

    return ctx;
}

} // namespace q2q
