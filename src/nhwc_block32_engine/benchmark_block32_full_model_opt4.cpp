// benchmark_block32_full_model_opt4.cpp -- AVX2-only (no AVX-VNNI) block32:
// full ResNet18 forward pass, Phase 15 on top of benchmark_block32_full_model_opt3.
//
// Phase 15: replaces every repeated 3x3/stride1/pad1 conv in layer2/3/4
// (9 convs total: {layer2.0.conv2, layer2.1.conv1, layer2.1.conv2},
// {layer3.0.conv2, layer3.1.conv1, layer3.1.conv2},
// {layer4.0.conv2, layer4.1.conv1, layer4.1.conv2}) with
// panel_im2col_gemm_oc4_n2_avx2 (im2col_gemm.hpp/.cpp), which builds a small
// A_panel[2][K_pad] per pair of adjacent output columns directly from the q8
// NHWC input and runs an AVX2 OC4 GEMM microkernel
// (_mm256_maddubs_epi16 + _mm256_madd_epi16, accumulators kept across all
// K_pad/32 blocks, hsum once, epilogue_oc4_avx2 for store) against a flat
// 32-byte-aligned PackedBPanel.
//
// Standalone benchmarks (benchmark_block32_im2col_layer2/3/4) showed
// panel_im2col_n2 beating the specialized conv3x3s1p1_*_oc4_n2*_avx2 kernels
// by a wide margin (0/0 mismatches in every case):
//   layer2 (3 convs): -38.9% on layer2.0.conv2 standalone
//   layer3 (3 convs): -1.53 ms total (vs >=0.5 ms gate)
//   layer4 (3 convs): -1.53 ms total (vs >=0.4 ms gate)
//
// Everything else (conv1 v2_split, layer1 N4 OC-major, layer2/3/4 conv1 +
// downsample + residuals, dequant/avgpool/fc, all OC4 epilogues) is UNCHANGED
// from opt3.
//
// Usage: ./benchmark_block32_full_model_opt4 <model.bin>

#include "block32_conv1.hpp"
#include "block32_conv1_opt.hpp"
#include "block32_conv_avx2.hpp"
#include "block32_layer1_opt.hpp"
#include "block32_layer234_opt.hpp"
#include "block32_plan.hpp"
#include "block32_residual.hpp"
#include "block32_resnet18.hpp"
#include "im2col_gemm.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include "nhwc_common.hpp"
#include "ops.hpp"
#include "q2q_common.hpp"
#include "q2q_pool.hpp"
#include "q2q_quant.hpp"
#include "q2q_tensor.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace nhwc_block32;
using q2q::make_q8_tensor;
using q2q::QTensorNHWC;

namespace {

constexpr int BORDER = 1;

// Baselines (all measured live, 1T, this session unless noted):
//   block32 baseline full model (benchmark_block32_full_model, total_profiled_ms med)
constexpr double BLOCK32_BASELINE_FULL_MS = 43.3364;
//   basic_engine full model (benchmark_q2q_full_prefix_optimized, "basic_engine full model" med)
constexpr double BASIC_ENGINE_FULL_MS = 45.4875;
//   q2q VNNI optimized full prefix (benchmark_q2q_full_prefix_optimized, optimized_total_ms med)
constexpr double Q2Q_VNNI_FULL_MS = 15.6768;
//   benchmark_block32_full_model_opt2, total_profiled_ms med, after Phase 14
//   Step 2 (vectorized OC4 epilogue, rebuilt opt2 binary, before layer4 spec.)
constexpr double PHASE14_STEP2_MS = 31.8198;
//   benchmark_block32_full_model_opt3, total_profiled_ms med (Phase 14 final,
//   re-measured this session immediately before the opt4 run below).
constexpr double PHASE14_OPT3_MS = 29.1921;

// Phase 15 decision gates (single-thread AVX2-only full model):
//   <= 26.5 ms: excellent; stop single-thread work.
//   26.5-27.5 ms: strong; stop and document.
//   27.5-28.2 ms: keep if code is clean.
//   > 28.2 ms: benchmark-only, keep opt3 as final.
constexpr double GATE_EXCELLENT_MS = 26.5;
constexpr double GATE_STRONG_MS = 27.5;
constexpr double GATE_KEEP_MS = 28.2;

auto compare = [](const char* label, const QTensorNHWC& a, const QTensorNHWC& b) {
    size_t mismatches = 0;
    int max_abs_diff = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const int diff = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (diff != 0) ++mismatches;
        max_abs_diff = std::max(max_abs_diff, diff);
    }
    std::cout << "Correctness (" << label << "): " << mismatches << " / " << a.data.size()
              << " mismatches, max_abs_diff=" << max_abs_diff << "\n";
    return mismatches;
};

// layer1 with the Phase 14 Step4 OC4_N4 OC-major conv kernel for all four
// convs; residuals unchanged (block32_residual_add_relu_avx2). Identical to opt3.
void run_block32_layer1_opt(Block32Resnet18Context& ctx, const QTensorNHWC& in) {
    conv_block32_n4_oc_major_avx2(ctx.p10_c1, *ctx.bn10_1, in,          true,  ctx.l10_mid, ctx.H1, ctx.W1);
    conv_block32_n4_oc_major_avx2(ctx.p10_c2, *ctx.bn10_2, ctx.l10_mid, false, ctx.l10_conv2_out, ctx.H1, ctx.W1);
    block32_residual_add_relu_avx2(ctx.l10_conv2_out, in, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, ctx.l10_out);
    conv_block32_n4_oc_major_avx2(ctx.p11_c1, *ctx.bn11_1, ctx.l10_out, true,  ctx.l11_mid, ctx.H1, ctx.W1);
    conv_block32_n4_oc_major_avx2(ctx.p11_c2, *ctx.bn11_2, ctx.l11_mid, false, ctx.l11_conv2_out, ctx.H1, ctx.W1);
    block32_residual_add_relu_avx2(ctx.l11_conv2_out, ctx.l10_out, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, ctx.l11_out);
}

// Phase 15: layer2 with panel_im2col_gemm_oc4_n2_avx2 for the repeated
// 128->128/3x3/s1/p1/28x28 convs (conv2 of block 0, both convs of block 1);
// conv1 of block 0, the downsample projection, and the residual adds are
// unchanged. ic_blocks=4 (128/32).
void run_block32_layer2_opt4(
    Block32Resnet18Context& ctx, const QTensorNHWC& in,
    const PackedBPanel& p20_c2_B, const PackedBPanel& p21_c1_B, const PackedBPanel& p21_c2_B)
{
    constexpr int IC_BLOCKS = 4;
    conv_block32_oc4_avx2(ctx.p20_c1, *ctx.bn20_1, in, true, ctx.l20_mid, ctx.H2, ctx.W2);
    panel_im2col_gemm_oc4_n2_avx2(p20_c2_B, ctx.l20_mid, IC_BLOCKS, *ctx.bn20_2, false, ctx.l20_conv2_out, ctx.H2, ctx.W2);
    conv_block32_oc4_avx2(ctx.p20_ds, *ctx.bn20_ds, in, false, ctx.l20_identity, ctx.H2, ctx.W2);
    block32_residual_add_relu_avx2(ctx.l20_conv2_out, ctx.l20_identity, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, ctx.l20_out);
    panel_im2col_gemm_oc4_n2_avx2(p21_c1_B, ctx.l20_out, IC_BLOCKS, *ctx.bn21_1, true, ctx.l21_mid, ctx.H2, ctx.W2);
    panel_im2col_gemm_oc4_n2_avx2(p21_c2_B, ctx.l21_mid, IC_BLOCKS, *ctx.bn21_2, false, ctx.l21_conv2_out, ctx.H2, ctx.W2);
    block32_residual_add_relu_avx2(ctx.l21_conv2_out, ctx.l20_out, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, ctx.l21_out);
}

// Phase 15: layer3 with panel_im2col_gemm_oc4_n2_avx2 for the repeated
// 256->256/3x3/s1/p1/14x14 convs; conv1 of block 0, the downsample
// projection, and the residual adds are unchanged. ic_blocks=8 (256/32).
void run_block32_layer3_opt4(
    Block32Resnet18Context& ctx, const QTensorNHWC& in,
    const PackedBPanel& p30_c2_B, const PackedBPanel& p31_c1_B, const PackedBPanel& p31_c2_B)
{
    constexpr int IC_BLOCKS = 8;
    conv_block32_oc4_avx2(ctx.p30_c1, *ctx.bn30_1, in, true, ctx.l30_mid, ctx.H3, ctx.W3);
    panel_im2col_gemm_oc4_n2_avx2(p30_c2_B, ctx.l30_mid, IC_BLOCKS, *ctx.bn30_2, false, ctx.l30_conv2_out, ctx.H3, ctx.W3);
    conv_block32_oc4_avx2(ctx.p30_ds, *ctx.bn30_ds, in, false, ctx.l30_identity, ctx.H3, ctx.W3);
    block32_residual_add_relu_avx2(ctx.l30_conv2_out, ctx.l30_identity, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, ctx.l30_out);
    panel_im2col_gemm_oc4_n2_avx2(p31_c1_B, ctx.l30_out, IC_BLOCKS, *ctx.bn31_1, true, ctx.l31_mid, ctx.H3, ctx.W3);
    panel_im2col_gemm_oc4_n2_avx2(p31_c2_B, ctx.l31_mid, IC_BLOCKS, *ctx.bn31_2, false, ctx.l31_conv2_out, ctx.H3, ctx.W3);
    block32_residual_add_relu_avx2(ctx.l31_conv2_out, ctx.l30_out, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, ctx.l31_out);
}

// Phase 15: layer4 with panel_im2col_gemm_oc4_n2_avx2 for the repeated
// 512->512/3x3/s1/p1/7x7 convs; conv1 of block 0, the downsample projection,
// and the residual adds are unchanged. ic_blocks=16 (512/32).
void run_block32_layer4_opt4(
    Block32Resnet18Context& ctx, const QTensorNHWC& in,
    const PackedBPanel& p40_c2_B, const PackedBPanel& p41_c1_B, const PackedBPanel& p41_c2_B)
{
    constexpr int IC_BLOCKS = 16;
    conv_block32_oc4_avx2(ctx.p40_c1, *ctx.bn40_1, in, true, ctx.l40_mid, ctx.H4, ctx.W4);
    panel_im2col_gemm_oc4_n2_avx2(p40_c2_B, ctx.l40_mid, IC_BLOCKS, *ctx.bn40_2, false, ctx.l40_conv2_out, ctx.H4, ctx.W4);
    conv_block32_oc4_avx2(ctx.p40_ds, *ctx.bn40_ds, in, false, ctx.l40_identity, ctx.H4, ctx.W4);
    block32_residual_add_relu_avx2(ctx.l40_conv2_out, ctx.l40_identity, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, ctx.l40_out);
    panel_im2col_gemm_oc4_n2_avx2(p41_c1_B, ctx.l40_out, IC_BLOCKS, *ctx.bn41_1, true, ctx.l41_mid, ctx.H4, ctx.W4);
    panel_im2col_gemm_oc4_n2_avx2(p41_c2_B, ctx.l41_mid, IC_BLOCKS, *ctx.bn41_2, false, ctx.l41_conv2_out, ctx.H4, ctx.W4);
    block32_residual_add_relu_avx2(ctx.l41_conv2_out, ctx.l40_out, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, ctx.l41_out);
}

struct Layer2Panels {
    PackedBPanel p20_c2, p21_c1, p21_c2;
};
struct Layer3Panels {
    PackedBPanel p30_c2, p31_c1, p31_c2;
};
struct Layer4Panels {
    PackedBPanel p40_c2, p41_c1, p41_c2;
};

// Full forward pass: Phase 7 conv1 (v2_split) + Phase 14 Step4 layer1
// (n4_oc_major) + Phase 15 layer2/3/4 (panel_im2col_gemm_oc4_n2_avx2 for the
// repeated 3x3/s1/p1 convs) + unchanged dequant/avgpool/fc.
void run_block32_full_opt4(
    Block32Resnet18Context& ctx, const Conv1Block32PlanV2& p_conv1_v2,
    const Layer2Panels& l2, const Layer3Panels& l3, const Layer4Panels& l4, const Tensor& golden_input)
{
    q2q::quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.input_q8);
    conv_block32_conv1_oc4_v2_split_avx2(p_conv1_v2, *ctx.bn1, ctx.input_q8, true, ctx.conv1_out, ctx.OH1, ctx.OW1);
    q2q::q8_maxpool_3x3_s2_p1(ctx.conv1_out, ctx.OHp, ctx.OWp, ctx.pool_out);
    run_block32_layer1_opt(ctx, ctx.pool_out);
    run_block32_layer2_opt4(ctx, ctx.l11_out, l2.p20_c2, l2.p21_c1, l2.p21_c2);
    run_block32_layer3_opt4(ctx, ctx.l21_out, l3.p30_c2, l3.p31_c1, l3.p31_c2);
    run_block32_layer4_opt4(ctx, ctx.l31_out, l4.p40_c2, l4.p41_c1, l4.p41_c2);
    q2q::dequantize_q8_nhwc_to_fp32_nchw(ctx.l41_out, ctx.H4, ctx.W4, ctx.OC4, ctx.dequant_out);
    adaptive_avgpool_1x1_into(ctx.dequant_out, ctx.avgpool_out);
    linear_into(ctx.avgpool_out, *ctx.fc, ctx.fc_out);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);
        Block32Resnet18Context ctx = build_block32_resnet18_context(model);
        const Conv1Block32PlanV2 p_conv1_v2 = build_conv1_block32_plan_v2(*ctx.conv1);

        const Layer2Panels l2{
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p20_c2)),
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p21_c1)),
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p21_c2)),
        };
        const Layer3Panels l3{
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p30_c2)),
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p31_c1)),
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p31_c2)),
        };
        const Layer4Panels l4{
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p40_c2)),
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p41_c1)),
            build_packed_b_panel(build_conv3x3s1p1_panel_plan(ctx.p41_c2)),
        };

        std::cout << "=== benchmark_block32_full_model_opt4: full ResNet18 forward (AVX2 block32, no VNNI) ===\n";
        std::cout << "  conv1: conv_block32_conv1_oc4_v2_split_avx2 (Phase 7 row-memcpy variant)\n";
        std::cout << "  layer1: conv_block32_n4_oc_major_avx2 (Phase 14 Step4 OC4_N4 OC-major variant)\n";
        std::cout << "  layer2/3/4 repeated 3x3 s1p1 convs: panel_im2col_gemm_oc4_n2_avx2 (Phase 15)\n";
        std::cout << "  all OC4 epilogues: epilogue_oc4_avx2 (Phase 14 vectorized SSE epilogue)\n";
        std::cout << "  layer2/3/4 conv1+downsample, residuals: unchanged (conv_block32_oc4_avx2 + block32_residual_add_relu_avx2)\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";
        std::cout << "  block32 baseline full model (measured, 1T, med): " << BLOCK32_BASELINE_FULL_MS << " ms\n";
        std::cout << "  basic_engine full model     (measured, 1T, med): " << BASIC_ENGINE_FULL_MS << " ms\n";
        std::cout << "  q2q VNNI optimized prefix    (measured, 1T, med): " << Q2Q_VNNI_FULL_MS << " ms\n";
        std::cout << "  benchmark_block32_full_model_opt2 (Phase 14 Step2 epilogue, med): " << PHASE14_STEP2_MS << " ms\n";
        std::cout << "  benchmark_block32_full_model_opt3 (Phase 14 final, med): " << PHASE14_OPT3_MS << " ms\n";

        // ---- golden input ----
        Tensor golden_input(ctx.IC_C1, ctx.H_IN, ctx.W_IN);
        golden_input.data = q2q::read_f32_file("artifacts/golden_input_fp32.bin",
                                                 static_cast<size_t>(ctx.IC_C1) * ctx.H_IN * ctx.W_IN);

        // ---- AVX2 forward pass (fills ctx.* and ctx.fc_out) ----
        run_block32_full_opt4(ctx, p_conv1_v2, l2, l3, l4, golden_input);

        // ---- correctness check: scalar reference pipeline vs AVX2 pipeline ----
        std::cout << "\n=== correctness: scalar reference vs AVX2 block32 (real golden input) ===\n";
        size_t total_mismatches = 0;
        {
            QTensorNHWC s_conv1_out = make_q8_tensor(ctx.OH1, ctx.OW1, ctx.OC_C1, BORDER, ACT_SCALE);
            QTensorNHWC s_pool_out  = make_q8_tensor(ctx.OHp, ctx.OWp, ctx.OC_C1, BORDER, ACT_SCALE);

            QTensorNHWC s10_mid       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
            QTensorNHWC s10_conv2_out = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
            QTensorNHWC s10_out       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
            QTensorNHWC s11_mid       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
            QTensorNHWC s11_conv2_out = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);
            QTensorNHWC s11_out       = make_q8_tensor(ctx.H1, ctx.W1, ctx.OC1, BORDER, ACT_SCALE);

            QTensorNHWC s20_mid       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
            QTensorNHWC s20_conv2_out = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
            QTensorNHWC s20_identity  = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
            QTensorNHWC s20_out       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
            QTensorNHWC s21_mid       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
            QTensorNHWC s21_conv2_out = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);
            QTensorNHWC s21_out       = make_q8_tensor(ctx.H2, ctx.W2, ctx.OC2, BORDER, ACT_SCALE);

            QTensorNHWC s30_mid       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
            QTensorNHWC s30_conv2_out = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
            QTensorNHWC s30_identity  = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
            QTensorNHWC s30_out       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
            QTensorNHWC s31_mid       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
            QTensorNHWC s31_conv2_out = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);
            QTensorNHWC s31_out       = make_q8_tensor(ctx.H3, ctx.W3, ctx.OC3, BORDER, ACT_SCALE);

            QTensorNHWC s40_mid       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
            QTensorNHWC s40_conv2_out = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
            QTensorNHWC s40_identity  = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
            QTensorNHWC s40_out       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
            QTensorNHWC s41_mid       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
            QTensorNHWC s41_conv2_out = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);
            QTensorNHWC s41_out       = make_q8_tensor(ctx.H4, ctx.W4, ctx.OC4, BORDER, ACT_SCALE);

            // conv1 + maxpool (scalar)
            conv1_scalar(*ctx.conv1, *ctx.bn1, ctx.input_q8, true, s_conv1_out, ctx.OH1, ctx.OW1);
            q2q::q8_maxpool_3x3_s2_p1(s_conv1_out, ctx.OHp, ctx.OWp, s_pool_out);

            // layer1 (scalar)
            conv_block32_scalar(ctx.p10_c1, *ctx.bn10_1, s_pool_out, true,  s10_mid, ctx.H1, ctx.W1);
            conv_block32_scalar(ctx.p10_c2, *ctx.bn10_2, s10_mid,    false, s10_conv2_out, ctx.H1, ctx.W1);
            block32_residual_add_relu_scalar(s10_conv2_out, s_pool_out, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, s10_out);
            conv_block32_scalar(ctx.p11_c1, *ctx.bn11_1, s10_out, true,  s11_mid, ctx.H1, ctx.W1);
            conv_block32_scalar(ctx.p11_c2, *ctx.bn11_2, s11_mid, false, s11_conv2_out, ctx.H1, ctx.W1);
            block32_residual_add_relu_scalar(s11_conv2_out, s10_out, ACT_SCALE, ctx.H1, ctx.W1, ctx.OC1, s11_out);

            // layer2 (scalar)
            conv_block32_scalar(ctx.p20_c1, *ctx.bn20_1, s11_out, true,  s20_mid, ctx.H2, ctx.W2);
            conv_block32_scalar(ctx.p20_c2, *ctx.bn20_2, s20_mid, false, s20_conv2_out, ctx.H2, ctx.W2);
            conv_block32_scalar(ctx.p20_ds, *ctx.bn20_ds, s11_out, false, s20_identity, ctx.H2, ctx.W2);
            block32_residual_add_relu_scalar(s20_conv2_out, s20_identity, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, s20_out);
            conv_block32_scalar(ctx.p21_c1, *ctx.bn21_1, s20_out, true,  s21_mid, ctx.H2, ctx.W2);
            conv_block32_scalar(ctx.p21_c2, *ctx.bn21_2, s21_mid, false, s21_conv2_out, ctx.H2, ctx.W2);
            block32_residual_add_relu_scalar(s21_conv2_out, s20_out, ACT_SCALE, ctx.H2, ctx.W2, ctx.OC2, s21_out);

            // layer3 (scalar)
            conv_block32_scalar(ctx.p30_c1, *ctx.bn30_1, s21_out, true,  s30_mid, ctx.H3, ctx.W3);
            conv_block32_scalar(ctx.p30_c2, *ctx.bn30_2, s30_mid, false, s30_conv2_out, ctx.H3, ctx.W3);
            conv_block32_scalar(ctx.p30_ds, *ctx.bn30_ds, s21_out, false, s30_identity, ctx.H3, ctx.W3);
            block32_residual_add_relu_scalar(s30_conv2_out, s30_identity, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, s30_out);
            conv_block32_scalar(ctx.p31_c1, *ctx.bn31_1, s30_out, true,  s31_mid, ctx.H3, ctx.W3);
            conv_block32_scalar(ctx.p31_c2, *ctx.bn31_2, s31_mid, false, s31_conv2_out, ctx.H3, ctx.W3);
            block32_residual_add_relu_scalar(s31_conv2_out, s30_out, ACT_SCALE, ctx.H3, ctx.W3, ctx.OC3, s31_out);

            // layer4 (scalar)
            conv_block32_scalar(ctx.p40_c1, *ctx.bn40_1, s31_out, true,  s40_mid, ctx.H4, ctx.W4);
            conv_block32_scalar(ctx.p40_c2, *ctx.bn40_2, s40_mid, false, s40_conv2_out, ctx.H4, ctx.W4);
            conv_block32_scalar(ctx.p40_ds, *ctx.bn40_ds, s31_out, false, s40_identity, ctx.H4, ctx.W4);
            block32_residual_add_relu_scalar(s40_conv2_out, s40_identity, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, s40_out);
            conv_block32_scalar(ctx.p41_c1, *ctx.bn41_1, s40_out, true,  s41_mid, ctx.H4, ctx.W4);
            conv_block32_scalar(ctx.p41_c2, *ctx.bn41_2, s41_mid, false, s41_conv2_out, ctx.H4, ctx.W4);
            block32_residual_add_relu_scalar(s41_conv2_out, s40_out, ACT_SCALE, ctx.H4, ctx.W4, ctx.OC4, s41_out);

            total_mismatches += compare("conv1 out", s_conv1_out, ctx.conv1_out);
            total_mismatches += compare("maxpool out", s_pool_out, ctx.pool_out);
            total_mismatches += compare("layer1 out", s11_out, ctx.l11_out);
            total_mismatches += compare("layer2 out", s21_out, ctx.l21_out);
            total_mismatches += compare("layer3 out", s31_out, ctx.l31_out);
            total_mismatches += compare("layer1-4 out", s41_out, ctx.l41_out);
        }
        std::cout << "  total mismatches: " << total_mismatches << "\n";

        // ---- timing (AVX2 only, stage-by-stage breakdown) ----
        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        for (int i = 0; i < N_WARMUP; ++i) run_block32_full_opt4(ctx, p_conv1_v2, l2, l3, l4, golden_input);

        std::vector<double> t_quant(N_TIMED), t_conv1(N_TIMED), t_maxpool(N_TIMED),
                            t_layer1(N_TIMED), t_layer2(N_TIMED), t_layer3(N_TIMED), t_layer4(N_TIMED),
                            t_dequant_avgpool_fc(N_TIMED), t_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q::quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.input_q8);
            const double t1 = now_ms();
            conv_block32_conv1_oc4_v2_split_avx2(p_conv1_v2, *ctx.bn1, ctx.input_q8, true, ctx.conv1_out, ctx.OH1, ctx.OW1);
            const double t2 = now_ms();
            q2q::q8_maxpool_3x3_s2_p1(ctx.conv1_out, ctx.OHp, ctx.OWp, ctx.pool_out);
            const double t3 = now_ms();
            run_block32_layer1_opt(ctx, ctx.pool_out);
            const double t4 = now_ms();
            run_block32_layer2_opt4(ctx, ctx.l11_out, l2.p20_c2, l2.p21_c1, l2.p21_c2);
            const double t5 = now_ms();
            run_block32_layer3_opt4(ctx, ctx.l21_out, l3.p30_c2, l3.p31_c1, l3.p31_c2);
            const double t6 = now_ms();
            run_block32_layer4_opt4(ctx, ctx.l31_out, l4.p40_c2, l4.p41_c1, l4.p41_c2);
            const double t7 = now_ms();
            q2q::dequantize_q8_nhwc_to_fp32_nchw(ctx.l41_out, ctx.H4, ctx.W4, ctx.OC4, ctx.dequant_out);
            adaptive_avgpool_1x1_into(ctx.dequant_out, ctx.avgpool_out);
            linear_into(ctx.avgpool_out, *ctx.fc, ctx.fc_out);
            const double t8 = now_ms();

            const size_t si = static_cast<size_t>(i);
            t_quant[si] = t1 - t0;
            t_conv1[si] = t2 - t1;
            t_maxpool[si] = t3 - t2;
            t_layer1[si] = t4 - t3;
            t_layer2[si] = t5 - t4;
            t_layer3[si] = t6 - t5;
            t_layer4[si] = t7 - t6;
            t_dequant_avgpool_fc[si] = t8 - t7;
            t_total[si] = t8 - t0;
        }
        volatile float sink = ctx.fc_out.data[0] + ctx.dequant_out.data[0];
        (void)sink;

        std::cout << "\n=== full-model timing (AVX2 block32 + Phase 7/8/14/15 variants, " << N_TIMED << " runs) ===\n";
        print_stats("quantize_input_ms     ", t_quant, N_TIMED);
        print_stats("conv1_ms              ", t_conv1, N_TIMED);
        print_stats("maxpool_ms            ", t_maxpool, N_TIMED);
        print_stats("layer1_ms             ", t_layer1, N_TIMED);
        print_stats("layer2_ms             ", t_layer2, N_TIMED);
        print_stats("layer3_ms             ", t_layer3, N_TIMED);
        print_stats("layer4_ms             ", t_layer4, N_TIMED);
        print_stats("dequant_avgpool_fc_ms ", t_dequant_avgpool_fc, N_TIMED);
        print_stats("total_profiled_ms     ", t_total, N_TIMED);

        std::vector<double> s_tot = t_total;
        std::sort(s_tot.begin(), s_tot.end());
        const double med_tot = s_tot[N_TIMED / 2];

        std::cout << "\n=== gate: AVX2 block32 (Phase 7/8/14/15 variants) full model vs baselines ===\n";
        std::cout << "  block32 baseline full model (med): " << BLOCK32_BASELINE_FULL_MS << " ms\n";
        std::cout << "  basic_engine full model     (med): " << BASIC_ENGINE_FULL_MS << " ms\n";
        std::cout << "  q2q VNNI optimized prefix    (med): " << Q2Q_VNNI_FULL_MS << " ms\n";
        std::cout << "  benchmark_block32_full_model_opt2 (Phase 14 Step2 epilogue, med): " << PHASE14_STEP2_MS << " ms\n";
        std::cout << "  benchmark_block32_full_model_opt3 (Phase 14 final, med): " << PHASE14_OPT3_MS << " ms\n";
        std::cout << "  total_profiled_ms (opt4, Phase 15, med): " << med_tot << " ms\n";

        if (med_tot < PHASE14_OPT3_MS) {
            std::cout << "  vs opt3 (Phase 14 final): IMPROVES by " << (PHASE14_OPT3_MS - med_tot)
                      << " ms (" << (PHASE14_OPT3_MS / med_tot) << "x).\n";
        } else {
            std::cout << "  vs opt3 (Phase 14 final): REGRESSES by " << (med_tot - PHASE14_OPT3_MS) << " ms.\n";
        }
        if (med_tot < BASIC_ENGINE_FULL_MS) {
            std::cout << "  vs basic_engine: BEATS by " << (BASIC_ENGINE_FULL_MS / med_tot) << "x.\n";
        } else {
            std::cout << "  vs basic_engine: does NOT beat (" << (med_tot / BASIC_ENGINE_FULL_MS) << "x).\n";
        }
        std::cout << "  vs q2q VNNI optimized prefix: " << (med_tot / Q2Q_VNNI_FULL_MS) << "x slower.\n";

        std::cout << "\n=== Phase 15 decision gate ===\n";
        if (med_tot <= GATE_EXCELLENT_MS) {
            std::cout << "  RESULT: full model (" << med_tot << " ms) <= " << GATE_EXCELLENT_MS
                      << " ms -- EXCELLENT; stop single-thread work.\n";
        } else if (med_tot <= GATE_STRONG_MS) {
            std::cout << "  RESULT: full model (" << med_tot << " ms) in (" << GATE_EXCELLENT_MS << ", " << GATE_STRONG_MS
                      << "] ms -- STRONG; stop and document.\n";
        } else if (med_tot <= GATE_KEEP_MS) {
            std::cout << "  RESULT: full model (" << med_tot << " ms) in (" << GATE_STRONG_MS << ", " << GATE_KEEP_MS
                      << "] ms -- keep if code is clean.\n";
        } else {
            std::cout << "  RESULT: full model (" << med_tot << " ms) > " << GATE_KEEP_MS
                      << " ms -- benchmark-only; keep opt3 as final.\n";
        }

        std::cout << "\nC++ engine latency\n";
        std::cout << "  runs:      " << N_TIMED << "\n";
        std::cout << "  min_ms:    " << s_tot.front() << "\n";
        std::cout << "  median_ms: " << med_tot << "\n";
        std::cout << "  mean_ms:   " << (std::accumulate(s_tot.begin(), s_tot.end(), 0.0) / N_TIMED) << "\n";
        std::cout << "  p90_ms:    " << s_tot[N_TIMED * 9 / 10] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
