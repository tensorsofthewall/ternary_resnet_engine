// benchmark_block32_layer234_stack.cpp -- AVX2-only (no AVX-VNNI) block32:
// combined layer2+3+4
//
// Chains all three residual stacks end-to-end in q2q::QTensorNHWC, with NO
// FP32 materialization at any stack boundary, using the AVX2 OC4 block32
// ternary conv kernel (conv_block32_oc4_avx2) validated on layer2.1.conv2:
//
//   synthetic q8 NHWC (56x56x64)
//   layer2.0 -> layer2.1                          (28x28x128 q8 out)
//   layer3.0 -> layer3.1                          (14x14x256 q8 out)
//   layer4.0 -> layer4.1                          (7x7x512   q8 out)
//
// Every conv (3x3 stride1/2 pad1, and the 1x1 stride2 pad0 downsamples) goes
// through the single generic conv_block32_oc4_avx2 kernel; residual adds use
// a local single-threaded AVX2 kernel (block32_residual_avx2). A scalar
// reference pipeline (conv_block32_scalar + scalar residual) verifies the
// AVX2 pipeline's final output before timing.
//
// Usage: ./benchmark_block32_layer234_stack <model.bin>

#include "block32_conv_avx2.hpp"
#include "block32_plan.hpp"
#include "block32_residual.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include "nhwc_common.hpp"
#include "q2q_tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace nhwc_block32;
using q2q::make_q8_tensor;
using q2q::QTensorNHWC;

namespace {

constexpr int BORDER = 1;

// Measured basic_engine costs (q8 OCBlock2 bucket path, 1T) for these same
// layers -- reused from benchmark_q2q_layer234_stack's gate baseline.
constexpr double BASIC_ENGINE_L2_MS = 10.28;
constexpr double BASIC_ENGINE_L3_MS = 12.50;
constexpr double BASIC_ENGINE_L4_MS = 9.59;
constexpr double BASIC_ENGINE_COMBINED_MS =
    BASIC_ENGINE_L2_MS + BASIC_ENGINE_L3_MS + BASIC_ENGINE_L4_MS;

void fill_random_q8(QTensorNHWC& t, int H, int W, int C, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            uint8_t* px = t.data.data() + ((static_cast<size_t>(h + t.border) * t.W + static_cast<size_t>(w + t.border)) * C);
            for (int c = 0; c < C; ++c) px[c] = static_cast<uint8_t>(dist(rng));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);

        // layer2.0 / layer2.1
        const Conv2DParams& c20_conv1 = model.convs.at("layer2.0.conv1");
        const Conv2DParams& c20_conv2 = model.convs.at("layer2.0.conv2");
        const Conv2DParams& c20_ds    = model.convs.at("layer2.0.downsample.0");
        const BNFusedParams& bn20_1   = model.fused_bns.at("layer2.0.bn1");
        const BNFusedParams& bn20_2   = model.fused_bns.at("layer2.0.bn2");
        const BNFusedParams& bn20_ds  = model.fused_bns.at("layer2.0.downsample.1");
        const Conv2DParams& c21_conv1 = model.convs.at("layer2.1.conv1");
        const Conv2DParams& c21_conv2 = model.convs.at("layer2.1.conv2");
        const BNFusedParams& bn21_1   = model.fused_bns.at("layer2.1.bn1");
        const BNFusedParams& bn21_2   = model.fused_bns.at("layer2.1.bn2");
        // layer3.0 / layer3.1
        const Conv2DParams& c30_conv1 = model.convs.at("layer3.0.conv1");
        const Conv2DParams& c30_conv2 = model.convs.at("layer3.0.conv2");
        const Conv2DParams& c30_ds    = model.convs.at("layer3.0.downsample.0");
        const BNFusedParams& bn30_1   = model.fused_bns.at("layer3.0.bn1");
        const BNFusedParams& bn30_2   = model.fused_bns.at("layer3.0.bn2");
        const BNFusedParams& bn30_ds  = model.fused_bns.at("layer3.0.downsample.1");
        const Conv2DParams& c31_conv1 = model.convs.at("layer3.1.conv1");
        const Conv2DParams& c31_conv2 = model.convs.at("layer3.1.conv2");
        const BNFusedParams& bn31_1   = model.fused_bns.at("layer3.1.bn1");
        const BNFusedParams& bn31_2   = model.fused_bns.at("layer3.1.bn2");
        // layer4.0 / layer4.1
        const Conv2DParams& c40_conv1 = model.convs.at("layer4.0.conv1");
        const Conv2DParams& c40_conv2 = model.convs.at("layer4.0.conv2");
        const Conv2DParams& c40_ds    = model.convs.at("layer4.0.downsample.0");
        const BNFusedParams& bn40_1   = model.fused_bns.at("layer4.0.bn1");
        const BNFusedParams& bn40_2   = model.fused_bns.at("layer4.0.bn2");
        const BNFusedParams& bn40_ds  = model.fused_bns.at("layer4.0.downsample.1");
        const Conv2DParams& c41_conv1 = model.convs.at("layer4.1.conv1");
        const Conv2DParams& c41_conv2 = model.convs.at("layer4.1.conv2");
        const BNFusedParams& bn41_1   = model.fused_bns.at("layer4.1.bn1");
        const BNFusedParams& bn41_2   = model.fused_bns.at("layer4.1.bn2");

        const int IC0 = c20_conv1.in_channels;   // 64   (input to layer2: 56x56x64)
        const int OC2 = c20_conv1.out_channels;  // 128  (layer2 output: 28x28x128)
        const int OC3 = c30_conv1.out_channels;  // 256  (layer3 output: 14x14x256)
        const int OC4 = c40_conv1.out_channels;  // 512  (layer4 output: 7x7x512)
        const int H_IN = 56, W_IN = 56;
        const int H2 = 28, W2 = 28;
        const int H3 = 14, W3 = 14;
        const int H4 = 7,  W4 = 7;

        if (IC0 % BLOCK32 != 0 || OC2 % 4 != 0 || OC3 % 4 != 0 || OC4 % 4 != 0)
            throw std::runtime_error("channel counts must be divisible by 32 (IC) / 4 (OC)");

        std::cout << "=== benchmark_block32_layer234_stack: combined layer2+3+4 q8-to-q8 (AVX2 block32, no VNNI) ===\n";
        std::cout << "  input: synthetic q8 NHWC " << IC0 << "x" << H_IN << "x" << W_IN << "\n";
        std::cout << "  layer2: 56x56x" << IC0 << " -> 28x28x" << OC2 << "\n";
        std::cout << "  layer3: 28x28x" << OC2 << " -> 14x14x" << OC3 << "\n";
        std::cout << "  layer4: 14x14x" << OC3 << " -> 7x7x"   << OC4 << "\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";
        std::cout << "  basic_engine baselines (measured, 1T, q8 OCBlock2 bucket):\n"
                  << "    layer2   = " << BASIC_ENGINE_L2_MS << " ms\n"
                  << "    layer3   = " << BASIC_ENGINE_L3_MS << " ms\n"
                  << "    layer4   = " << BASIC_ENGINE_L4_MS << " ms\n"
                  << "    combined = " << BASIC_ENGINE_COMBINED_MS << " ms\n";

        // Build all fifteen block32 plans up front.
        const Block32ConvPlan p20_c1 = build_block32_plan(c20_conv1);
        const Block32ConvPlan p20_c2 = build_block32_plan(c20_conv2);
        const Block32ConvPlan p20_ds = build_block32_plan(c20_ds);
        const Block32ConvPlan p21_c1 = build_block32_plan(c21_conv1);
        const Block32ConvPlan p21_c2 = build_block32_plan(c21_conv2);
        const Block32ConvPlan p30_c1 = build_block32_plan(c30_conv1);
        const Block32ConvPlan p30_c2 = build_block32_plan(c30_conv2);
        const Block32ConvPlan p30_ds = build_block32_plan(c30_ds);
        const Block32ConvPlan p31_c1 = build_block32_plan(c31_conv1);
        const Block32ConvPlan p31_c2 = build_block32_plan(c31_conv2);
        const Block32ConvPlan p40_c1 = build_block32_plan(c40_conv1);
        const Block32ConvPlan p40_c2 = build_block32_plan(c40_conv2);
        const Block32ConvPlan p40_ds = build_block32_plan(c40_ds);
        const Block32ConvPlan p41_c1 = build_block32_plan(c41_conv1);
        const Block32ConvPlan p41_c2 = build_block32_plan(c41_conv2);

        // Synthetic q8 NHWC input: stand-in for the layer2-input activation
        // (64x56x56) -- generated directly as q8, no FP32 anywhere.
        QTensorNHWC input = make_q8_tensor(H_IN, W_IN, IC0, BORDER, ACT_SCALE);
        fill_random_q8(input, H_IN, W_IN, IC0, 0x9E3779B9u);

        // ---- AVX2 (block32 OC4) scratch tensors ----
        QTensorNHWC l20_mid       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
        QTensorNHWC l20_conv2_out = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
        QTensorNHWC l20_identity  = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
        QTensorNHWC l20_out       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
        QTensorNHWC l21_mid       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
        QTensorNHWC l21_conv2_out = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
        QTensorNHWC l21_out       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);  // -> layer3 input
        QTensorNHWC l30_mid       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
        QTensorNHWC l30_conv2_out = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
        QTensorNHWC l30_identity  = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
        QTensorNHWC l30_out       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
        QTensorNHWC l31_mid       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
        QTensorNHWC l31_conv2_out = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
        QTensorNHWC l31_out       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);  // -> layer4 input
        QTensorNHWC l40_mid       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
        QTensorNHWC l40_conv2_out = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
        QTensorNHWC l40_identity  = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
        QTensorNHWC l40_out       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
        QTensorNHWC l41_mid       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
        QTensorNHWC l41_conv2_out = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
        QTensorNHWC l41_out       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);  // combined output

        auto run_layer2 = [&](const QTensorNHWC& in) {
            conv_block32_oc4_avx2(p20_c1, bn20_1, in,      true,  l20_mid, H2, W2);
            conv_block32_oc4_avx2(p20_c2, bn20_2, l20_mid, false, l20_conv2_out, H2, W2);
            conv_block32_oc4_avx2(p20_ds, bn20_ds, in,     false, l20_identity, H2, W2);
            block32_residual_add_relu_avx2(l20_conv2_out, l20_identity, ACT_SCALE, H2, W2, OC2, l20_out);
            conv_block32_oc4_avx2(p21_c1, bn21_1, l20_out, true,  l21_mid, H2, W2);
            conv_block32_oc4_avx2(p21_c2, bn21_2, l21_mid, false, l21_conv2_out, H2, W2);
            block32_residual_add_relu_avx2(l21_conv2_out, l20_out, ACT_SCALE, H2, W2, OC2, l21_out);
        };
        auto run_layer3 = [&](const QTensorNHWC& in) {
            conv_block32_oc4_avx2(p30_c1, bn30_1, in,      true,  l30_mid, H3, W3);
            conv_block32_oc4_avx2(p30_c2, bn30_2, l30_mid, false, l30_conv2_out, H3, W3);
            conv_block32_oc4_avx2(p30_ds, bn30_ds, in,     false, l30_identity, H3, W3);
            block32_residual_add_relu_avx2(l30_conv2_out, l30_identity, ACT_SCALE, H3, W3, OC3, l30_out);
            conv_block32_oc4_avx2(p31_c1, bn31_1, l30_out, true,  l31_mid, H3, W3);
            conv_block32_oc4_avx2(p31_c2, bn31_2, l31_mid, false, l31_conv2_out, H3, W3);
            block32_residual_add_relu_avx2(l31_conv2_out, l30_out, ACT_SCALE, H3, W3, OC3, l31_out);
        };
        auto run_layer4 = [&](const QTensorNHWC& in) {
            conv_block32_oc4_avx2(p40_c1, bn40_1, in,      true,  l40_mid, H4, W4);
            conv_block32_oc4_avx2(p40_c2, bn40_2, l40_mid, false, l40_conv2_out, H4, W4);
            conv_block32_oc4_avx2(p40_ds, bn40_ds, in,     false, l40_identity, H4, W4);
            block32_residual_add_relu_avx2(l40_conv2_out, l40_identity, ACT_SCALE, H4, W4, OC4, l40_out);
            conv_block32_oc4_avx2(p41_c1, bn41_1, l40_out, true,  l41_mid, H4, W4);
            conv_block32_oc4_avx2(p41_c2, bn41_2, l41_mid, false, l41_conv2_out, H4, W4);
            block32_residual_add_relu_avx2(l41_conv2_out, l40_out, ACT_SCALE, H4, W4, OC4, l41_out);
        };

        // ---- correctness check: scalar reference pipeline vs AVX2 pipeline ----
        {
            QTensorNHWC s20_mid       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s20_conv2_out = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s20_identity  = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s20_out       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s21_mid       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s21_conv2_out = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s21_out       = make_q8_tensor(H2, W2, OC2, BORDER, ACT_SCALE);
            QTensorNHWC s30_mid       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s30_conv2_out = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s30_identity  = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s30_out       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s31_mid       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s31_conv2_out = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s31_out       = make_q8_tensor(H3, W3, OC3, BORDER, ACT_SCALE);
            QTensorNHWC s40_mid       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
            QTensorNHWC s40_conv2_out = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
            QTensorNHWC s40_identity  = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
            QTensorNHWC s40_out       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
            QTensorNHWC s41_mid       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
            QTensorNHWC s41_conv2_out = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);
            QTensorNHWC s41_out       = make_q8_tensor(H4, W4, OC4, BORDER, ACT_SCALE);

            conv_block32_scalar(p20_c1, bn20_1, input,   true,  s20_mid, H2, W2);
            conv_block32_scalar(p20_c2, bn20_2, s20_mid, false, s20_conv2_out, H2, W2);
            conv_block32_scalar(p20_ds, bn20_ds, input,  false, s20_identity, H2, W2);
            block32_residual_add_relu_scalar(s20_conv2_out, s20_identity, ACT_SCALE, H2, W2, OC2, s20_out);
            conv_block32_scalar(p21_c1, bn21_1, s20_out, true,  s21_mid, H2, W2);
            conv_block32_scalar(p21_c2, bn21_2, s21_mid, false, s21_conv2_out, H2, W2);
            block32_residual_add_relu_scalar(s21_conv2_out, s20_out, ACT_SCALE, H2, W2, OC2, s21_out);

            conv_block32_scalar(p30_c1, bn30_1, s21_out, true,  s30_mid, H3, W3);
            conv_block32_scalar(p30_c2, bn30_2, s30_mid, false, s30_conv2_out, H3, W3);
            conv_block32_scalar(p30_ds, bn30_ds, s21_out, false, s30_identity, H3, W3);
            block32_residual_add_relu_scalar(s30_conv2_out, s30_identity, ACT_SCALE, H3, W3, OC3, s30_out);
            conv_block32_scalar(p31_c1, bn31_1, s30_out, true,  s31_mid, H3, W3);
            conv_block32_scalar(p31_c2, bn31_2, s31_mid, false, s31_conv2_out, H3, W3);
            block32_residual_add_relu_scalar(s31_conv2_out, s30_out, ACT_SCALE, H3, W3, OC3, s31_out);

            conv_block32_scalar(p40_c1, bn40_1, s31_out, true,  s40_mid, H4, W4);
            conv_block32_scalar(p40_c2, bn40_2, s40_mid, false, s40_conv2_out, H4, W4);
            conv_block32_scalar(p40_ds, bn40_ds, s31_out, false, s40_identity, H4, W4);
            block32_residual_add_relu_scalar(s40_conv2_out, s40_identity, ACT_SCALE, H4, W4, OC4, s40_out);
            conv_block32_scalar(p41_c1, bn41_1, s40_out, true,  s41_mid, H4, W4);
            conv_block32_scalar(p41_c2, bn41_2, s41_mid, false, s41_conv2_out, H4, W4);
            block32_residual_add_relu_scalar(s41_conv2_out, s40_out, ACT_SCALE, H4, W4, OC4, s41_out);

            run_layer2(input);
            run_layer3(l21_out);
            run_layer4(l31_out);

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
            };
            compare("layer2 out", s21_out, l21_out);
            compare("layer3 out", s31_out, l31_out);
            compare("layer4 out", s41_out, l41_out);
        }
        std::cout << "\n";

        // ---- timing (AVX2 only) ----
        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        auto run_all = [&]() {
            run_layer2(input);
            run_layer3(l21_out);
            run_layer4(l31_out);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_all();

        std::vector<double> t_layer2(N_TIMED), t_layer3(N_TIMED), t_layer4(N_TIMED), t_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            run_layer2(input);
            const double t1 = now_ms();
            run_layer3(l21_out);
            const double t2 = now_ms();
            run_layer4(l31_out);
            const double t3 = now_ms();
            t_layer2[static_cast<size_t>(i)] = t1 - t0;
            t_layer3[static_cast<size_t>(i)] = t2 - t1;
            t_layer4[static_cast<size_t>(i)] = t3 - t2;
            t_total[static_cast<size_t>(i)]  = t3 - t0;
        }
        volatile uint32_t sink = l41_out.data[0] + l41_out.data[l41_out.data.size() - 1];
        (void)sink;

        std::cout << "=== combined layer2+3+4 q8-to-q8 timing (AVX2 block32 OC4, " << N_TIMED << " runs) ===\n";
        print_stats("layer2_ms", t_layer2, N_TIMED);
        print_stats("layer3_ms", t_layer3, N_TIMED);
        print_stats("layer4_ms", t_layer4, N_TIMED);
        print_stats("total_ms ", t_total, N_TIMED);

        std::vector<double> s_l2 = t_layer2, s_l3 = t_layer3, s_l4 = t_layer4, s_tot = t_total;
        std::sort(s_l2.begin(), s_l2.end());
        std::sort(s_l3.begin(), s_l3.end());
        std::sort(s_l4.begin(), s_l4.end());
        std::sort(s_tot.begin(), s_tot.end());
        const double med_l2  = s_l2[N_TIMED / 2];
        const double med_l3  = s_l3[N_TIMED / 2];
        const double med_l4  = s_l4[N_TIMED / 2];
        const double med_tot = s_tot[N_TIMED / 2];

        std::cout << "\n=== gate: AVX2 block32 layer2+3+4 vs basic_engine ===\n";
        std::cout << "  basic_engine combined (l2+l3+l4): " << BASIC_ENGINE_COMBINED_MS << " ms"
                  << "  (l2=" << BASIC_ENGINE_L2_MS << ", l3=" << BASIC_ENGINE_L3_MS
                  << ", l4=" << BASIC_ENGINE_L4_MS << ")\n";
        std::cout << "  layer2_ms (med): " << med_l2  << " ms  (basic_engine " << BASIC_ENGINE_L2_MS << " ms)\n";
        std::cout << "  layer3_ms (med): " << med_l3  << " ms  (basic_engine " << BASIC_ENGINE_L3_MS << " ms)\n";
        std::cout << "  layer4_ms (med): " << med_l4  << " ms  (basic_engine " << BASIC_ENGINE_L4_MS << " ms)\n";
        std::cout << "  total_ms  (med): " << med_tot << " ms  (basic_engine " << BASIC_ENGINE_COMBINED_MS << " ms)\n";

        if (med_tot < BASIC_ENGINE_COMBINED_MS) {
            std::cout << "  RESULT: combined (" << med_tot << " ms) BEATS basic_engine ("
                      << BASIC_ENGINE_COMBINED_MS << " ms) by "
                      << (BASIC_ENGINE_COMBINED_MS / med_tot) << "x.\n";
        } else {
            std::cout << "  RESULT: combined (" << med_tot << " ms) does NOT beat basic_engine ("
                      << BASIC_ENGINE_COMBINED_MS << " ms).\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
