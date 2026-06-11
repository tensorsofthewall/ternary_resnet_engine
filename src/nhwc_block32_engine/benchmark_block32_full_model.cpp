// benchmark_block32_full_model.cpp -- AVX2-only (no AVX-VNNI) block32:
// full ResNet18 forward pass
//
// End-to-end FP32 NCHW input -> q8 NHWC quantize -> conv1 -> maxpool ->
// layer1 -> layer2 -> layer3 -> layer4 -> dequant -> avgpool -> fc, using
// the AVX2 OC4 block32 ternary conv kernel (conv_block32_oc4_avx2), the
// gather-based AVX2 conv1 kernel (conv_block32_conv1_oc4_avx2), and the AVX2
// residual add+ReLU kernel (block32_residual_add_relu_avx2) -- no FP32
// materialization between conv1 and the layer4 output.
//
// A scalar reference pipeline (conv1_scalar + conv_block32_scalar + scalar
// residual) verifies conv1, layer1, and the full layer1-4 stack against the
// AVX2 pipeline before timing.
//
// Usage: ./benchmark_block32_full_model <model.bin>
//   (reads artifacts/golden_input_fp32.bin as the model input, and
//    artifacts/golden_logits_fp32.bin as an approximate logits reference,
//    if present)

#include "block32_conv1.hpp"
#include "block32_conv_avx2.hpp"
#include "block32_plan.hpp"
#include "block32_residual.hpp"
#include "block32_resnet18.hpp"
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
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace nhwc_block32;
using q2q::make_q8_tensor;
using q2q::QTensorNHWC;

namespace {

constexpr int BORDER = 1;

// basic_engine full-model cost (median, 1T) -- measured live via
// benchmark_q2q_full_prefix_optimized (read-only reuse, not linked into this
// AVX2-only / VNNI-free binary):
//   OMP_NUM_THREADS=1 ./build/benchmark_q2q_full_prefix_optimized <model> --threads 1
//   basic_full_ms: min=43.5287 med=43.8679 p90=44.1948 avg=43.8971 ms
constexpr double BASIC_ENGINE_FULL_MS = 43.8679;

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

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);
        Block32Resnet18Context ctx = build_block32_resnet18_context(model);

        std::cout << "=== benchmark_block32_full_model: full ResNet18 forward (AVX2 block32, no VNNI) ===\n";
        std::cout << "  input: " << ctx.IC_C1 << "x" << ctx.H_IN << "x" << ctx.W_IN << " FP32 NCHW (golden_input)\n";
        std::cout << "  conv1: " << ctx.IC_C1 << "x" << ctx.H_IN << "x" << ctx.W_IN
                  << " -> " << ctx.OC_C1 << "x" << ctx.OH1 << "x" << ctx.OW1 << "\n";
        std::cout << "  maxpool: -> " << ctx.OC_C1 << "x" << ctx.OHp << "x" << ctx.OWp << "\n";
        std::cout << "  layer1: -> " << ctx.OC1 << "x" << ctx.H1 << "x" << ctx.W1 << "\n";
        std::cout << "  layer2: -> " << ctx.OC2 << "x" << ctx.H2 << "x" << ctx.W2 << "\n";
        std::cout << "  layer3: -> " << ctx.OC3 << "x" << ctx.H3 << "x" << ctx.W3 << "\n";
        std::cout << "  layer4: -> " << ctx.OC4 << "x" << ctx.H4 << "x" << ctx.W4 << "\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";
        std::cout << "  basic_engine full-model baseline (measured live, 1T, med): "
                  << BASIC_ENGINE_FULL_MS << " ms\n";

        // ---- golden input / logits ----
        Tensor golden_input(ctx.IC_C1, ctx.H_IN, ctx.W_IN);
        golden_input.data = q2q::read_f32_file("artifacts/golden_input_fp32.bin",
                                                 static_cast<size_t>(ctx.IC_C1) * ctx.H_IN * ctx.W_IN);

        std::vector<float> golden_logits;
        bool have_golden_logits = false;
        try {
            golden_logits = q2q::read_f32_file("artifacts/golden_logits_fp32.bin", 1000);
            have_golden_logits = true;
        } catch (const std::exception&) {
            std::cout << "  (no artifacts/golden_logits_fp32.bin found -- skipping logits reference comparison)\n";
        }

        // ---- AVX2 forward pass (fills ctx.* and ctx.fc_out) ----
        run_block32_full(ctx, golden_input);

        // ---- correctness check: scalar reference pipeline vs AVX2 pipeline ----
        std::cout << "\n=== correctness: scalar reference vs AVX2 block32 (real golden input) ===\n";
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

            compare("conv1 out", s_conv1_out, ctx.conv1_out);
            compare("maxpool out", s_pool_out, ctx.pool_out);
            compare("layer1 out", s11_out, ctx.l11_out);
            compare("layer2 out", s21_out, ctx.l21_out);
            compare("layer3 out", s31_out, ctx.l31_out);
            compare("layer1-4 out", s41_out, ctx.l41_out);
        }

        // ---- final logits ----
        std::cout << "\n=== final logits (sample) ===\n";
        std::cout << "  logits[0..9]:";
        for (int i = 0; i < 10; ++i) std::cout << " " << ctx.fc_out.data[static_cast<size_t>(i)];
        std::cout << "\n";
        {
            int argmax = 0;
            for (size_t i = 1; i < ctx.fc_out.data.size(); ++i)
                if (ctx.fc_out.data[i] > ctx.fc_out.data[static_cast<size_t>(argmax)]) argmax = static_cast<int>(i);
            std::cout << "  argmax: class " << argmax << " (logit=" << ctx.fc_out.data[static_cast<size_t>(argmax)] << ")\n";
        }
        if (have_golden_logits) {
            double max_abs = 0.0, sum_abs = 0.0;
            for (size_t i = 0; i < golden_logits.size(); ++i) {
                const double diff = std::abs(static_cast<double>(ctx.fc_out.data[i]) - static_cast<double>(golden_logits[i]));
                max_abs = std::max(max_abs, diff);
                sum_abs += diff;
            }
            const double mean_abs = sum_abs / static_cast<double>(golden_logits.size());
            std::cout << "  vs golden_logits_fp32.bin: max_abs=" << max_abs << "  mean_abs=" << mean_abs << "\n";
        }

        // ---- timing (AVX2 only, stage-by-stage breakdown) ----
        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        for (int i = 0; i < N_WARMUP; ++i) run_block32_full(ctx, golden_input);

        std::vector<double> t_quant(N_TIMED), t_conv1(N_TIMED), t_maxpool(N_TIMED),
                            t_layer1(N_TIMED), t_layer2(N_TIMED), t_layer3(N_TIMED), t_layer4(N_TIMED),
                            t_dequant_avgpool_fc(N_TIMED), t_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q::quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.input_q8);
            const double t1 = now_ms();
            conv_block32_conv1_oc4_avx2(ctx.p_conv1, *ctx.bn1, ctx.input_q8, true, ctx.conv1_out, ctx.OH1, ctx.OW1);
            const double t2 = now_ms();
            q2q::q8_maxpool_3x3_s2_p1(ctx.conv1_out, ctx.OHp, ctx.OWp, ctx.pool_out);
            const double t3 = now_ms();
            run_block32_layer1(ctx, ctx.pool_out);
            const double t4 = now_ms();
            run_block32_layer2(ctx, ctx.l11_out);
            const double t5 = now_ms();
            run_block32_layer3(ctx, ctx.l21_out);
            const double t6 = now_ms();
            run_block32_layer4(ctx, ctx.l31_out);
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

        std::cout << "\n=== full-model timing (AVX2 block32, " << N_TIMED << " runs) ===\n";
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

        std::cout << "\n=== gate: AVX2 block32 full model vs basic_engine ===\n";
        std::cout << "  basic_engine full model (measured live, med): " << BASIC_ENGINE_FULL_MS << " ms\n";
        std::cout << "  total_profiled_ms       (med)               : " << med_tot << " ms\n";
        if (med_tot < BASIC_ENGINE_FULL_MS) {
            std::cout << "  RESULT: full model (" << med_tot << " ms) BEATS basic_engine ("
                      << BASIC_ENGINE_FULL_MS << " ms) by "
                      << (BASIC_ENGINE_FULL_MS / med_tot) << "x.\n";
        } else if (med_tot < BASIC_ENGINE_FULL_MS * 1.10) {
            std::cout << "  RESULT: full model (" << med_tot << " ms) is within 10% of basic_engine ("
                      << BASIC_ENGINE_FULL_MS << " ms).\n";
        } else {
            std::cout << "  RESULT: full model (" << med_tot << " ms) does NOT beat basic_engine ("
                      << BASIC_ENGINE_FULL_MS << " ms) by more than 10%.\n";
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
