// benchmark_q2q_full_prefix_optimized.cpp — the final Q2Q/VNNI engine benchmark.
//
// Runs the full all-Q2Q forward prefix:
//   FP32 input -> quantize -> conv1(Q2Q) -> maxpool(q8) -> layer1-4(Q2Q)
//   -> dequant -> basic_engine avgpool+fc
// using the tuned dense-conv and residual microkernels (validated standalone
// in benchmark_q2q_microkernel_opt.cpp, ~2.08x and ~17x paired wins over the
// straightforward versions) wired into every layer1-4 conv/residual call:
//
//   q2q_conv_bn_requant        -> q2q_conv_bn_requant_apanel
//   q2q_residual_add_relu      -> q2q_residual_add_relu_avx2
//
// conv1 (im2col VNNI) and maxpool (q8-direct) use their own validated kernels
// and are shared by both phases below -- the tuned dense-conv/residual
// microkernels only target the dense layer1-4 family (the ~1.85-1.88x trunk),
// so that's where the wins land.
//
// To keep the comparison live and fair, BOTH the straightforward and the tuned
// kernel pairings run in this one binary back-to-back ("baseline_total" then
// "optimized_total"), sharing one scratch-tensor set across the two
// self-contained warmup+timed phases. Both are gated against a freshly,
// live-measured basic_engine full-model cost.
//
// run_q2q_layerN / run_q2q_full are templates parameterized over (conv_fn,
// res_fn) (q2q_resnet18.hpp) -- the compiler generates a distinct
// instantiation per kernel pair, so orchestration code is written once and
// shared by both phases, with zero indirect-call overhead (the same property
// the original C++14 generic lambdas (auto&& params) provided).
//
// All kernels, panel builders, tensor types, and forward-step orchestration
// now live in reusable q2q_engine modules (q2q_*.hpp/.cpp); this file is thin
// CLI + harness orchestration over them.
//
// Do not touch basic_engine.
//
// Usage: ./benchmark_q2q_full_prefix_optimized <model.bin>
//   (reads artifacts/golden_input_fp32.bin as the model input)

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"
#include "resnet18_engine.hpp"
#include "ops.hpp"

#include "q2q_common.hpp"
#include "q2q_conv1_vnni.hpp"
#include "q2q_conv_vnni.hpp"
#include "q2q_epilogue.hpp"
#include "q2q_pool.hpp"
#include "q2q_quant.hpp"
#include "q2q_residual.hpp"
#include "q2q_resnet18.hpp"
#include "q2q_tensor.hpp"
#include "q2q_weight_panel.hpp"

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace q2q;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> [--threads N]\n";
        return 1;
    }

    int threads = 1;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }
    if (threads < 1) {
        std::cerr << "--threads must be >= 1\n";
        return 1;
    }
    omp_set_num_threads(threads);

    try {
        const Model model = load_model(argv[1]);

        auto ctx = build_q2q_resnet18_context(model);

        std::cout << "=== benchmark_q2q_full_prefix_optimized: tuned full-prefix forward pass ===\n";
        std::cout << "  pipeline: FP32 -> quantize -> conv1(Q2Q) -> maxpool(q8) -> layer1-4(Q2Q) "
                     "-> dequant -> avgpool+fc\n";
        std::cout << "  baseline kernels  : q2q_conv_bn_requant        + q2q_residual_add_relu\n";
        std::cout << "  optimized kernels : q2q_conv_bn_requant_apanel + q2q_residual_add_relu_avx2 "
                     "(~2.08x / ~17x paired wins standalone)\n";
        std::cout << "  conv1 uses the fast-gather + SIMD-epilogue kernel in both phases "
                     "(~4.6x standalone win, validated in benchmark_q2q_conv1_opt); "
                     "maxpool stays q8-direct/unchanged\n";
        std::cout << "  ACT_SCALE: " << ACT_SCALE << "\n";
        std::cout << "  optimized-path kernels (conv1, layer1-4 conv/residual, maxpool) "
                     "run multi-threaded via OpenMP -- requested threads: " << threads
                  << " (omp_get_max_threads() = " << omp_get_max_threads() << ")\n";

        // ---- live basic_engine full-model baseline ----
        Tensor golden_input(3, 224, 224);
        golden_input.data = read_f32_file("artifacts/golden_input_fp32.bin", 3 * 224 * 224);

        EngineWorkspace ws;
        constexpr int N_WARMUP_BASIC = 10;
        constexpr int N_TIMED_BASIC  = 30;
        for (int i = 0; i < N_WARMUP_BASIC; ++i) {
            std::vector<TimingEntry> t;
            forward_resnet18_workspace_timed(model, golden_input, ws, t);
        }
        std::vector<double> t_basic_full(N_TIMED_BASIC);
        for (int i = 0; i < N_TIMED_BASIC; ++i) {
            std::vector<TimingEntry> t;
            forward_resnet18_workspace_timed(model, golden_input, ws, t);
            double full = 0.0;
            for (const auto& e : t) full += e.ms;
            t_basic_full[i] = full;
        }
        std::vector<double> s_basic = t_basic_full;
        std::sort(s_basic.begin(), s_basic.end());
        const double basic_full_ms = s_basic[N_TIMED_BASIC / 2];
        std::cout << "\n=== basic_engine full-model cost (measured live, " << N_TIMED_BASIC << " runs) ===\n";
        print_stats("basic_full_ms", t_basic_full, N_TIMED_BASIC);

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        // ---- Phase A: straightforward dense-conv/residual kernels + tuned conv1.
        // (conv1 is the fast-gather+SIMD kernel in BOTH phases -- it's a
        // strict win, so there is no reason to keep a slower copy around;
        // this isolates the dense conv/residual microkernel tuning's
        // contribution on a like-for-like conv1.) ----
        for (int i = 0; i < N_WARMUP; ++i) run_q2q_full(ctx, q2q_conv_bn_requant, q2q_residual_add_relu, golden_input);
        std::vector<double> t_baseline(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            run_q2q_full(ctx, q2q_conv_bn_requant, q2q_residual_add_relu, golden_input);
            t_baseline[i] = now_ms() - t0;
        }
        volatile float sink_a = ctx.fc_out.data[0];
        (void)sink_a;

        // ---- Phase B: tuned kernels, with per-stage breakdown ----
        for (int i = 0; i < N_WARMUP; ++i) run_q2q_full(ctx, q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, golden_input);
        std::vector<double> t_quant(N_TIMED), t_conv1(N_TIMED), t_maxpool(N_TIMED),
                            t_layer1(N_TIMED), t_layer2(N_TIMED), t_layer3(N_TIMED), t_layer4(N_TIMED),
                            t_dequant(N_TIMED), t_avgpool_fc(N_TIMED), t_optimized(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), ctx.H_IN, ctx.W_IN, ctx.IC_C1, ctx.input_q8);
            const double t1 = now_ms();
            q2q_conv1_im2col_bn_relu_requant(ctx.input_q8, ctx.p_c1, *ctx.conv1, *ctx.bn1, ACT_SCALE, ctx.conv1_out, ctx.OH1, ctx.OW1);
            const double t2 = now_ms();
            q8_maxpool_3x3_s2_p1(ctx.conv1_out, ctx.OHp, ctx.OWp, ctx.pool_out);
            const double t3 = now_ms();
            run_q2q_layer1(ctx, q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, ctx.pool_out);
            const double t4 = now_ms();
            run_q2q_layer2(ctx, q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, ctx.l11_out);
            const double t5 = now_ms();
            run_q2q_layer3(ctx, q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, ctx.l21_out);
            const double t6 = now_ms();
            run_q2q_layer4(ctx, q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, ctx.l31_out);
            const double t7 = now_ms();
            dequantize_q8_nhwc_to_fp32_nchw(ctx.l41_out, ctx.H4, ctx.W4, ctx.OC4, ctx.dequant_out);
            const double t8 = now_ms();
            adaptive_avgpool_1x1_into(ctx.dequant_out, ctx.avgpool_out);
            linear_into(ctx.avgpool_out, *ctx.fc, ctx.fc_out);
            const double t9 = now_ms();

            t_quant[i] = t1 - t0; t_conv1[i] = t2 - t1; t_maxpool[i] = t3 - t2;
            t_layer1[i] = t4 - t3; t_layer2[i] = t5 - t4; t_layer3[i] = t6 - t5; t_layer4[i] = t7 - t6;
            t_dequant[i] = t8 - t7; t_avgpool_fc[i] = t9 - t8; t_optimized[i] = t9 - t0;
        }
        volatile float sink_b = ctx.fc_out.data[0] + ctx.dequant_out.data[0];
        (void)sink_b;

        std::cout << "\n=== Phase A: straightforward dense-conv/residual + tuned conv1 (" << N_TIMED << " runs) ===\n";
        print_stats("baseline_total_ms (q2q_conv_bn_requant + q2q_residual_add_relu)", t_baseline, N_TIMED);

        std::cout << "\n=== Phase B: tuned-kernel full prefix (" << N_TIMED << " runs) ===\n";
        print_stats("quantize_ms     (unchanged)                     ", t_quant, N_TIMED);
        print_stats("conv1_ms        (fast-gather + simd epilogue)   ", t_conv1, N_TIMED);
        print_stats("maxpool_ms      (unchanged, q8-direct)          ", t_maxpool, N_TIMED);
        print_stats("layer1_ms       (apanel conv + avx2 residual)   ", t_layer1, N_TIMED);
        print_stats("layer2_ms       (apanel conv + avx2 residual)   ", t_layer2, N_TIMED);
        print_stats("layer3_ms       (apanel conv + avx2 residual)   ", t_layer3, N_TIMED);
        print_stats("layer4_ms       (apanel conv + avx2 residual)   ", t_layer4, N_TIMED);
        print_stats("dequant_ms      (unchanged)                     ", t_dequant, N_TIMED);
        print_stats("avgpool_fc_ms   (unchanged, basic_engine ops)   ", t_avgpool_fc, N_TIMED);
        print_stats("optimized_total_ms                              ", t_optimized, N_TIMED);

        std::vector<double> sb = t_baseline, so = t_optimized;
        std::sort(sb.begin(), sb.end());
        std::sort(so.begin(), so.end());
        const double med_baseline  = sb[N_TIMED / 2];
        const double med_optimized = so[N_TIMED / 2];

        std::cout << "\n=== gate: optimized full prefix vs baseline full prefix vs basic_engine ===\n";
        std::cout << "  basic_engine full model (measured live, med)    : " << basic_full_ms << " ms\n";
        std::cout << "  baseline full prefix    (med, straightforward)  : " << med_baseline << " ms\n";
        std::cout << "  optimized full prefix   (med, tuned kernels)    : " << med_optimized << " ms\n";
        std::cout << "  tuned-kernel speedup over straightforward prefix: "
                  << (med_baseline / med_optimized) << "x\n";
        std::cout << "  optimized prefix speedup over basic_engine      : "
                  << (basic_full_ms / med_optimized) << "x\n";

        if (med_optimized < med_baseline) {
            std::cout << "  RESULT: the tuned kernels IMPROVE the full prefix -- " << med_optimized
                      << " ms vs " << med_baseline << " ms (" << (med_baseline / med_optimized)
                      << "x). The microkernel tuning paid off end-to-end, not just standalone.\n";
        } else {
            std::cout << "  RESULT: the tuned kernels do NOT improve the full prefix end-to-end ("
                      << med_optimized << " ms vs " << med_baseline << " ms) despite standalone wins "
                         "-- some interaction (cache pressure, dispatch overhead, generic-lambda "
                         "codegen) erases the microkernel-level gains at full-pipeline scale.\n";
        }
        if (med_optimized < basic_full_ms) {
            std::cout << "  (optimized prefix also beats basic_engine: " << med_optimized << " ms < "
                      << basic_full_ms << " ms, " << (basic_full_ms / med_optimized) << "x)\n";
        } else {
            std::cout << "  (optimized prefix does NOT beat basic_engine: " << med_optimized
                      << " ms >= " << basic_full_ms << " ms)\n";
        }

        std::cout << "\nC++ engine latency\n";
        std::cout << "  runs:      " << N_TIMED << "\n";
        std::cout << "  min_ms:    " << so.front() << "\n";
        std::cout << "  median_ms: " << med_optimized << "\n";
        std::cout << "  mean_ms:   " << (std::accumulate(so.begin(), so.end(), 0.0) / N_TIMED) << "\n";
        std::cout << "  p90_ms:    " << so[N_TIMED * 9 / 10] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
