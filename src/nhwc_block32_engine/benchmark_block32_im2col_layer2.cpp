// benchmark_block32_im2col_layer2.cpp -- experimental im2col/GEMM framework
// for the AVX2 block32 engine, tested on the layer2.0.conv2 shape (IC=128,
// OC=128, 3x3, stride1, pad1, OH=OW=28, K=K_pad=1152, N=784).
//
// Compares the current specialized conv3x3s1p1_128x128_28x28_oc4_n2_avx2
// kernel against:
//   - a materialized im2col matrix A[N][K_pad] + GEMM-style OC4 microkernel
//     (gemm_oc4_n1/n2/n4_avx2), timed as im2col_build_ms / gemm_ms / total_ms
//   - a panel-im2col variant that builds a small A_panel[N_TILE][K_pad] per
//     tile of output pixels, avoiding full materialization
//     (panel_im2col_gemm_oc4_n2/n4_avx2)
//
// AVX2-only (no AVX-VNNI). Benchmark-only -- does not modify basic_engine,
// q2q_engine, or the existing opt3 full-model path.
//
// Usage: ./benchmark_block32_im2col_layer2 <model.bin>

#include "block32_conv_avx2.hpp"
#include "block32_layer234_opt.hpp"
#include "block32_plan.hpp"
#include "block32_resnet18.hpp"
#include "im2col_gemm.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include "nhwc_common.hpp"
#include "q2q_common.hpp"
#include "q2q_tensor.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nhwc_block32;
using q2q::make_q8_tensor;
using q2q::QTensorNHWC;

namespace {

constexpr int BORDER = 1;
constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 100;

struct Stats {
    double min;
    double median;
    double p90;
};

Stats time_stats(const std::function<void()>& fn) {
    for (int i = 0; i < N_WARMUP; ++i) fn();
    std::vector<double> t(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        const double t0 = now_ms();
        fn();
        t[i] = now_ms() - t0;
    }
    std::sort(t.begin(), t.end());
    Stats s;
    s.min = t.front();
    s.median = t[t.size() / 2];
    s.p90 = t[static_cast<size_t>(static_cast<double>(t.size()) * 0.9)];
    return s;
}

void print_row(const std::string& label, const Stats& s) {
    std::cout << "    " << label << ": min=" << s.min << " ms, median=" << s.median
              << " ms, p90=" << s.p90 << " ms\n";
}

size_t compare_tensors(const std::string& label, const QTensorNHWC& ref, const QTensorNHWC& got) {
    size_t mismatches = 0;
    int max_abs_diff = 0;
    for (size_t i = 0; i < ref.data.size(); ++i) {
        const int diff = std::abs(static_cast<int>(ref.data[i]) - static_cast<int>(got.data[i]));
        if (diff != 0) ++mismatches;
        max_abs_diff = std::max(max_abs_diff, diff);
    }
    std::cout << "    [" << label << "] " << mismatches << " / " << ref.data.size()
              << " mismatches, max_abs_diff=" << max_abs_diff << "\n";
    return mismatches;
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

        Tensor golden_input(ctx.IC_C1, ctx.H_IN, ctx.W_IN);
        golden_input.data = q2q::read_f32_file("artifacts/golden_input_fp32.bin",
                                                 static_cast<size_t>(ctx.IC_C1) * ctx.H_IN * ctx.W_IN);

        // One full forward pass populates ctx.* with real intermediate
        // activations so the im2col/correctness/timing all use realistic data.
        run_block32_full(ctx, golden_input);

        std::cout << "=== im2col/GEMM experiment: layer2.0.conv2 (128x128, 3x3 s1p1, 28x28) ===\n";

        const QTensorNHWC& in = ctx.l20_mid;
        const BNFusedParams& bn = *ctx.bn20_2;
        const int OH = ctx.H2, OW = ctx.W2;
        const bool relu = false;
        const int ic_blocks = 4;

        // ---- Step 1: packed B layout ----
        std::cout << "\n  Step 1: packed B layout\n";
        const Conv3x3S1P1PanelPlan panel = build_conv3x3s1p1_panel_plan(ctx.p20_c2);
        const PackedBPanel B = build_packed_b_panel(panel);
        std::cout << "    out_channels=" << B.out_channels << " K_pad=" << B.K_pad
                  << " (expected K_pad=1152)\n";
        {
            size_t b_mismatches = 0;
            const int n_oc_tiles = panel.out_channels / 4;
            const int k_blocks = panel.ic_blocks * 9;
            for (int t = 0; t < n_oc_tiles; ++t) {
                for (int kb = 0; kb < k_blocks; ++kb) {
                    for (int lane = 0; lane < 4; ++lane) {
                        const int8_t* a = panel.w[static_cast<size_t>(t)][static_cast<size_t>(kb)][static_cast<size_t>(lane)].data();
                        const int8_t* b = B.block(t, kb, lane);
                        for (int i = 0; i < 32; ++i) {
                            if (a[i] != b[i]) ++b_mismatches;
                        }
                    }
                }
            }
            std::cout << "    packed-vs-unpacked weight mismatches: " << b_mismatches << "\n";
        }

        // ---- Step 2: materialized im2col ----
        std::cout << "\n  Step 2: materialized im2col\n";
        Im2colMatrix A = build_im2col_3x3s1p1(in, ic_blocks, OH, OW);
        std::cout << "    N=" << A.N << " K_pad=" << A.K_pad
                  << " (expected N=784, K_pad=1152)\n";

        // ---- Step 3/4: GEMM microkernels + correctness ----
        std::cout << "\n  Step 4: correctness\n";
        QTensorNHWC out_ref = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
        conv_block32_scalar(ctx.p20_c2, bn, in, relu, out_ref, OH, OW);

        QTensorNHWC out_spec = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
        conv3x3s1p1_128x128_28x28_oc4_n2_avx2(panel, bn, in, relu, out_spec, OH, OW);
        size_t total_mismatches = compare_tensors("specialized_n2 vs scalar", out_ref, out_spec);

        QTensorNHWC out_n1 = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
        gemm_oc4_n1_avx2(B, A, bn, in.scale, relu, out_n1, OH, OW);
        total_mismatches += compare_tensors("im2col_n1 vs scalar", out_ref, out_n1);
        total_mismatches += compare_tensors("im2col_n1 vs specialized_n2", out_spec, out_n1);

        QTensorNHWC out_n2 = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
        gemm_oc4_n2_avx2(B, A, bn, in.scale, relu, out_n2, OH, OW);
        total_mismatches += compare_tensors("im2col_n2 vs scalar", out_ref, out_n2);
        total_mismatches += compare_tensors("im2col_n2 vs specialized_n2", out_spec, out_n2);

        QTensorNHWC out_n4 = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
        gemm_oc4_n4_avx2(B, A, bn, in.scale, relu, out_n4, OH, OW);
        total_mismatches += compare_tensors("im2col_n4 vs scalar", out_ref, out_n4);
        total_mismatches += compare_tensors("im2col_n4 vs specialized_n2", out_spec, out_n4);

        // ---- Step 5: benchmark ----
        std::cout << "\n  Step 5: benchmark (median/min/p90 over " << N_TIMED << " runs)\n";

        QTensorNHWC out_bench = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);

        std::cout << "  -- current specialized layer2 N2 --\n";
        const Stats spec_stats = time_stats([&] {
            conv3x3s1p1_128x128_28x28_oc4_n2_avx2(panel, bn, in, relu, out_bench, OH, OW);
        });
        print_row("specialized_n2", spec_stats);

        std::cout << "  -- materialized im2col: build only --\n";
        const Stats build_stats = time_stats([&] {
            A = build_im2col_3x3s1p1(in, ic_blocks, OH, OW);
        });
        print_row("im2col_build", build_stats);

        std::cout << "  -- materialized im2col: gemm only (A prebuilt) --\n";
        A = build_im2col_3x3s1p1(in, ic_blocks, OH, OW);
        const Stats gemm_n1_stats = time_stats([&] { gemm_oc4_n1_avx2(B, A, bn, in.scale, relu, out_bench, OH, OW); });
        const Stats gemm_n2_stats = time_stats([&] { gemm_oc4_n2_avx2(B, A, bn, in.scale, relu, out_bench, OH, OW); });
        const Stats gemm_n4_stats = time_stats([&] { gemm_oc4_n4_avx2(B, A, bn, in.scale, relu, out_bench, OH, OW); });
        print_row("gemm_n1", gemm_n1_stats);
        print_row("gemm_n2", gemm_n2_stats);
        print_row("gemm_n4", gemm_n4_stats);

        std::cout << "  -- materialized im2col: total (build + gemm) --\n";
        const Stats total_n1_stats = time_stats([&] {
            Im2colMatrix A_local = build_im2col_3x3s1p1(in, ic_blocks, OH, OW);
            gemm_oc4_n1_avx2(B, A_local, bn, in.scale, relu, out_bench, OH, OW);
        });
        const Stats total_n2_stats = time_stats([&] {
            Im2colMatrix A_local = build_im2col_3x3s1p1(in, ic_blocks, OH, OW);
            gemm_oc4_n2_avx2(B, A_local, bn, in.scale, relu, out_bench, OH, OW);
        });
        const Stats total_n4_stats = time_stats([&] {
            Im2colMatrix A_local = build_im2col_3x3s1p1(in, ic_blocks, OH, OW);
            gemm_oc4_n4_avx2(B, A_local, bn, in.scale, relu, out_bench, OH, OW);
        });
        print_row("im2col_total_n1", total_n1_stats);
        print_row("im2col_total_n2", total_n2_stats);
        print_row("im2col_total_n4", total_n4_stats);

        // ---- Decision gate ----
        std::cout << "\n  Decision gate (vs specialized_n2 median=" << spec_stats.median << " ms):\n";
        const double best_total = std::min({total_n1_stats.median, total_n2_stats.median, total_n4_stats.median});
        const std::string best_total_name =
            (best_total == total_n1_stats.median) ? "im2col_total_n1" :
            (best_total == total_n2_stats.median) ? "im2col_total_n2" : "im2col_total_n4";
        const double best_gemm = std::min({gemm_n1_stats.median, gemm_n2_stats.median, gemm_n4_stats.median});
        const double total_pct = (spec_stats.median - best_total) / spec_stats.median * 100.0;
        const double gemm_pct = (spec_stats.median - best_gemm) / spec_stats.median * 100.0;

        std::cout << "    best materialized total: " << best_total_name << " = " << best_total
                  << " ms (" << total_pct << "% vs specialized_n2)\n";
        std::cout << "    best gemm-only:          " << best_gemm
                  << " ms (" << gemm_pct << "% vs specialized_n2)\n";

        bool run_panel_im2col = false;
        if (total_pct >= 5.0) {
            std::cout << "    GATE: materialized im2col total beats specialized_n2 by >= 5%. Continue.\n";
            run_panel_im2col = true;
        } else if (gemm_pct >= 5.0 && total_pct < 5.0) {
            std::cout << "    GATE: gemm-only is faster (>=5%) but im2col_build overhead makes the\n"
                      << "    total slower. Implementing panel im2col (Step 6) next.\n";
            run_panel_im2col = true;
        } else if (total_pct <= -5.0) {
            std::cout << "    GATE: materialized im2col total is >= 5% slower than specialized_n2.\n"
                      << "    NEGATIVE RESULT -- stopping here, documenting and not integrating.\n";
        } else {
            std::cout << "    GATE: result is within +/-5% of specialized_n2 (no clear win or loss).\n"
                      << "    Running panel im2col (Step 6) anyway for completeness.\n";
            run_panel_im2col = true;
        }

        // ---- Step 6: panel im2col (run for completeness regardless of gate
        // outcome above, so the negative-result writeup -- if any -- has
        // numbers for both materialized and panel variants). OW=28 is not a
        // multiple of 8, so N_TILE=8 is skipped. ----
        if (run_panel_im2col) {
            std::cout << "\n  Step 6: panel im2col (A_panel[N_TILE][K_pad], built per-tile)\n";

            QTensorNHWC out_panel_n2 = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n2_avx2(B, in, ic_blocks, bn, relu, out_panel_n2, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n2 vs scalar", out_ref, out_panel_n2);

            QTensorNHWC out_panel_n4 = make_q8_tensor(OH, OW, ctx.OC2, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n4_avx2(B, in, ic_blocks, bn, relu, out_panel_n4, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n4 vs scalar", out_ref, out_panel_n4);

            std::cout << "    OW=" << OW << " is not a multiple of 8 -- skipping panel_im2col N_TILE=8.\n";

            const Stats panel_n2_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n2_avx2(B, in, ic_blocks, bn, relu, out_bench, OH, OW);
            });
            const Stats panel_n4_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n4_avx2(B, in, ic_blocks, bn, relu, out_bench, OH, OW);
            });
            print_row("panel_im2col_n2", panel_n2_stats);
            print_row("panel_im2col_n4", panel_n4_stats);

            const double best_panel = std::min(panel_n2_stats.median, panel_n4_stats.median);
            const std::string best_panel_name = (best_panel == panel_n2_stats.median) ? "panel_im2col_n2" : "panel_im2col_n4";
            const double panel_pct = (spec_stats.median - best_panel) / spec_stats.median * 100.0;
            std::cout << "    best panel im2col: " << best_panel_name << " = " << best_panel
                      << " ms (" << panel_pct << "% vs specialized_n2)\n";

            if (panel_pct >= 5.0) {
                std::cout << "    GATE: panel im2col beats specialized_n2 by >= 5%. KEEP.\n";
            } else {
                std::cout << "    GATE: panel im2col does not beat specialized_n2 by >= 5%. DO NOT integrate.\n";
            }
        }

        std::cout << "\n  total correctness mismatches: " << total_mismatches << "\n";
        std::cout << "\n  NOTE: per the experiment spec, no integration into\n"
                  << "  benchmark_block32_full_model_opt3 happens here regardless of outcome.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
