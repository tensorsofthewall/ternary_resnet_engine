// benchmark_block32_im2col_layer3.cpp -- Phase 15 Step 3: panel-im2col GEMM
// for the layer3 repeated 256->256/3x3/s1/p1/14x14 convs (p30_c2, p31_c1,
// p31_c2; ic_blocks=8, K_pad=2304, OH=OW=14, N=196).
//
// Compares the current specialized conv3x3s1p1_256x256_14x14_oc4_n2_avx2
// kernel against panel_im2col_gemm_oc4_n1/n2/n4_avx2 for each of the 3
// repeated convs, and reports the layer3-total (sum over 3 convs).
//
// AVX2-only (no AVX-VNNI). Benchmark-only.
//
// Usage: ./benchmark_block32_im2col_layer3 <model.bin>

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
constexpr int IC_BLOCKS = 8;  // 256 / 32

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

struct ConvSpec {
    std::string name;
    const Block32ConvPlan* plan;
    const BNFusedParams* bn;
    const QTensorNHWC* in;
    bool relu;
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

        Tensor golden_input(ctx.IC_C1, ctx.H_IN, ctx.W_IN);
        golden_input.data = q2q::read_f32_file("artifacts/golden_input_fp32.bin",
                                                 static_cast<size_t>(ctx.IC_C1) * ctx.H_IN * ctx.W_IN);

        run_block32_full(ctx, golden_input);

        std::cout << "=== Phase 15 Step 3: panel-im2col GEMM, layer3 (256x256, 3x3 s1p1, 14x14) ===\n";

        const int OH = ctx.H3, OW = ctx.W3;

        std::vector<ConvSpec> specs = {
            {"layer3.0.conv2", &ctx.p30_c2, ctx.bn30_2, &ctx.l30_mid, false},
            {"layer3.1.conv1", &ctx.p31_c1, ctx.bn31_1, &ctx.l30_out, true},
            {"layer3.1.conv2", &ctx.p31_c2, ctx.bn31_2, &ctx.l31_mid, false},
        };

        double spec_total = 0.0, n1_total = 0.0, n2_total = 0.0, n4_total = 0.0;
        size_t total_mismatches = 0;

        for (const auto& sp : specs) {
            std::cout << "  -- " << sp.name << " --\n";

            const Conv3x3S1P1PanelPlan panel = build_conv3x3s1p1_panel_plan(*sp.plan);
            const PackedBPanel B = build_packed_b_panel(panel);
            std::cout << "    out_channels=" << B.out_channels << " K_pad=" << B.K_pad << "\n";

            QTensorNHWC out_ref = make_q8_tensor(OH, OW, ctx.OC3, BORDER, ACT_SCALE);
            conv_block32_scalar(*sp.plan, *sp.bn, *sp.in, sp.relu, out_ref, OH, OW);

            QTensorNHWC out_spec = make_q8_tensor(OH, OW, ctx.OC3, BORDER, ACT_SCALE);
            conv3x3s1p1_256x256_14x14_oc4_n2_avx2(panel, *sp.bn, *sp.in, sp.relu, out_spec, OH, OW);
            total_mismatches += compare_tensors("specialized_n2 vs scalar", out_ref, out_spec);

            QTensorNHWC out_n1 = make_q8_tensor(OH, OW, ctx.OC3, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n1_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_n1, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n1 vs scalar", out_ref, out_n1);

            QTensorNHWC out_n2 = make_q8_tensor(OH, OW, ctx.OC3, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n2_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_n2, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n2 vs scalar", out_ref, out_n2);

            QTensorNHWC out_n4 = make_q8_tensor(OH, OW, ctx.OC3, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n4_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_n4, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n4 vs scalar", out_ref, out_n4);

            QTensorNHWC out_bench = make_q8_tensor(OH, OW, ctx.OC3, BORDER, ACT_SCALE);
            const Stats spec_stats = time_stats([&] {
                conv3x3s1p1_256x256_14x14_oc4_n2_avx2(panel, *sp.bn, *sp.in, sp.relu, out_bench, OH, OW);
            });
            const Stats n1_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n1_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_bench, OH, OW);
            });
            const Stats n2_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n2_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_bench, OH, OW);
            });
            const Stats n4_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n4_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_bench, OH, OW);
            });

            print_row("specialized_n2  ", spec_stats);
            print_row("panel_im2col_n1 ", n1_stats);
            print_row("panel_im2col_n2 ", n2_stats);
            print_row("panel_im2col_n4 ", n4_stats);

            spec_total += spec_stats.median;
            n1_total += n1_stats.median;
            n2_total += n2_stats.median;
            n4_total += n4_stats.median;
        }

        std::cout << "\n  layer3 (3 convs) totals (median sums):\n";
        std::cout << "    specialized_n2:  " << spec_total << " ms\n";
        std::cout << "    panel_im2col_n1: " << n1_total << " ms\n";
        std::cout << "    panel_im2col_n2: " << n2_total << " ms\n";
        std::cout << "    panel_im2col_n4: " << n4_total << " ms\n";

        const double best = std::min({n1_total, n2_total, n4_total});
        const std::string best_name = (best == n1_total) ? "panel_im2col_n1" : (best == n2_total) ? "panel_im2col_n2" : "panel_im2col_n4";
        const double improvement = spec_total - best;
        std::cout << "  best panel im2col variant: " << best_name << " (" << best << " ms)\n";
        std::cout << "  layer3 improvement vs specialized_n2: " << improvement << " ms\n";

        std::cout << "\n  Gate: keep if layer3 repeated-conv total improves >= 0.5 ms\n";
        if (improvement >= 0.5) {
            std::cout << "  GATE: PASS (" << improvement << " ms >= 0.5 ms). Candidate for opt4 integration: " << best_name << ".\n";
        } else {
            std::cout << "  GATE: layer3-total gate not met (" << improvement << " ms < 0.5 ms);"
                      << " full-model improvement may still justify integration -- check opt4.\n";
        }

        std::cout << "\n  total correctness mismatches: " << total_mismatches << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
