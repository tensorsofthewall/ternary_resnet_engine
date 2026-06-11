// benchmark_block32_im2col_layer1.cpp -- Phase 16 Step 1: panel-im2col GEMM
// for the layer1 repeated 64->64/3x3/s1/p1/56x56 convs (p10_c1, p10_c2,
// p11_c1, p11_c2; ic_blocks=2, K=K_pad=576, OH=OW=56, N=3136).
//
// Compares the current conv_block32_n4_oc_major_avx2 (OC4_N4 OC-major)
// kernel against:
//   - materialized im2col GEMM (gemm_oc4_n4/n8_avx2 + build_im2col_3x3s1p1)
//   - panel-im2col GEMM (panel_im2col_gemm_oc4_n2/n4/n8_avx2)
// for each of the 4 layer1 convs, and reports the layer1-total (sum over 4
// convs).
//
// AVX2-only (no AVX-VNNI). Benchmark-only.
//
// Usage: ./benchmark_block32_im2col_layer1 <model.bin>

#include "block32_conv_avx2.hpp"
#include "block32_layer1_opt.hpp"
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
constexpr int IC_BLOCKS = 2;  // 64 / 32

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

        std::cout << "=== Phase 16 Step 1: panel-im2col GEMM, layer1 (64x64, 3x3 s1p1, 56x56) ===\n";

        const int OH = ctx.H1, OW = ctx.W1;

        std::vector<ConvSpec> specs = {
            {"layer1.0.conv1", &ctx.p10_c1, ctx.bn10_1, &ctx.pool_out, true},
            {"layer1.0.conv2", &ctx.p10_c2, ctx.bn10_2, &ctx.l10_mid, false},
            {"layer1.1.conv1", &ctx.p11_c1, ctx.bn11_1, &ctx.l10_out, true},
            {"layer1.1.conv2", &ctx.p11_c2, ctx.bn11_2, &ctx.l11_mid, false},
        };

        double cur_total = 0.0, im_n4_total = 0.0, im_n8_total = 0.0;
        double pn_n2_total = 0.0, pn_n4_total = 0.0, pn_n8_total = 0.0;
        size_t total_mismatches = 0;

        for (const auto& sp : specs) {
            std::cout << "  -- " << sp.name << " --\n";

            const Conv3x3S1P1PanelPlan panel = build_conv3x3s1p1_panel_plan(*sp.plan);
            const PackedBPanel B = build_packed_b_panel(panel);
            std::cout << "    out_channels=" << B.out_channels << " K_pad=" << B.K_pad << "\n";

            QTensorNHWC out_ref = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            conv_block32_scalar(*sp.plan, *sp.bn, *sp.in, sp.relu, out_ref, OH, OW);

            QTensorNHWC out_cur = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            conv_block32_n4_oc_major_avx2(*sp.plan, *sp.bn, *sp.in, sp.relu, out_cur, OH, OW);
            total_mismatches += compare_tensors("current_n4 vs scalar", out_ref, out_cur);

            const Im2colMatrix A = build_im2col_3x3s1p1(*sp.in, IC_BLOCKS, OH, OW);
            const float in_scale = sp.in->scale;

            QTensorNHWC out_im_n4 = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            gemm_oc4_n4_avx2(B, A, *sp.bn, in_scale, sp.relu, out_im_n4, OH, OW);
            total_mismatches += compare_tensors("im2col_n4 vs scalar", out_ref, out_im_n4);

            QTensorNHWC out_im_n8 = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            gemm_oc4_n8_avx2(B, A, *sp.bn, in_scale, sp.relu, out_im_n8, OH, OW);
            total_mismatches += compare_tensors("im2col_n8 vs scalar", out_ref, out_im_n8);

            QTensorNHWC out_pn_n2 = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n2_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_pn_n2, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n2 vs scalar", out_ref, out_pn_n2);

            QTensorNHWC out_pn_n4 = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n4_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_pn_n4, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n4 vs scalar", out_ref, out_pn_n4);

            QTensorNHWC out_pn_n8 = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            panel_im2col_gemm_oc4_n8_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_pn_n8, OH, OW);
            total_mismatches += compare_tensors("panel_im2col_n8 vs scalar", out_ref, out_pn_n8);

            QTensorNHWC out_bench = make_q8_tensor(OH, OW, ctx.OC1, BORDER, ACT_SCALE);
            const Stats cur_stats = time_stats([&] {
                conv_block32_n4_oc_major_avx2(*sp.plan, *sp.bn, *sp.in, sp.relu, out_bench, OH, OW);
            });
            const Stats im_n4_stats = time_stats([&] {
                gemm_oc4_n4_avx2(B, A, *sp.bn, in_scale, sp.relu, out_bench, OH, OW);
            });
            const Stats im_n8_stats = time_stats([&] {
                gemm_oc4_n8_avx2(B, A, *sp.bn, in_scale, sp.relu, out_bench, OH, OW);
            });
            const Stats pn_n2_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n2_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_bench, OH, OW);
            });
            const Stats pn_n4_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n4_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_bench, OH, OW);
            });
            const Stats pn_n8_stats = time_stats([&] {
                panel_im2col_gemm_oc4_n8_avx2(B, *sp.in, IC_BLOCKS, *sp.bn, sp.relu, out_bench, OH, OW);
            });

            print_row("current_n4      ", cur_stats);
            print_row("im2col_n4       ", im_n4_stats);
            print_row("im2col_n8       ", im_n8_stats);
            print_row("panel_im2col_n2 ", pn_n2_stats);
            print_row("panel_im2col_n4 ", pn_n4_stats);
            print_row("panel_im2col_n8 ", pn_n8_stats);

            cur_total += cur_stats.median;
            im_n4_total += im_n4_stats.median;
            im_n8_total += im_n8_stats.median;
            pn_n2_total += pn_n2_stats.median;
            pn_n4_total += pn_n4_stats.median;
            pn_n8_total += pn_n8_stats.median;
        }

        std::cout << "\n  layer1 (4 convs) totals (median sums):\n";
        std::cout << "    current_n4:      " << cur_total << " ms\n";
        std::cout << "    im2col_n4:       " << im_n4_total << " ms\n";
        std::cout << "    im2col_n8:       " << im_n8_total << " ms\n";
        std::cout << "    panel_im2col_n2: " << pn_n2_total << " ms\n";
        std::cout << "    panel_im2col_n4: " << pn_n4_total << " ms\n";
        std::cout << "    panel_im2col_n8: " << pn_n8_total << " ms\n";

        const double best = std::min({im_n4_total, im_n8_total, pn_n2_total, pn_n4_total, pn_n8_total});
        std::string best_name;
        if (best == im_n4_total) best_name = "im2col_n4";
        else if (best == im_n8_total) best_name = "im2col_n8";
        else if (best == pn_n2_total) best_name = "panel_im2col_n2";
        else if (best == pn_n4_total) best_name = "panel_im2col_n4";
        else best_name = "panel_im2col_n8";

        const double improvement = cur_total - best;
        std::cout << "  best variant: " << best_name << " (" << best << " ms)\n";
        std::cout << "  layer1 improvement vs current_n4: " << improvement << " ms\n";

        std::cout << "\n  Gates:\n";
        std::cout << "    >=1.0 ms: integrate into opt5\n";
        std::cout << "    >=2.0 ms: new headline path\n";
        std::cout << "    <0.5 ms: stop layer1 work\n";
        if (improvement >= 2.0) {
            std::cout << "  GATE: HEADLINE (" << improvement << " ms >= 2.0 ms). Candidate: " << best_name << ".\n";
        } else if (improvement >= 1.0) {
            std::cout << "  GATE: INTEGRATE (" << improvement << " ms >= 1.0 ms). Candidate: " << best_name << ".\n";
        } else if (improvement < 0.5) {
            std::cout << "  GATE: STOP (" << improvement << " ms < 0.5 ms).\n";
        } else {
            std::cout << "  GATE: marginal (" << improvement << " ms in [0.5, 1.0) ms); use judgment.\n";
        }

        std::cout << "\n  total correctness mismatches: " << total_mismatches << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
