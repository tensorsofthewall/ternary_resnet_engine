// Blocked activation layout diagnostic — layer2.1.conv2 only.
//
// Layout A: NCHWc16 = data[CB * H_pad * W_pad * 16], 16 channels innermost
// per (h, w) pixel. CB = IC/16 = 8 for IC=128.
//
// Access pattern for OW output position n at (cb, ky, kx, ci):
//   nc_data[(cb*H_pad*W_pad + (oh+ky)*W_pad + (n+kx)) * 16 + ci]
// Consecutive n values are separated by stride 16 — pure scalar gathers.
//
// Layout B: not attempted (stop condition triggered by Layout A result).
//
// Usage: ./build/benchmark_nchwc artifacts/model_ternary_i8_weights_v1.bin

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_loader.hpp"
#include "resnet18_engine.hpp"
#include "tensor.hpp"
#include "workspace.hpp"
#include "quantized/quant_utils.hpp"
#include "quantized/conv3x3_q8_pattern_ocblock2_avx2.hpp"
#include "quantized/packed_q8_3x3_kernel.hpp"

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

static std::vector<float> read_f32_file(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Short read: " + path);
    return data;
}

static Tensor load_tensor(const std::string& path, int c, int h, int w) {
    Tensor t(c, h, w);
    t.data = read_f32_file(path, static_cast<size_t>(c * h * w));
    return t;
}

static double duration_ms(
    std::chrono::time_point<std::chrono::high_resolution_clock> t0,
    std::chrono::time_point<std::chrono::high_resolution_clock> t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Stats { double min_ms = 0.0, median_ms = 0.0; };

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2]};
}

struct ErrorStats { float max_abs = 0.0f, mean_abs = 0.0f, rel_mean = 0.0f; };

static ErrorStats compute_error(const Tensor& a, const Tensor& b) {
    float max_abs = 0.0f, max_gold = 0.0f;
    double sum_abs = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const float err = std::abs(a.data[i] - b.data[i]);
        max_abs  = std::max(max_abs, err);
        sum_abs += static_cast<double>(err);
        max_gold = std::max(max_gold, std::abs(b.data[i]));
    }
    return {max_abs,
            static_cast<float>(sum_abs / a.data.size()),
            max_gold > 0.0f ? static_cast<float>(sum_abs / a.data.size()) / max_gold : 0.0f};
}

// ---------------------------------------------------------------------------
// NCHWc16 layout
//   data[(cb * H_pad * W_pad + h * W_pad + w) * 16 + ci]
//   CB = IC/16 (assumes IC % 16 == 0)
// ---------------------------------------------------------------------------

struct NCHWc16 {
    int CB, H, W;
    float scale;
    std::vector<int8_t> data;  // CB * H * W * 16 bytes

    const int8_t* pixel(int cb, int h, int w) const {
        return data.data() + static_cast<size_t>(cb * H * W + h * W + w) * 16;
    }
};

// Scalar NCHW-padded → NCHWc16-padded conversion.
// Source:  xq.data[ic * H_pad * W_pad + h * W_pad + w]
// Dest:    nc.data[(cb * H_pad * W_pad + h * W_pad + w) * 16 + ci]
// Reads sequentially per-row; writes with stride-16 scatter.
static NCHWc16 convert_to_nchwc16(const QuantTensorI8& xq, int IC) {
    const int CB     = IC / 16;
    const int H_pad  = xq.h;
    const int W_pad  = xq.w;
    const size_t plane = static_cast<size_t>(H_pad * W_pad);

    NCHWc16 nc;
    nc.CB = CB; nc.H = H_pad; nc.W = W_pad; nc.scale = xq.scale;
    nc.data.assign(static_cast<size_t>(CB) * plane * 16, 0);

    for (int ic = 0; ic < IC; ++ic) {
        const int     cb  = ic >> 4;
        const int     ci  = ic & 15;
        const int8_t* src = xq.data.data() + static_cast<size_t>(ic) * plane;
        for (size_t pos = 0; pos < plane; ++pos) {
            nc.data[(static_cast<size_t>(cb) * plane + pos) * 16 + ci] = src[pos];
        }
    }
    return nc;
}

// ---------------------------------------------------------------------------
// NCHWc sparse action plan
//   Outer: (CB, KY, KX) block — 72 blocks for layer2 (CB=8, KH=3, KW=3)
//   Inner: (OC, CI, sign) for each nonzero weight in this block
// ---------------------------------------------------------------------------

struct NCHWcAction {
    int16_t oc;
    int8_t  ci;
    int8_t  sign;  // +1 or -1
};

struct NCHWcBlock {
    int cb, ky, kx;
    std::vector<NCHWcAction> actions;
};

static std::vector<NCHWcBlock> build_nchwc_plan(const Conv2DParams& conv) {
    const int IC = conv.in_channels;
    const int OC = conv.out_channels;
    const int KH = conv.kh;
    const int KW = conv.kw;
    const int CB = IC / 16;

    std::vector<NCHWcBlock> blocks;
    blocks.reserve(static_cast<size_t>(CB * KH * KW));

    for (int cb = 0; cb < CB; ++cb) {
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                NCHWcBlock blk;
                blk.cb = cb; blk.ky = ky; blk.kx = kx;
                for (int oc = 0; oc < OC; ++oc) {
                    for (int ci = 0; ci < 16; ++ci) {
                        const int   ic = cb * 16 + ci;
                        const int8_t w = conv.weight[((oc * IC + ic) * KH + ky) * KW + kx];
                        if (w != 0)
                            blk.actions.push_back({static_cast<int16_t>(oc),
                                                   static_cast<int8_t>(ci), w});
                    }
                }
                blocks.push_back(std::move(blk));
            }
        }
    }
    return blocks;
}

// ---------------------------------------------------------------------------
// NCHWc16 kernel — layer2.1.conv2 (no ReLU, bn2 only)
//
// Accumulator acc[OC * OW] = 128*28 int32 = 14 KB, fits L1D.
// Inner loop: stride-16 byte gathers for 28 OW positions per action.
// One branch per action (sign = +1/-1): ~50% misprediction on ternary weights.
// ---------------------------------------------------------------------------

static void compute_nchwc16_layer2(
    const NCHWc16&                  nc,
    const std::vector<NCHWcBlock>&  plan,
    const BNFusedParams&            bn,
    Tensor&                         y_out)
{
    constexpr int OC = 128, OH = 28, OW = 28;
    const int H_pad = nc.H;
    const int W_pad = nc.W;

    std::array<int32_t, OC * OW> acc;

    for (int oh = 0; oh < OH; ++oh) {
        acc.fill(0);

        for (const auto& blk : plan) {
            // Pointer to the first OW position at (cb, oh+ky, kx) in NCHWc16.
            // row[n * 16 + ci] = activation of channel (cb*16+ci) at output col n.
            const int8_t* row = nc.data.data() +
                static_cast<size_t>(blk.cb * H_pad * W_pad +
                                    (oh + blk.ky) * W_pad +
                                    blk.kx) * 16;

            for (const auto& act : blk.actions) {
                const int8_t* src = row + act.ci;  // stride 16 per OW step
                int32_t*  oc_acc  = acc.data() + act.oc * OW;

                if (act.sign == 1) {
                    for (int n = 0; n < OW; ++n)
                        oc_acc[n] += static_cast<int32_t>(src[n * 16]);
                } else {
                    for (int n = 0; n < OW; ++n)
                        oc_acc[n] -= static_cast<int32_t>(src[n * 16]);
                }
            }
        }

        // BN scale/bias (no ReLU — layer2.1.bn2)
        for (int oc = 0; oc < OC; ++oc) {
            const float sc   = nc.scale * bn.scale[oc];
            const float bias = bn.bias[oc];
            float* dst = y_out.data.data() + oc * (OH * OW) + oh * OW;
            const int32_t* oc_acc = acc.data() + oc * OW;
            for (int n = 0; n < OW; ++n)
                dst[n] = oc_acc[n] * sc + bias;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_nchwc <model.bin>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_bin  = argc >= 3 ? argv[2] : "artifacts/golden_input_fp32.bin";

    std::cout << "Loading model: " << model_path << "\n";
    const Model model = load_model(model_path);

    std::cout << "Loading input: " << input_bin << "\n";
    Tensor full_input(3, 224, 224);
    full_input.data = read_f32_file(input_bin, 3 * 224 * 224);

    EngineWorkspace ws;
    std::cout << "Warmup forward passes...\n";
    for (int i = 0; i < 5; ++i)
        forward_resnet18_scalar(model, full_input, ws);

    // ── Check dump artifacts ──────────────────────────────────────────────────

    const std::string x_path = "artifacts/layer2_1_conv2_input_fp32.bin";
    const std::string g_path = "artifacts/layer2_1_conv2_output_fp32.bin";
    if (!std::filesystem::exists(x_path) || !std::filesystem::exists(g_path)) {
        std::cerr << "Missing dump artifacts. Run benchmark_packed_3x3 first.\n";
        return 1;
    }

    constexpr int IC = 128, OC = 128, OH = 28, OW = 28;
    const Tensor x_fp32 = load_tensor(x_path, IC, OH, OW);
    const Tensor golden  = load_tensor(g_path, OC, OH, OW);

    // ── Model objects ─────────────────────────────────────────────────────────

    const Conv2DParams&            conv        = model.convs.at("layer2.1.conv2");
    const PatternOCBlock2ConvPlan& bucket_plan = model.pattern_ocblock2_plans.at("layer2.1.conv2");
    const BNFusedParams&           bn          = model.fused_bns.at("layer2.1.bn2");

    constexpr int EXTRA_PAD = 4;

    const PackedSparseQ8Plan   sparse_plan = build_packed_sparse_q8_plan(conv);
    const std::vector<NCHWcBlock> nchwc_plan = build_nchwc_plan(conv);

    // Count total actions
    size_t total_actions = 0;
    for (const auto& blk : nchwc_plan) total_actions += blk.actions.size();

    QuantTensorI8 xq;
    xq.data.reserve(static_cast<size_t>(IC) * (OH + 2) * (OW + 2 + EXTRA_PAD));

    std::vector<int8_t> A_panel(static_cast<size_t>(IC * 9) * 32);

    Tensor y_bucket(OC, OH, OW);
    Tensor y_sparse(OC, OH, OW);
    Tensor y_nchwc(OC, OH, OW);

    // ── Warmup ────────────────────────────────────────────────────────────────

    quantize_pad_tensor_symmetric_i8_asymmetric_into(
        x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);

    NCHWc16 nc_warm = convert_to_nchwc16(xq, IC);

    for (int i = 0; i < 5; ++i) {
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);
        conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_avx2(xq, conv, bucket_plan, bn, y_bucket);
        for (int oh = 0; oh < OH; ++oh) {
            pack_a_panel_layer2(xq, oh, A_panel.data());
            packed_q8_3x3_s1_layer2_bn_avx2(A_panel.data(), sparse_plan, bn, xq.scale, oh, y_sparse);
        }
        NCHWc16 nc = convert_to_nchwc16(xq, IC);
        compute_nchwc16_layer2(nc, nchwc_plan, bn, y_nchwc);
    }

    // ── Timed runs ────────────────────────────────────────────────────────────

    constexpr int N_RUNS = 100;

    std::vector<double> bq(N_RUNS), bc(N_RUNS), bt(N_RUNS);
    std::vector<double> sq(N_RUNS), sc_v(N_RUNS), st(N_RUNS);
    std::vector<double> nq(N_RUNS), nc_v(N_RUNS), nn(N_RUNS), nt(N_RUNS);

    // Bucket kernel
    for (int i = 0; i < N_RUNS; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);
        const auto t1 = std::chrono::high_resolution_clock::now();
        conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_avx2(xq, conv, bucket_plan, bn, y_bucket);
        const auto t2 = std::chrono::high_resolution_clock::now();
        bq[i] = duration_ms(t0, t1);
        bc[i] = duration_ms(t1, t2);
        bt[i] = duration_ms(t0, t2);
    }

    // Packed sparse OC_TILE=1
    for (int i = 0; i < N_RUNS; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);
        const auto t1 = std::chrono::high_resolution_clock::now();
        for (int oh = 0; oh < OH; ++oh) {
            pack_a_panel_layer2(xq, oh, A_panel.data());
            packed_q8_3x3_s1_layer2_bn_avx2(A_panel.data(), sparse_plan, bn, xq.scale, oh, y_sparse);
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        sq[i] = duration_ms(t0, t1);
        sc_v[i] = duration_ms(t1, t2);
        st[i] = duration_ms(t0, t2);
    }

    // NCHWc16 Layout A
    for (int i = 0; i < N_RUNS; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);
        const auto t1 = std::chrono::high_resolution_clock::now();
        NCHWc16 nc = convert_to_nchwc16(xq, IC);
        const auto t2 = std::chrono::high_resolution_clock::now();
        compute_nchwc16_layer2(nc, nchwc_plan, bn, y_nchwc);
        const auto t3 = std::chrono::high_resolution_clock::now();
        nq[i]  = duration_ms(t0, t1);
        nc_v[i] = duration_ms(t1, t2);
        nn[i]  = duration_ms(t2, t3);
        nt[i]  = duration_ms(t0, t3);
    }

    // ── Results ───────────────────────────────────────────────────────────────

    const Stats bqs = compute_stats(bq);
    const Stats bcs = compute_stats(bc);
    const Stats bts = compute_stats(bt);
    const Stats sqs = compute_stats(sq);
    const Stats scs = compute_stats(sc_v);
    const Stats sts = compute_stats(st);
    const Stats nqs = compute_stats(nq);
    const Stats ncs = compute_stats(nc_v);
    const Stats nns = compute_stats(nn);
    const Stats nts = compute_stats(nt);

    const ErrorStats err_bucket = compute_error(y_bucket, golden);
    const ErrorStats err_sparse = compute_error(y_sparse, y_bucket);
    const ErrorStats err_nchwc  = compute_error(y_nchwc,  y_bucket);

    const int H_pad = xq.h, W_pad = xq.w;
    const size_t nchwc_bytes = static_cast<size_t>(IC / 16) * H_pad * W_pad * 16;

    std::cout << "\n=== Blocked activation layout diagnostic ===\n";
    std::cout << "Layer:  layer2.1.conv2  IC=" << IC << " OC=" << OC
              << " OH=" << OH << " OW=" << OW << "\n";
    std::cout << "Layout A: NCHWc16 [CB=" << (IC/16) << "][H=" << H_pad
              << "][W=" << W_pad << "][16]\n";
    std::cout << "NCHWc16 size:     " << (nchwc_bytes / 1024) << " KB\n";
    std::cout << "NCHWc plan blocks: " << nchwc_plan.size()
              << "  total actions: " << total_actions << "\n\n";

    std::cout << "--- bucket kernel (" << N_RUNS << " runs) ---\n";
    std::cout << "  quantize:  min=" << bqs.min_ms << " ms  median=" << bqs.median_ms << " ms\n";
    std::cout << "  compute:   min=" << bcs.min_ms << " ms  median=" << bcs.median_ms << " ms\n";
    std::cout << "  total:     min=" << bts.min_ms << " ms  median=" << bts.median_ms << " ms\n\n";

    std::cout << "--- packed sparse OC_TILE=1 ---\n";
    std::cout << "  quantize:  min=" << sqs.min_ms << " ms  median=" << sqs.median_ms << " ms\n";
    std::cout << "  compute:   min=" << scs.min_ms << " ms  median=" << scs.median_ms << " ms\n";
    std::cout << "  total:     min=" << sts.min_ms << " ms  median=" << sts.median_ms << " ms\n";
    std::cout << "  speedup vs bucket (total): " << bts.min_ms / sts.min_ms << "x\n\n";

    std::cout << "--- NCHWc16 Layout A (stride-16 gather) ---\n";
    std::cout << "  quantize:  min=" << nqs.min_ms << " ms  median=" << nqs.median_ms << " ms\n";
    std::cout << "  convert:   min=" << ncs.min_ms << " ms  median=" << ncs.median_ms << " ms\n";
    std::cout << "  compute:   min=" << nns.min_ms << " ms  median=" << nns.median_ms << " ms\n";
    std::cout << "  total:     min=" << nts.min_ms << " ms  median=" << nts.median_ms << " ms\n";
    std::cout << "  speedup vs bucket (compute): " << bcs.min_ms / nns.min_ms << "x\n";
    std::cout << "  speedup vs bucket (total):   " << bts.min_ms / nts.min_ms << "x\n\n";

    std::cout << "--- correctness (vs bucket output) ---\n";
    std::cout << "bucket vs golden:  max_abs=" << err_bucket.max_abs
              << "  rel_mean=" << err_bucket.rel_mean << "\n";
    std::cout << "sparse vs bucket:  max_abs=" << err_sparse.max_abs
              << "  rel_mean=" << err_sparse.rel_mean << "\n";
    std::cout << "nchwc  vs bucket:  max_abs=" << err_nchwc.max_abs
              << "  rel_mean=" << err_nchwc.rel_mean << "\n\n";

    std::cout << "--- verdict ---\n";
    const bool compute_beats_bucket = nns.min_ms < bcs.min_ms;
    const bool total_beats_bucket   = nts.min_ms < bts.min_ms;
    const bool meets_threshold      = nts.min_ms <= 1.7 ||
                                       bts.min_ms / nts.min_ms >= 1.5;

    if (meets_threshold) {
        std::cout << "PASS: NCHWc16 total " << nts.min_ms
                  << " ms meets threshold (<=1.7ms or >=1.5x). Consider Layout B.\n";
    } else if (!compute_beats_bucket) {
        std::cout << "STOP (compute worse): NCHWc16 compute " << nns.min_ms
                  << " ms >= bucket compute " << bcs.min_ms
                  << " ms. Stride-16 gathers + sign-branch misprediction dominate.\n";
        std::cout << "Layout B: not attempted.\n";
    } else if (!total_beats_bucket) {
        std::cout << "STOP (convert overhead): NCHWc16 compute " << nns.min_ms
                  << " ms < bucket compute " << bcs.min_ms
                  << " ms, but convert cost (" << ncs.min_ms
                  << " ms) kills total (" << nts.min_ms << " ms vs bucket "
                  << bts.min_ms << " ms).\n";
        std::cout << "NCHWc16 would require whole-engine layout propagation to avoid convert overhead.\n";
        std::cout << "Layout B: not attempted.\n";
    } else {
        std::cout << "STOP (below threshold): total " << nts.min_ms
                  << " ms > 1.7 ms, speedup " << bts.min_ms / nts.min_ms << "x < 1.5x.\n";
        std::cout << "Layout B: not attempted.\n";
    }

    return 0;
}
