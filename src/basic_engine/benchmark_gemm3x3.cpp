// Dense ternary GEMM-style microkernel experiment — layer2.1.conv2 only.
//
// Tests whether a custom dense M_TILE=2 kernel with shared A_panel loading
// beats the existing q8 OCBlock2 bucket kernel on layer2 (IC=128, 28x28).
//
// Kernel design:
//   A_panel[k * N_PAD + n], K=1152, N_TILE=28, N_PAD=32 (reused from packed sparse)
//   W_dense[oc * K + k], int8_t in {-1, 0, +1}
//   M_TILE=2: two OCs share one A_panel traversal per output row
//   No sign_epi32. Branch only: if w==+1 add; if w==-1 sub; skip zero.
//
// Usage:
//   ./build/benchmark_gemm3x3 artifacts/model_ternary_i8_weights_v1.bin

#include <algorithm>
#include <chrono>
#include <cmath>
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
// Helpers (shared with benchmark_packed_3x3)
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

struct Stats {
    double min_ms    = 0.0;
    double median_ms = 0.0;
};

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2]};
}

struct ErrorStats {
    float max_abs  = 0.0f;
    float mean_abs = 0.0f;
    float rel_mean = 0.0f;
};

static ErrorStats compute_error(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size())
        throw std::runtime_error("Size mismatch in compute_error");
    float max_abs   = 0.0f;
    double sum_abs  = 0.0;
    float max_gold  = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i) {
        const float err = std::abs(a.data[i] - b.data[i]);
        max_abs  = std::max(max_abs, err);
        sum_abs += static_cast<double>(err);
        max_gold = std::max(max_gold, std::abs(b.data[i]));
    }
    ErrorStats e;
    e.max_abs  = max_abs;
    e.mean_abs = static_cast<float>(sum_abs / static_cast<double>(a.data.size()));
    e.rel_mean = max_gold > 0.0f ? e.mean_abs / max_gold : 0.0f;
    return e;
}

// ---------------------------------------------------------------------------
// W_dense builder: W_dense[oc * K + k], int8_t in {-1, 0, +1}
// k = (ic * KH + ky) * KW + kx  (same ordering as A_panel K dimension)
// ---------------------------------------------------------------------------

static std::vector<int8_t> build_w_dense(const Conv2DParams& conv) {
    const int IC = conv.in_channels;
    const int OC = conv.out_channels;
    const int KH = conv.kh;
    const int KW = conv.kw;
    const int K  = IC * KH * KW;

    std::vector<int8_t> W(static_cast<size_t>(OC) * K, 0);
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    const int k = (ic * KH + ky) * KW + kx;
                    W[oc * K + k] = conv.weight[((oc * IC + ic) * KH + ky) * KW + kx];
                }
            }
        }
    }
    return W;
}

// ---------------------------------------------------------------------------
// Dense GEMM M_TILE=2 kernel — layer2.1.conv2
//
// Processes 2 OCs per A_panel pass. A_panel is shared between oc0 and oc1.
// 8 int32 accumulators per OC pair (4 per OC × 2 OCs = 8 YMM regs).
// 4 temporaries for expanded panel row = 12 YMM total. No spills on AVX2.
//
// conv_bn only (no ReLU) — layer2.1.bn2 has no trailing relu in the engine.
// ---------------------------------------------------------------------------

static void compute_gemm_layer2_m2(
    const int8_t*        A_panel,
    const int8_t*        W_dense,
    const BNFusedParams& bn,
    float                xq_scale,
    int                  oh,
    int                  K,
    Tensor&              y_out)
{
    constexpr int OC = 128, OH = 28, OW = 28;

    for (int oc0 = 0; oc0 < OC; oc0 += 2) {
        const int    oc1 = oc0 + 1;
        const int8_t* w0 = W_dense + oc0 * K;
        const int8_t* w1 = W_dense + oc1 * K;

        __m256i acc0_lo   = _mm256_setzero_si256();
        __m256i acc0_mid1 = _mm256_setzero_si256();
        __m256i acc0_mid2 = _mm256_setzero_si256();
        __m256i acc0_hi   = _mm256_setzero_si256();
        __m256i acc1_lo   = _mm256_setzero_si256();
        __m256i acc1_mid1 = _mm256_setzero_si256();
        __m256i acc1_mid2 = _mm256_setzero_si256();
        __m256i acc1_hi   = _mm256_setzero_si256();

        for (int k = 0; k < K; ++k) {
            const int8_t wv0 = w0[k];
            const int8_t wv1 = w1[k];
            if (wv0 == 0 && wv1 == 0) continue;

            const __m128i raw_lo = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 32));
            const __m128i raw_hi = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 32 + 16));

            const __m256i xlo   = _mm256_cvtepi8_epi32(raw_lo);
            const __m256i xmid1 = _mm256_cvtepi8_epi32(_mm_srli_si128(raw_lo, 8));
            const __m256i xmid2 = _mm256_cvtepi8_epi32(raw_hi);
            const __m256i xhi   = _mm256_cvtepi8_epi32(_mm_srli_si128(raw_hi, 8));

            if (wv0 == 1) {
                acc0_lo   = _mm256_add_epi32(acc0_lo,   xlo);
                acc0_mid1 = _mm256_add_epi32(acc0_mid1, xmid1);
                acc0_mid2 = _mm256_add_epi32(acc0_mid2, xmid2);
                acc0_hi   = _mm256_add_epi32(acc0_hi,   xhi);
            } else if (wv0 == -1) {
                acc0_lo   = _mm256_sub_epi32(acc0_lo,   xlo);
                acc0_mid1 = _mm256_sub_epi32(acc0_mid1, xmid1);
                acc0_mid2 = _mm256_sub_epi32(acc0_mid2, xmid2);
                acc0_hi   = _mm256_sub_epi32(acc0_hi,   xhi);
            }

            if (wv1 == 1) {
                acc1_lo   = _mm256_add_epi32(acc1_lo,   xlo);
                acc1_mid1 = _mm256_add_epi32(acc1_mid1, xmid1);
                acc1_mid2 = _mm256_add_epi32(acc1_mid2, xmid2);
                acc1_hi   = _mm256_add_epi32(acc1_hi,   xhi);
            } else if (wv1 == -1) {
                acc1_lo   = _mm256_sub_epi32(acc1_lo,   xlo);
                acc1_mid1 = _mm256_sub_epi32(acc1_mid1, xmid1);
                acc1_mid2 = _mm256_sub_epi32(acc1_mid2, xmid2);
                acc1_hi   = _mm256_sub_epi32(acc1_hi,   xhi);
            }
        }

        // BN scale/bias — no ReLU (layer2.1.bn2)
        {
            const __m256 sc   = _mm256_set1_ps(xq_scale * bn.scale[oc0]);
            const __m256 bias = _mm256_set1_ps(bn.bias[oc0]);
            float* dst = y_out.data.data() + oc0 * (OH * OW) + oh * OW;
            _mm256_storeu_ps(dst,      _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0_lo),   sc, bias));
            _mm256_storeu_ps(dst + 8,  _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0_mid1), sc, bias));
            _mm256_storeu_ps(dst + 16, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0_mid2), sc, bias));
            _mm_storeu_ps(   dst + 24, _mm256_castps256_ps128(
                                _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0_hi), sc, bias)));
        }
        {
            const __m256 sc   = _mm256_set1_ps(xq_scale * bn.scale[oc1]);
            const __m256 bias = _mm256_set1_ps(bn.bias[oc1]);
            float* dst = y_out.data.data() + oc1 * (OH * OW) + oh * OW;
            _mm256_storeu_ps(dst,      _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1_lo),   sc, bias));
            _mm256_storeu_ps(dst + 8,  _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1_mid1), sc, bias));
            _mm256_storeu_ps(dst + 16, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1_mid2), sc, bias));
            _mm_storeu_ps(   dst + 24, _mm256_castps256_ps128(
                                _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1_hi), sc, bias)));
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_gemm3x3 <model.bin>\n";
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

    // ── Load dump artifacts ───────────────────────────────────────────────────

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

    constexpr int K             = IC * 3 * 3;  // 1152
    constexpr int EXTRA_PAD     = 4;            // right pad beyond conv.pad_w

    // Build plans outside timing
    const PackedSparseQ8Plan sparse_plan = build_packed_sparse_q8_plan(conv);
    const std::vector<int8_t> W_dense    = build_w_dense(conv);

    // Count sparsity for reporting
    int nonzero = 0;
    for (int8_t w : W_dense) nonzero += (w != 0) ? 1 : 0;
    const double sparsity = 1.0 - static_cast<double>(nonzero) / W_dense.size();

    QuantTensorI8 xq;
    xq.data.reserve(static_cast<size_t>(IC) * (OH + 2) * (OW + 2 + EXTRA_PAD));

    // N_PAD=32 for layer2
    std::vector<int8_t> A_panel(static_cast<size_t>(K) * 32);

    Tensor y_bucket(OC, OH, OW);
    Tensor y_sparse(OC, OH, OW);
    Tensor y_gemm(OC, OH, OW);

    // ── Warmup ────────────────────────────────────────────────────────────────

    for (int i = 0; i < 5; ++i) {
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);
        conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_avx2(xq, conv, bucket_plan, bn, y_bucket);
        for (int oh = 0; oh < OH; ++oh) {
            pack_a_panel_layer2(xq, oh, A_panel.data());
            packed_q8_3x3_s1_layer2_bn_avx2(A_panel.data(), sparse_plan, bn, xq.scale, oh, y_sparse);
            compute_gemm_layer2_m2(A_panel.data(), W_dense.data(), bn, xq.scale, oh, K, y_gemm);
        }
    }

    // ── Timed runs ────────────────────────────────────────────────────────────

    constexpr int N_RUNS = 100;

    std::vector<double> bq(N_RUNS), bc(N_RUNS), bt(N_RUNS);
    std::vector<double> sq(N_RUNS), sc_v(N_RUNS), st(N_RUNS);
    std::vector<double> gq(N_RUNS), gc(N_RUNS), gt(N_RUNS);

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

    // Dense GEMM M_TILE=2
    for (int i = 0; i < N_RUNS; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + EXTRA_PAD, xq);
        const auto t1 = std::chrono::high_resolution_clock::now();
        for (int oh = 0; oh < OH; ++oh) {
            pack_a_panel_layer2(xq, oh, A_panel.data());
            compute_gemm_layer2_m2(A_panel.data(), W_dense.data(), bn, xq.scale, oh, K, y_gemm);
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        gq[i] = duration_ms(t0, t1);
        gc[i] = duration_ms(t1, t2);
        gt[i] = duration_ms(t0, t2);
    }

    // ── Results ───────────────────────────────────────────────────────────────

    const Stats bqs = compute_stats(bq);
    const Stats bcs = compute_stats(bc);
    const Stats bts = compute_stats(bt);
    const Stats sqs = compute_stats(sq);
    const Stats scs = compute_stats(sc_v);
    const Stats sts = compute_stats(st);
    const Stats gqs = compute_stats(gq);
    const Stats gcs = compute_stats(gc);
    const Stats gts = compute_stats(gt);

    const ErrorStats err_bucket = compute_error(y_bucket, golden);
    const ErrorStats err_sparse = compute_error(y_sparse, y_bucket);
    const ErrorStats err_gemm   = compute_error(y_gemm,   y_bucket);

    std::cout << "\n=== Dense GEMM-style ternary kernel experiment ===\n";
    std::cout << "Layer:  layer2.1.conv2\n";
    std::cout << "Shape:  IC=" << IC << " OC=" << OC << " OH=" << OH << " OW=" << OW << "\n";
    std::cout << "K=" << K << "  N=" << (OH * OW) << "  M=" << OC << "\n";
    std::cout << "W_dense sparsity: " << (sparsity * 100.0) << "% zeros\n";
    std::cout << "A_panel size: " << (K * 32 / 1024) << " KB (K=" << K << " x N_PAD=32)\n\n";

    auto print_row = [](const std::string& label,
                        const Stats& qs, const Stats& cs, const Stats& ts,
                        double bcs_min, double bts_min)
    {
        std::cout << label << "\n";
        std::cout << "  quantize:  min=" << qs.min_ms    << " ms  median=" << qs.median_ms << " ms\n";
        std::cout << "  compute:   min=" << cs.min_ms    << " ms  median=" << cs.median_ms << " ms\n";
        std::cout << "  total:     min=" << ts.min_ms    << " ms  median=" << ts.median_ms << " ms\n";
        std::cout << "  speedup (compute): " << bcs_min / cs.min_ms << "x\n";
        std::cout << "  speedup (total):   " << bts_min / ts.min_ms << "x\n";
    };

    std::cout << "--- bucket kernel (" << N_RUNS << " runs) ---\n";
    std::cout << "  quantize:  min=" << bqs.min_ms << " ms  median=" << bqs.median_ms << " ms\n";
    std::cout << "  compute:   min=" << bcs.min_ms << " ms  median=" << bcs.median_ms << " ms\n";
    std::cout << "  total:     min=" << bts.min_ms << " ms  median=" << bts.median_ms << " ms\n\n";

    print_row("--- packed sparse OC_TILE=1 (baseline) ---",
              sqs, scs, sts, bcs.min_ms, bts.min_ms);

    std::cout << "\n";

    print_row("--- dense GEMM M_TILE=2 (experiment) ---",
              gqs, gcs, gts, bcs.min_ms, bts.min_ms);

    std::cout << "\n--- correctness (vs bucket output) ---\n";
    std::cout << "bucket vs golden:   max_abs=" << err_bucket.max_abs
              << "  rel_mean=" << err_bucket.rel_mean << "\n";
    std::cout << "sparse vs bucket:   max_abs=" << err_sparse.max_abs
              << "  rel_mean=" << err_sparse.rel_mean << "\n";
    std::cout << "gemm   vs bucket:   max_abs=" << err_gemm.max_abs
              << "  rel_mean=" << err_gemm.rel_mean << "\n";

    std::cout << "\n--- verdict ---\n";
    const double speedup = bts.min_ms / gts.min_ms;
    if (gts.min_ms <= 1.6) {
        std::cout << "PASS: gemm total " << gts.min_ms << " ms (<= 1.6 ms threshold). Continue to M_TILE=4.\n";
    } else if (gts.min_ms <= 1.9) {
        std::cout << "PROMISING: gemm total " << gts.min_ms
                  << " ms (1.6-1.9 ms). Speedup " << speedup
                  << "x vs bucket. Below integration threshold; document as experiment.\n";
    } else {
        std::cout << "STOP: gemm total " << gts.min_ms
                  << " ms (>= 1.9 ms). Speedup " << speedup
                  << "x vs bucket. Dense GEMM M_TILE=2 does not beat packed sparse sufficiently.\n";
    }

    return 0;
}
