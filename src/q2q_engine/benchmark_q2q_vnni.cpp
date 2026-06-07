// benchmark_q2q_vnni.cpp — Q2Q / AVX-VNNI dense ternary kernel (runtime-only)
//
// Hypothesis: dense int8 dot-product via AVX-VNNI _mm256_dpbusd_epi32 may beat
// sparse add/sub kernels because dpbusd has much higher arithmetic throughput,
// even though zero weights are not skipped.
//
//   - NHWC uint8 activations (zero-point = 0; valid because the input to
//     layer2.1.conv2 is post-ReLU / non-negative)
//   - dense int8 ternary weights {-1, 0, +1}, packed into OC4/K32 panels:
//     panel[oc_tile][ky][kx][ic_chunk][oc_in_tile][32] — for each 32-wide
//     K-chunk, four contiguous 32-byte weight vectors (one per OC in the
//     4-wide tile) share one loaded 32-byte activation vector.
//   - one _mm256_dpbusd_epi32 per (oc_in_tile, k_chunk): 8 lanes of partial
//     dot products over 4 K-values each; horizontal-reduced once per output
//     pixel per OC (not per K-chunk).
//   - fused BN scale/bias + optional ReLU at the end (FP32 store; this is a
//     standalone single-layer benchmark — no cross-layer round-trip).
//
// Target: layer2.1.conv2 (IC=128, OC=128, H=28, W=28, K=1152, stride=1, pad=1)
//
// Gate: total<=2.13ms continue, <=1.5ms strong, >2.67ms stop & profile.
//
// Usage: ./benchmark_q2q_vnni <model.bin>

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"

#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int OC_TILE = 4;
constexpr int K_CHUNK = 32;

// ---- NHWC uint8 padded activation, zero-point = 0 ----
struct U8NHWCPadded {
    int H = 0, W = 0, C = 0;   // padded spatial dims
    float scale = 1.0f;
    std::vector<uint8_t> data; // [H][W][C], contiguous in C
};

static void quantize_relu_nhwc_to_u8_padded(
    const float* src, int H, int W, int C, int pad, U8NHWCPadded& dst)
{
    const int H_PAD = H + 2 * pad;
    const int W_PAD = W + 2 * pad;

    float max_val = 0.0f;
    for (int i = 0; i < H * W * C; ++i)
        max_val = std::max(max_val, src[i]);
    dst.scale = (max_val == 0.0f) ? 1.0f : max_val / 255.0f;
    const float inv_scale = 1.0f / dst.scale;

    dst.H = H_PAD; dst.W = W_PAD; dst.C = C;
    dst.data.assign((size_t)H_PAD * W_PAD * C, uint8_t(0));

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const float* s = src + ((size_t)h * W + w) * C;
            uint8_t* d = dst.data.data()
                       + ((size_t)(h + pad) * W_PAD + (w + pad)) * C;
            for (int c = 0; c < C; ++c) {
                int32_t q = static_cast<int32_t>(std::roundf(s[c] * inv_scale));
                q = std::max(0, std::min(255, q));
                d[c] = static_cast<uint8_t>(q);
            }
        }
    }
}

// ---- Dense ternary weight panel: [oc_tile][ky][kx][ic_chunk][oc_in_tile][32] ----
static std::vector<int8_t> build_oc4_k32_panel(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const size_t base   = (size_t)IC * KH * KW;

    std::vector<int8_t> panel(
        (size_t)OC_TILES * KH * KW * IC_CHUNKS * OC_TILE * K_CHUNK, int8_t(0));

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                    for (int j = 0; j < OC_TILE; ++j) {
                        const int oc = oc_tile * OC_TILE + j;
                        for (int k = 0; k < K_CHUNK; ++k) {
                            const int ic = icc * K_CHUNK + k;
                            const int8_t w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                            const size_t idx =
                                ((((((size_t)oc_tile * KH + ky) * KW + kx) * IC_CHUNKS + icc)
                                  * OC_TILE + j) * K_CHUNK) + k;
                            panel[idx] = w;
                        }
                    }
                }
            }
        }
    }
    return panel;
}

static inline int32_t hsum_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// ---- Dense VNNI OC4/K32 kernel, stride-1, 3×3, pad=1 ----
static void kernel_dense_vnni_oc4(
    const U8NHWCPadded& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float* y_out, int OC, int OH, int OW, bool relu)
{
    const int IC = conv.in_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const int W_PAD = x.W;
    const int C     = x.C;
    const float dq  = x.scale;

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data();

    const size_t st_icc = (size_t)OC_TILE * K_CHUNK;
    const size_t st_kx  = (size_t)IC_CHUNKS * st_icc;
    const size_t st_ky  = (size_t)KW * st_kx;
    const size_t st_oct = (size_t)KH * st_ky;

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        const int oc_base = oc_tile * OC_TILE;
        float s[OC_TILE], b[OC_TILE];
        for (int j = 0; j < OC_TILE; ++j) {
            s[j] = bn.scale[oc_base + j] * dq;
            b[j] = bn.bias[oc_base + j];
        }
        const int8_t* panel_tile = base_panel + (size_t)oc_tile * st_oct;

        for (int oh = 0; oh < OH; ++oh) {
            float* row[OC_TILE];
            for (int j = 0; j < OC_TILE; ++j)
                row[j] = y_out + (size_t)(oc_base + j) * OH * OW + (size_t)oh * OW;

            for (int ow = 0; ow < OW; ++ow) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                for (int ky = 0; ky < KH; ++ky) {
                    const int ih = oh + ky;   // padded input — always in range
                    const int8_t* panel_ky = panel_tile + (size_t)ky * st_ky;

                    for (int kx = 0; kx < KW; ++kx) {
                        const int iw = ow + kx;
                        const uint8_t* a_row = base_in + ((size_t)ih * W_PAD + iw) * C;
                        const int8_t* panel_kx = panel_ky + (size_t)kx * st_kx;

                        for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                            const __m256i a32 = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(a_row + icc * K_CHUNK));
                            const int8_t* wp = panel_kx + (size_t)icc * st_icc;
                            const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 0*K_CHUNK));
                            const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 1*K_CHUNK));
                            const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 2*K_CHUNK));
                            const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 3*K_CHUNK));
                            acc0 = _mm256_dpbusd_epi32(acc0, a32, w0);
                            acc1 = _mm256_dpbusd_epi32(acc1, a32, w1);
                            acc2 = _mm256_dpbusd_epi32(acc2, a32, w2);
                            acc3 = _mm256_dpbusd_epi32(acc3, a32, w3);
                        }
                    }
                }

                const int32_t d0 = hsum_epi32(acc0);
                const int32_t d1 = hsum_epi32(acc1);
                const int32_t d2 = hsum_epi32(acc2);
                const int32_t d3 = hsum_epi32(acc3);

                float v0 = static_cast<float>(d0) * s[0] + b[0];
                float v1 = static_cast<float>(d1) * s[1] + b[1];
                float v2 = static_cast<float>(d2) * s[2] + b[2];
                float v3 = static_cast<float>(d3) * s[3] + b[3];
                if (relu) {
                    v0 = v0 < 0.f ? 0.f : v0; v1 = v1 < 0.f ? 0.f : v1;
                    v2 = v2 < 0.f ? 0.f : v2; v3 = v3 < 0.f ? 0.f : v3;
                }
                row[0][ow] = v0; row[1][ow] = v1; row[2][ow] = v2; row[3][ow] = v3;
            }
        }
    }
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

static void print_stats(const char* label, std::vector<double> v, int N) {
    std::sort(v.begin(), v.end());
    const double mn  = v.front();
    const double med = v[N / 2];
    const double p90 = v[N * 9 / 10];
    const double avg = std::accumulate(v.begin(), v.end(), 0.0) / N;
    std::cout << "  " << label
              << ":  min=" << mn << "  med=" << med
              << "  p90=" << p90 << "  avg=" << avg << " ms\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);

        const char* conv_name = "layer2.1.conv2";
        const char* bn_name   = "layer2.1.bn2";
        const bool  apply_relu = false;

        const Conv2DParams& conv = model.convs.at(conv_name);
        const BNFusedParams& bn  = model.fused_bns.at(bn_name);

        const int IC = conv.in_channels, OC = conv.out_channels;
        const int H = 28, W = 28, OH = 28, OW = 28;
        const int pad = conv.pad_h;
        const int K = IC * conv.kh * conv.kw;

        if (IC % K_CHUNK != 0 || OC % OC_TILE != 0)
            throw std::runtime_error("IC must be divisible by 32, OC by 4");

        std::cout << "=== benchmark_q2q_vnni: dense ternary AVX-VNNI dpbusd kernel ===\n";
        std::cout << "  Target: " << conv_name << "  IC=" << IC << "  OC=" << OC
                  << "  H=" << H << "  W=" << W << "  K=" << K
                  << "  OC_TILE=" << OC_TILE << "  K_CHUNK=" << K_CHUNK << "\n";
        std::cout << "  MACs: " << (size_t)OC * OH * OW * K
                  << "  dpbusd_calls(theory): " << (size_t)OC * OH * OW * K / 32 << "\n";

        std::cout << "\n--- baselines ---\n";
        std::cout << "  bucket (q8 ocblock2):    ~2.67 ms\n";
        std::cout << "  packed sparse:           ~2.13 ms\n";
        std::cout << "  blocked W32 (best wide): ~2.48 ms compute / ~3.23 ms total\n";

        // Synthetic post-ReLU NHWC input: non-negative random values.
        std::vector<float> src_nhwc((size_t)H * W * IC);
        {
            uint32_t rng = 0xC0FFEEu;
            for (float& v : src_nhwc) {
                rng = rng * 1664525u + 1013904223u;
                v = (rng >> 31) ? 0.0f : static_cast<float>((rng >> 8) & 0xFFFF) / 65535.0f;
            }
        }

        const double t_pack0 = now_ms();
        const std::vector<int8_t> panel = build_oc4_k32_panel(conv);
        const double t_pack1 = now_ms();
        std::cout << "\n  panel size: " << panel.size() << " bytes"
                  << "  (built once, " << (t_pack1 - t_pack0) << " ms — not in total_ms)\n";

        std::vector<float> y_out((size_t)OC * OH * OW, 0.0f);
        U8NHWCPadded xq;

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        for (int i = 0; i < N_WARMUP; ++i) {
            quantize_relu_nhwc_to_u8_padded(src_nhwc.data(), H, W, IC, pad, xq);
            kernel_dense_vnni_oc4(xq, panel, conv, bn, y_out.data(), OC, OH, OW, apply_relu);
        }

        std::vector<double> ct(N_TIMED), kt(N_TIMED), tt(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_relu_nhwc_to_u8_padded(src_nhwc.data(), H, W, IC, pad, xq);
            const double t1 = now_ms();
            kernel_dense_vnni_oc4(xq, panel, conv, bn, y_out.data(), OC, OH, OW, apply_relu);
            const double t2 = now_ms();
            ct[i] = t1 - t0; kt[i] = t2 - t1; tt[i] = t2 - t0;
        }

        volatile float sink = y_out[0] + y_out[(size_t)OC * OH * OW - 1];
        (void)sink;

        std::cout << "\n=== Q2Q VNNI dense OC4/K32 — layer2.1.conv2 ===\n";
        print_stats("convert_ms", ct, N_TIMED);
        print_stats("compute_ms", kt, N_TIMED);
        print_stats("total_ms  ", tt, N_TIMED);

        std::vector<double> sk = kt, st = tt;
        std::sort(sk.begin(), sk.end());
        std::sort(st.begin(), st.end());
        const double compute_med = sk[N_TIMED/2];
        const double total_med   = st[N_TIMED/2];

        std::cout << "\n  --- gate ---\n";
        if (total_med > 2.67)
            std::cout << "  STOP: total_med=" << total_med
                      << " ms > 2.67 ms — profile packing vs compute split.\n";
        else if (total_med <= 1.5)
            std::cout << "  STRONG: total_med=" << total_med << " ms <= 1.5 ms.\n";
        else if (total_med <= 2.13)
            std::cout << "  CONTINUE: total_med=" << total_med << " ms <= 2.13 ms.\n";
        else
            std::cout << "  CLOSE: total_med=" << total_med
                      << " ms (between 2.13 and 2.67 ms bucket).\n";
        std::cout << "  (compute_med=" << compute_med << " ms, convert_med="
                  << st[N_TIMED/2] - sk[N_TIMED/2] << " ms approx)\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
