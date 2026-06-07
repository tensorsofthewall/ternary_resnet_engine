// benchmark_q2q_conv1_maxpool.cpp — conv1 + maxpool in q8/VNNI
//
// Extends the Q2Q/VNNI path to the network's FIRST two ops -- conv1 (7x7,
// stride=2, IC=3, OC=64, pad=3) and the following 3x3/stride=2/pad=1 maxpool.
// These are the only remaining non-residual ops between the FP32 input
// boundary and the validated layer1-4 trunk.
//
// conv1 is structurally unlike every conv handled so far: IC=3 does not divide
// K_CHUNK=32, so the dense per-(ky,kx,ic-chunk) panel layout used by
// q2q_conv_bn_requant cannot apply directly. Instead this benchmark packs an
// im2col-style activation panel per output pixel -- gathering the full
// receptive field (IC*KH*KW = 147 bytes, zero-padded to 160 = 5*32) into one
// contiguous uint8 vector -- then runs the SAME dpbusd_epi32 OC_TILE=4 VNNI
// core over flat 32-wide K-chunks (no ky/kx looping needed; the gather already
// encodes the spatial offsets). This pack-then-stream structure is the "A-panel"
// pattern referenced in the next phase.
//
// maxpool runs directly on q8 data: since every activation in this pipeline
// shares one static ACT_SCALE, max(quantized) tracks max(original) under the
// monotonic affine map -- no dequant/requant needed, output stays q8 NHWC.
//
// Do not touch basic_engine. basic_engine baselines are measured live (via
// forward_resnet18_workspace_timed) for a fair, current comparison.
//
// Usage: ./benchmark_q2q_conv1_maxpool <model.bin>
//   (reads artifacts/golden_input_fp32.bin as the model input)

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"
#include "resnet18_engine.hpp"

#include <immintrin.h>
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

namespace {

constexpr int OC_TILE = 4;
constexpr int K_CHUNK = 32;
constexpr float ACT_SCALE = 0.05f;

// ---- Quantized NHWC activation tensor — same shape as every Q2Q benchmark,
// but the border is parameterized per-tensor (conv1 needs border=pad_h=3,
// downstream q8 tensors need border=1 like every other Q2Q stage). ----
struct QTensorNHWC {
    int H = 0, W = 0, C = 0;   // padded (allocated) spatial dims
    int border = 0;
    float scale = 1.0f;
    std::vector<uint8_t> data; // [H][W][C], C contiguous, zero-filled border
};

static QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, int border, float scale) {
    QTensorNHWC t;
    t.H = H_logical + 2 * border; t.W = W_logical + 2 * border; t.C = C;
    t.border = border; t.scale = scale;
    t.data.assign((size_t)t.H * t.W * C, uint8_t(0));
    return t;
}

// FP32 NCHW [C][H][W] -> q8 NHWC, transposing while quantizing into the
// tensor's interior (border stays zero-filled).
static void quantize_nchw_fp32_to_q8_nhwc(const float* src, int H, int W, int C, QTensorNHWC& dst) {
    const float inv_scale = 1.0f / ACT_SCALE;
    const int b = dst.border;
    for (int y = 0; y < H; ++y) {
        uint8_t* row = dst.data.data() + ((size_t)(y + b) * dst.W + b) * C;
        for (int x = 0; x < W; ++x) {
            uint8_t* px = row + (size_t)x * C;
            for (int c = 0; c < C; ++c) {
                const float v = src[((size_t)c * H + y) * W + x];
                int32_t q = static_cast<int32_t>(std::lround(v * inv_scale));
                q = q < 0 ? 0 : (q > 255 ? 255 : q);
                px[c] = static_cast<uint8_t>(q);
            }
        }
    }
}

static inline int32_t hsum_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

static inline uint8_t requantize_u8(float v, float inv_out_scale) {
    int32_t q = static_cast<int32_t>(std::lround(v * inv_out_scale));
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    return static_cast<uint8_t>(q);
}

// ---- conv1 im2col panel: K = IC*KH*KW = 147, padded to K_PAD = 160 = 5*32.
// Layout matches build_oc4_k32_panel's tiling (oc_tile, k_chunk, oc_in_tile,
// k_in_chunk) but over a SINGLE flat K axis -- the spatial (ky,kx) and channel
// (ic) axes are pre-flattened here, at pack time, instead of inside the dot
// product. Padding lanes (k_global >= 147) carry weight 0 (a legal ternary
// value), contributing nothing to the accumulation. ----
struct Conv1Panel {
    int K_PAD = 0, IC_CHUNKS = 0, OC_TILES = 0;
    std::vector<int8_t> data;
};

static Conv1Panel build_conv1_im2col_panel(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int K_FULL = IC * KH * KW;                       // 147
    const int IC_CHUNKS = (K_FULL + K_CHUNK - 1) / K_CHUNK; // 5
    const int K_PAD = IC_CHUNKS * K_CHUNK;                  // 160
    const int OC_TILES = OC / OC_TILE;
    const size_t base = (size_t)IC * KH * KW;

    Conv1Panel panel;
    panel.K_PAD = K_PAD; panel.IC_CHUNKS = IC_CHUNKS; panel.OC_TILES = OC_TILES;
    panel.data.assign((size_t)OC_TILES * IC_CHUNKS * OC_TILE * K_CHUNK, int8_t(0));

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile)
        for (int icc = 0; icc < IC_CHUNKS; ++icc)
            for (int j = 0; j < OC_TILE; ++j) {
                const int oc = oc_tile * OC_TILE + j;
                for (int k = 0; k < K_CHUNK; ++k) {
                    const int kg = icc * K_CHUNK + k;       // flat (ic,ky,kx) index
                    int8_t w = 0;
                    if (kg < K_FULL) {
                        const int ic = kg / (KH * KW);
                        const int rem = kg % (KH * KW);
                        const int ky = rem / KW;
                        const int kx = rem % KW;
                        w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                    }
                    const size_t idx = (((size_t)oc_tile * IC_CHUNKS + icc) * OC_TILE + j) * K_CHUNK + k;
                    panel.data[idx] = w;
                }
            }
    return panel;
}

// ---- Q2Q conv1: pack im2col A-panel per output pixel, then stream flat
// 32-wide VNNI dpbusd chunks against the panel -- fused BN + ReLU + requant. ----
static void q2q_conv1_im2col_bn_relu_requant(
    const QTensorNHWC& x, const Conv1Panel& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, KH = conv.kh, KW = conv.kw;
    const int stride = conv.stride_h;
    const int K_FULL = IC * KH * KW;
    const int K_PAD = panel.K_PAD;
    const int IC_CHUNKS = panel.IC_CHUNKS;
    const int OC_TILES = panel.OC_TILES;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const int b = x.border;

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data.data();
    uint8_t*       base_out   = y.data.data();

    std::vector<uint8_t> a_panel(K_PAD, uint8_t(0));

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * stride - conv.pad_h + b; // border absorbs -pad: ih0 = oh*stride (since b == pad_h)
        uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C;
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * stride - conv.pad_w + b;

            // ---- pack: gather the 147-element receptive field into a flat
            // im2col vector, in the SAME (ic, ky, kx) order the panel uses ----
            for (int ic = 0; ic < IC; ++ic) {
                for (int ky = 0; ky < KH; ++ky) {
                    const uint8_t* row = base_in + ((size_t)(ih0 + ky) * x.W + iw0) * x.C + ic;
                    uint8_t* dst = a_panel.data() + (size_t)(ic * KH + ky) * KW;
                    for (int kx = 0; kx < KW; ++kx) dst[kx] = row[(size_t)kx * x.C];
                }
            }
            for (int kg = K_FULL; kg < K_PAD; ++kg) a_panel[kg] = 0;

            // ---- stream: flat 32-wide VNNI chunks, OC_TILE=4 at a time ----
            uint8_t* out_px = out_row + (size_t)ow * y.C;
            for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
                const int oc_base = oc_tile * OC_TILE;
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();
                const int8_t* panel_tile = base_panel + (size_t)oc_tile * IC_CHUNKS * OC_TILE * K_CHUNK;
                for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                    const __m256i a32 = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(a_panel.data() + (size_t)icc * K_CHUNK));
                    const int8_t* wp = panel_tile + (size_t)icc * OC_TILE * K_CHUNK;
                    const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 0*K_CHUNK));
                    const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 1*K_CHUNK));
                    const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 2*K_CHUNK));
                    const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 3*K_CHUNK));
                    acc0 = _mm256_dpbusd_epi32(acc0, a32, w0);
                    acc1 = _mm256_dpbusd_epi32(acc1, a32, w1);
                    acc2 = _mm256_dpbusd_epi32(acc2, a32, w2);
                    acc3 = _mm256_dpbusd_epi32(acc3, a32, w3);
                }
                const int32_t d0 = hsum_epi32(acc0), d1 = hsum_epi32(acc1);
                const int32_t d2 = hsum_epi32(acc2), d3 = hsum_epi32(acc3);

                float v0 = static_cast<float>(d0) * (bn.scale[oc_base + 0] * dq) + bn.bias[oc_base + 0];
                float v1 = static_cast<float>(d1) * (bn.scale[oc_base + 1] * dq) + bn.bias[oc_base + 1];
                float v2 = static_cast<float>(d2) * (bn.scale[oc_base + 2] * dq) + bn.bias[oc_base + 2];
                float v3 = static_cast<float>(d3) * (bn.scale[oc_base + 3] * dq) + bn.bias[oc_base + 3];
                v0 = v0 < 0.f ? 0.f : v0; v1 = v1 < 0.f ? 0.f : v1;
                v2 = v2 < 0.f ? 0.f : v2; v3 = v3 < 0.f ? 0.f : v3;

                out_px[oc_base + 0] = requantize_u8(v0, inv_out_scale);
                out_px[oc_base + 1] = requantize_u8(v1, inv_out_scale);
                out_px[oc_base + 2] = requantize_u8(v2, inv_out_scale);
                out_px[oc_base + 3] = requantize_u8(v3, inv_out_scale);
            }
        }
    }
}

// ---- q8 maxpool: 3x3/stride=2/pad=1, operating directly on uint8 NHWC data.
// Every activation shares one static ACT_SCALE, so max(uint8) tracks
// max(fp32) under the monotonic affine map -- no dequant/requant round trip.
static void q8_maxpool_3x3_s2_p1(const QTensorNHWC& x, int OH, int OW, QTensorNHWC& y) {
    constexpr int kh = 3, kw = 3, stride = 2, pad = 1;
    const int C = x.C;
    const int read_offset = x.border - pad; // border absorbs -pad when border==pad
    const uint8_t* base_in  = x.data.data();
    uint8_t*       base_out = y.data.data();

    for (int oh = 0; oh < OH; ++oh) {
        uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * C;
        for (int ow = 0; ow < OW; ++ow) {
            uint8_t* out_px = out_row + (size_t)ow * C;
            const uint8_t* p00 = base_in + ((size_t)(oh * stride + 0 + read_offset) * x.W + (ow * stride + 0 + read_offset)) * C;
            const uint8_t* p01 = p00 + (size_t)1 * C;
            const uint8_t* p02 = p00 + (size_t)2 * C;
            const uint8_t* p10 = p00 + (size_t)x.W * C;
            const uint8_t* p11 = p10 + (size_t)1 * C;
            const uint8_t* p12 = p10 + (size_t)2 * C;
            const uint8_t* p20 = p10 + (size_t)x.W * C;
            const uint8_t* p21 = p20 + (size_t)1 * C;
            const uint8_t* p22 = p20 + (size_t)2 * C;
            for (int c = 0; c < C; ++c) {
                uint8_t m = p00[c];
                m = std::max(m, p01[c]); m = std::max(m, p02[c]);
                m = std::max(m, p10[c]); m = std::max(m, p11[c]); m = std::max(m, p12[c]);
                m = std::max(m, p20[c]); m = std::max(m, p21[c]); m = std::max(m, p22[c]);
                out_px[c] = m;
            }
        }
    }
    (void)kh; (void)kw;
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

static std::vector<float> read_f32_file(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open file: " + path);
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
    if (!f) throw std::runtime_error("Failed to read expected float count from: " + path);
    return data;
}

static bool is_basic_conv1_maxpool_entry(const std::string& name) {
    return name == "conv1+bn1+relu" || name == "maxpool";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);
        const Conv2DParams& conv1   = model.convs.at("conv1");
        const BNFusedParams& bn1    = model.fused_bns.at("bn1");

        if (conv1.in_channels != 3 || conv1.out_channels % OC_TILE != 0)
            throw std::runtime_error("conv1 expected IC=3, OC % 4 == 0");

        const int IC = conv1.in_channels, OC = conv1.out_channels;
        const int H_IN = 224, W_IN = 224;
        const int OH1 = (H_IN + 2 * conv1.pad_h - conv1.kh) / conv1.stride_h + 1; // 112
        const int OW1 = (W_IN + 2 * conv1.pad_w - conv1.kw) / conv1.stride_w + 1; // 112
        const int OH2 = (OH1 + 2 * 1 - 3) / 2 + 1; // maxpool: 56
        const int OW2 = (OW1 + 2 * 1 - 3) / 2 + 1; // 56

        std::cout << "=== benchmark_q2q_conv1_maxpool: conv1 (im2col VNNI) + q8 maxpool ===\n";
        std::cout << "  conv1  : " << IC << "x" << H_IN << "x" << W_IN << " -> " << OC << "x" << OH1 << "x" << OW1
                  << "  (k=" << conv1.kh << "x" << conv1.kw << " s=" << conv1.stride_h << " p=" << conv1.pad_h << ")\n";
        std::cout << "  maxpool: " << OC << "x" << OH1 << "x" << OW1 << " -> " << OC << "x" << OH2 << "x" << OW2
                  << "  (k=3x3 s=2 p=1, q8-direct)\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";

        // ---- measure basic_engine conv1+maxpool cost live ----
        Tensor golden_input(3, 224, 224);
        golden_input.data = read_f32_file("artifacts/golden_input_fp32.bin", 3 * 224 * 224);

        EngineWorkspace ws;
        constexpr int N_WARMUP_BASIC = 10;
        constexpr int N_TIMED_BASIC  = 30;
        for (int i = 0; i < N_WARMUP_BASIC; ++i) {
            std::vector<TimingEntry> t;
            forward_resnet18_workspace_timed(model, golden_input, ws, t);
        }
        std::vector<double> t_basic(N_TIMED_BASIC);
        for (int i = 0; i < N_TIMED_BASIC; ++i) {
            std::vector<TimingEntry> t;
            forward_resnet18_workspace_timed(model, golden_input, ws, t);
            double sum = 0.0;
            for (const auto& e : t) if (is_basic_conv1_maxpool_entry(e.name.data())) sum += e.ms;
            t_basic[i] = sum;
        }
        std::vector<double> s_basic = t_basic;
        std::sort(s_basic.begin(), s_basic.end());
        const double basic_conv1_maxpool_ms = s_basic[N_TIMED_BASIC / 2];

        std::cout << "\n=== basic_engine conv1+maxpool cost (measured live, workspace-timed path, "
                  << N_TIMED_BASIC << " runs) ===\n";
        print_stats("basic_conv1_maxpool_ms", t_basic, N_TIMED_BASIC);

        // ---- q8 pipeline: real input, quantized once ----
        constexpr int PAD_C1 = 3; // == conv1.pad_h: border absorbs the conv padding
        constexpr int PAD_Q8 = 1; // standard Q2Q border for downstream 3x3/pad=1 consumers

        QTensorNHWC input_q8  = make_q8_tensor(H_IN, W_IN, IC, PAD_C1, ACT_SCALE);
        QTensorNHWC conv1_out = make_q8_tensor(OH1, OW1, OC, PAD_Q8, ACT_SCALE);
        QTensorNHWC pool_out  = make_q8_tensor(OH2, OW2, OC, PAD_Q8, ACT_SCALE);

        quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), H_IN, W_IN, IC, input_q8);

        const Conv1Panel panel = build_conv1_im2col_panel(conv1);
        std::cout << "\n  conv1 im2col panel: K_full=" << (IC * conv1.kh * conv1.kw)
                  << "  K_pad=" << panel.K_PAD << "  IC_CHUNKS=" << panel.IC_CHUNKS
                  << "  OC_TILES=" << panel.OC_TILES << "\n";

        auto run_q2q = [&]() {
            q2q_conv1_im2col_bn_relu_requant(input_q8, panel, conv1, bn1, ACT_SCALE, conv1_out, OH1, OW1);
            q8_maxpool_3x3_s2_p1(conv1_out, OH2, OW2, pool_out);
        };

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;
        for (int i = 0; i < N_WARMUP; ++i) run_q2q();

        std::vector<double> t_conv1(N_TIMED), t_maxpool(N_TIMED), t_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q_conv1_im2col_bn_relu_requant(input_q8, panel, conv1, bn1, ACT_SCALE, conv1_out, OH1, OW1);
            const double t1 = now_ms();
            q8_maxpool_3x3_s2_p1(conv1_out, OH2, OW2, pool_out);
            const double t2 = now_ms();
            t_conv1[i]   = t1 - t0;
            t_maxpool[i] = t2 - t1;
            t_total[i]   = t2 - t0;
        }
        volatile uint32_t sink = pool_out.data[0] + pool_out.data[pool_out.data.size() - 1];
        (void)sink;

        std::cout << "\n=== Q2Q conv1+maxpool timing (q8-to-q8, over " << N_TIMED << " runs) ===\n";
        print_stats("conv1_q2q_ms   (im2col VNNI + BN + ReLU + requant)", t_conv1, N_TIMED);
        print_stats("maxpool_q8_ms  (direct uint8 max, no dequant)     ", t_maxpool, N_TIMED);
        print_stats("conv1_maxpool_total_ms                            ", t_total, N_TIMED);

        std::vector<double> s_total = t_total;
        std::sort(s_total.begin(), s_total.end());
        const double med_total = s_total[N_TIMED / 2];

        std::cout << "\n=== gate: Q2Q conv1+maxpool vs basic_engine ===\n";
        std::cout << "  basic_engine conv1+maxpool (measured live): " << basic_conv1_maxpool_ms << " ms\n";
        std::cout << "  Q2Q conv1+maxpool (med)                   : " << med_total << " ms\n";
        if (med_total < basic_conv1_maxpool_ms) {
            std::cout << "  RESULT: Q2Q conv1+maxpool (" << med_total << " ms) BEATS basic_engine ("
                      << basic_conv1_maxpool_ms << " ms) by " << (basic_conv1_maxpool_ms / med_total)
                      << "x -- worth carrying into the prefix chain.\n";
        } else {
            std::cout << "  RESULT: Q2Q conv1+maxpool (" << med_total << " ms) does NOT beat basic_engine ("
                      << basic_conv1_maxpool_ms << " ms) -- the im2col pack overhead and/or small-IC "
                         "VNNI underutilization (K_pad=160 vs K_full=147, ~8% padding waste; OC_TILES=16 "
                         "re-walk the same panel per pixel) outweighs the dense-VNNI advantage here.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
