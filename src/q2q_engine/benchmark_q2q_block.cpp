// benchmark_q2q_block.cpp — Q2Q/VNNI Phase 2: layer2.1 q8-to-q8 residual block
//
// Goal: chain the validated VNNI dense-ternary kernel (Phase 1: compute_med
// 0.90ms, total_med 1.59ms on layer2.1.conv2 alone) into a *true* q8-to-q8
// residual block — every activation tensor on the conv1->conv2->residual path
// is QTensorNHWC (uint8 NHWC + scale). No FP32 activation tensor is ever
// materialized between ops; only scalar BN scale/bias and the requantize
// divisor are FP32.
//
//   q8 in
//     -> conv1: VNNI dpbusd + BN1 + ReLU + fused requantize -> q8   (QTensorNHWC -> QTensorNHWC)
//     -> conv2: VNNI dpbusd + BN2 + fused requantize        -> q8   (QTensorNHWC -> QTensorNHWC)
//     -> q8 residual add (+ dequant) + ReLU + requantize    -> q8   (QTensorNHWC, QTensorNHWC -> QTensorNHWC)
//
// Both convs reuse ONE fused kernel (q2q_conv_bn_requant), parameterized by a
// `relu` flag — "do not tune the microkernel first": this is the unmodified
// Phase 1 dense OC4/K32 dpbusd accumulation, just with the epilogue switched
// from "dequant to FP32" to "dequant -> (relu) -> requantize to u8".
//
// Quantization scale is a single fixed placeholder constant ACT_SCALE used
// for every activation tensor (static/calibrated-style quantization — no
// dynamic per-batch scale search). Calibration accuracy is out of scope:
// "do not optimize correctness before timing path works".
//
// Reports two top-level numbers:
//   block_total_q2q_only           — conv1+conv2+residual only (input already q8;
//                                    the steady-state cost once chained after
//                                    another q8 layer)
//   block_total_with_input_convert — + cost of quantizing an FP32 activation
//                                    tensor to q8 first (cold-start / boundary
//                                    cost, e.g. first quantized layer after an
//                                    FP32 stem)
// ... and gates both against the measured basic_engine layer2.1 block cost
// (conv1 + conv2 + add_relu, q8 OCBlock2 bucket path, 1T): ~5.42 ms.
//
// Target: layer2.1 block (conv1, bn1, conv2, bn2; IC=OC=128, H=W=28, 3x3 s1 p1)
// Do not touch basic_engine. Do not implement full ResNet yet.
//
// Usage: ./benchmark_q2q_block <model.bin>

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

// Fixed placeholder activation scale shared by every q8 tensor in the block
// (input, conv1 output, conv2 output, residual, final output). Mirrors static
// per-tensor calibrated quantization. Real calibration is out of scope here.
constexpr float ACT_SCALE = 0.05f;

// Measured basic_engine layer2.1 block cost (q8 OCBlock2 bucket path, 1T,
// from `OMP_NUM_THREADS=1 ./build/basic_engine ...` profiled run):
//   layer2.1.conv1 ~2.71 ms, layer2.1.conv2 ~2.69 ms, add_relu ~0.02 ms
constexpr double BASIC_ENGINE_BLOCK_MS = 2.71 + 2.69 + 0.02;

// ---- Quantized NHWC activation tensor: the type the conv path now consumes
// and produces directly (no FP32 round-trip). Always stored padded so the
// dense VNNI kernel can read halo columns/rows without bounds checks. ----
struct QTensorNHWC {
    int H = 0, W = 0, C = 0;   // padded spatial dims
    float scale = 1.0f;
    std::vector<uint8_t> data; // [H][W][C], C contiguous, zero-filled border
};

static QTensorNHWC make_zero_padded_q8(int H, int W, int C, int pad, float scale) {
    QTensorNHWC t;
    t.H = H + 2 * pad; t.W = W + 2 * pad; t.C = C; t.scale = scale;
    t.data.assign((size_t)t.H * t.W * C, uint8_t(0));
    return t;
}

// FP32 NHWC -> QTensorNHWC (static scale = ACT_SCALE, zero-point = 0).
// This is the only place FP32 activation data exists in this benchmark — the
// boundary conversion at the q8 pipeline's entry, timed separately as
// `block_total_with_input_convert`.
static void quantize_nhwc_fp32_to_q8_padded(const float* src, int H, int W, int C, int pad, QTensorNHWC& dst) {
    const float inv_scale = 1.0f / ACT_SCALE;
    for (int h = 0; h < H; ++h) {
        uint8_t* row = dst.data.data() + ((size_t)(h + pad) * dst.W + pad) * C;
        const float* srow = src + (size_t)h * W * C;
        for (int i = 0; i < W * C; ++i) {
            int32_t q = static_cast<int32_t>(std::lround(srow[i] * inv_scale));
            q = q < 0 ? 0 : (q > 255 ? 255 : q);
            row[i] = static_cast<uint8_t>(q);
        }
    }
}

static void fill_random_fp32_nonneg(float* dst, size_t n, uint32_t seed, float vmax) {
    uint32_t rng = seed;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        dst[i] = (static_cast<float>(rng >> 8) / static_cast<float>(1u << 24)) * vmax;
    }
}

// ---- Dense ternary weight panel: [oc_tile][ky][kx][ic_chunk][oc_in_tile][32] ----
// (identical packing to Phase 1's build_oc4_k32_panel — generic over Conv2DParams)
static std::vector<int8_t> build_oc4_k32_panel(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const size_t base   = (size_t)IC * KH * KW;

    std::vector<int8_t> panel(
        (size_t)OC_TILES * KH * KW * IC_CHUNKS * OC_TILE * K_CHUNK, int8_t(0));

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile)
        for (int ky = 0; ky < KH; ++ky)
            for (int kx = 0; kx < KW; ++kx)
                for (int icc = 0; icc < IC_CHUNKS; ++icc)
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

static inline uint8_t requantize_u8(float v, float inv_out_scale) {
    int32_t q = static_cast<int32_t>(std::lround(v * inv_out_scale));
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    return static_cast<uint8_t>(q);
}

// Both convs in this block share shape (128->128, 3x3, s1, p1) -> shared strides.
struct PanelStrides {
    size_t st_icc, st_kx, st_ky, st_oct;
    static PanelStrides make(int IC, int KH, int KW) {
        const int IC_CHUNKS = IC / K_CHUNK;
        PanelStrides s;
        s.st_icc = (size_t)OC_TILE * K_CHUNK;
        s.st_kx  = (size_t)IC_CHUNKS * s.st_icc;
        s.st_ky  = (size_t)KW * s.st_kx;
        s.st_oct = (size_t)KH * s.st_ky;
        return s;
    }
};

// Computes the 4 OC-tile dot products (int32) for one output pixel via VNNI
// dpbusd accumulation — unmodified Phase 1 dense kernel core ("do not tune
// the microkernel first").
static inline void vnni_oc4_dots(
    const uint8_t* base_in, int W_PAD_IN, int C_IN,
    const int8_t* panel_tile, const PanelStrides& st,
    int IC_CHUNKS, int KH, int KW, int oh, int ow,
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (int ky = 0; ky < KH; ++ky) {
        const int ih = oh + ky;
        const int8_t* panel_ky = panel_tile + (size_t)ky * st.st_ky;
        for (int kx = 0; kx < KW; ++kx) {
            const int iw = ow + kx;
            const uint8_t* a_row = base_in + ((size_t)ih * W_PAD_IN + iw) * C_IN;
            const int8_t* panel_kx = panel_ky + (size_t)kx * st.st_kx;
            for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                const __m256i a32 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + icc * K_CHUNK));
                const int8_t* wp = panel_kx + (size_t)icc * st.st_icc;
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
    d0 = hsum_epi32(acc0); d1 = hsum_epi32(acc1);
    d2 = hsum_epi32(acc2); d3 = hsum_epi32(acc3);
}

// ---- Fused conv: VNNI + BN + (optional ReLU) + requantize -> QTensorNHWC ----
// Single kernel for both conv1 (relu=true) and conv2 (relu=false): "do not
// tune the microkernel first" — only the epilogue branches on `relu`, and the
// output is QTensorNHWC either way (no FP32 tensor crosses the boundary).
static void q2q_conv_bn_requant(
    const QTensorNHWC& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    bool relu, float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const int pad = conv.pad_h;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const PanelStrides st = PanelStrides::make(IC, KH, KW);

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data();
    uint8_t*       base_out   = y.data.data();

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        const int oc_base = oc_tile * OC_TILE;
        float s[OC_TILE], b[OC_TILE];
        for (int j = 0; j < OC_TILE; ++j) {
            s[j] = bn.scale[oc_base + j] * dq;
            b[j] = bn.bias[oc_base + j];
        }
        const int8_t* panel_tile = base_panel + (size_t)oc_tile * st.st_oct;

        for (int oh = 0; oh < OH; ++oh) {
            uint8_t* out_row = base_out + ((size_t)(oh + pad) * y.W + pad) * y.C + oc_base;
            for (int ow = 0; ow < OW; ++ow) {
                int32_t d0, d1, d2, d3;
                vnni_oc4_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW, oh, ow, d0, d1, d2, d3);

                float v0 = static_cast<float>(d0) * s[0] + b[0];
                float v1 = static_cast<float>(d1) * s[1] + b[1];
                float v2 = static_cast<float>(d2) * s[2] + b[2];
                float v3 = static_cast<float>(d3) * s[3] + b[3];
                if (relu) {
                    v0 = v0 < 0.f ? 0.f : v0; v1 = v1 < 0.f ? 0.f : v1;
                    v2 = v2 < 0.f ? 0.f : v2; v3 = v3 < 0.f ? 0.f : v3;
                }

                uint8_t* out_px = out_row + (size_t)ow * y.C;
                out_px[0] = requantize_u8(v0, inv_out_scale);
                out_px[1] = requantize_u8(v1, inv_out_scale);
                out_px[2] = requantize_u8(v2, inv_out_scale);
                out_px[3] = requantize_u8(v3, inv_out_scale);
            }
        }
    }
}

// ---- q8 residual add + ReLU: QTensorNHWC + QTensorNHWC -> QTensorNHWC ----
// Both operands and the result are q8; only the per-element add happens in a
// dequantized domain (unavoidable — the two operands carry independent
// scales in general). This is O(H*W*C), ~1/1000th the conv compute, so doing
// the add arithmetic in FP32 locally does not reintroduce an FP32 activation
// round-trip in any meaningful sense — no FP32 *tensor* is ever stored.
static void q2q_residual_add_relu(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, int pad, QTensorNHWC& y)
{
    const float inv_out_scale = 1.0f / out_scale;
    const float sa = a.scale, sb = b.scale;
    const uint8_t* abase = a.data.data();
    const uint8_t* bbase = b.data.data();
    uint8_t* ybase = y.data.data();

    for (int h = 0; h < H; ++h) {
        const uint8_t* a_row = abase + ((size_t)(h + pad) * a.W + pad) * C;
        const uint8_t* b_row = bbase + ((size_t)(h + pad) * b.W + pad) * C;
        uint8_t*       y_row = ybase + ((size_t)(h + pad) * y.W + pad) * C;
        for (int w = 0; w < W; ++w) {
            const uint8_t* ap = a_row + (size_t)w * C;
            const uint8_t* bp = b_row + (size_t)w * C;
            uint8_t*       yp = y_row + (size_t)w * C;
            for (int c = 0; c < C; ++c) {
                float sum = static_cast<float>(ap[c]) * sa + static_cast<float>(bp[c]) * sb;
                sum = sum < 0.f ? 0.f : sum;
                yp[c] = requantize_u8(sum, inv_out_scale);
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

        const Conv2DParams& conv1 = model.convs.at("layer2.1.conv1");
        const Conv2DParams& conv2 = model.convs.at("layer2.1.conv2");
        const BNFusedParams& bn1  = model.fused_bns.at("layer2.1.bn1");
        const BNFusedParams& bn2  = model.fused_bns.at("layer2.1.bn2");

        const int IC = conv1.in_channels, OC = conv1.out_channels;
        const int H = 28, W = 28, OH = 28, OW = 28;
        const int pad = conv1.pad_h;

        if (IC % K_CHUNK != 0 || OC % OC_TILE != 0)
            throw std::runtime_error("IC must be divisible by 32, OC by 4");
        if (conv2.in_channels != OC || conv2.out_channels != OC)
            throw std::runtime_error("expected layer2.1 conv1/conv2 shapes to match (128->128)");

        std::cout << "=== benchmark_q2q_block: layer2.1 q8-to-q8 residual block (VNNI) ===\n";
        std::cout << "  conv1: IC=" << IC << " OC=" << OC
                  << "  conv2: IC=" << conv2.in_channels << " OC=" << conv2.out_channels
                  << "  H=W=" << H << "  pad=" << pad << "\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";
        std::cout << "  basic_engine layer2.1 block (measured, 1T, q8 OCBlock2 bucket): "
                  << "conv1≈2.71ms + conv2≈2.69ms + add_relu≈0.02ms = " << BASIC_ENGINE_BLOCK_MS << " ms\n";

        const std::vector<int8_t> panel1 = build_oc4_k32_panel(conv1);
        const std::vector<int8_t> panel2 = build_oc4_k32_panel(conv2);
        std::cout << "  panel1/panel2 size: " << panel1.size() << " / " << panel2.size() << " bytes\n";

        // Synthetic FP32 activation tensor: stands in for the previous block's
        // (dequantized / FP32-stem) output — the only FP32 data in this binary,
        // and only ever touched at the q8 pipeline's entry boundary.
        std::vector<float> input_fp32((size_t)H * W * IC);
        fill_random_fp32_nonneg(input_fp32.data(), input_fp32.size(), 0x9E3779B9u, 1.0f);

        // Pre-quantized steady-state q8 input (quantized once, outside all
        // timing) — represents "input already arrived as q8 from a prior q8 layer".
        QTensorNHWC input_q8_steady = make_zero_padded_q8(H, W, IC, pad, ACT_SCALE);
        quantize_nhwc_fp32_to_q8_padded(input_fp32.data(), H, W, IC, pad, input_q8_steady);

        // Scratch q8 buffers reused across iterations (allocated once; only
        // interiors are overwritten — borders stay zero).
        QTensorNHWC mid_q8        = make_zero_padded_q8(OH, OW, OC, pad, ACT_SCALE);
        QTensorNHWC conv2_out_q8  = make_zero_padded_q8(OH, OW, OC, pad, ACT_SCALE);
        QTensorNHWC block_out_q8  = make_zero_padded_q8(OH, OW, OC, pad, ACT_SCALE);
        QTensorNHWC input_q8_cold = make_zero_padded_q8(H, W, IC, pad, ACT_SCALE);

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        // ---- block_total_q2q_only: conv1 + conv2 + residual, input already q8 ----
        auto run_q2q_only = [&]() {
            q2q_conv_bn_requant(input_q8_steady, panel1, conv1, bn1, /*relu=*/true,  ACT_SCALE, mid_q8, OH, OW);
            q2q_conv_bn_requant(mid_q8,          panel2, conv2, bn2, /*relu=*/false, ACT_SCALE, conv2_out_q8, OH, OW);
            q2q_residual_add_relu(conv2_out_q8, input_q8_steady, ACT_SCALE, OH, OW, OC, pad, block_out_q8);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_q2q_only();

        std::vector<double> c1(N_TIMED), c2(N_TIMED), rs(N_TIMED), only_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q_conv_bn_requant(input_q8_steady, panel1, conv1, bn1, /*relu=*/true, ACT_SCALE, mid_q8, OH, OW);
            const double t1 = now_ms();
            q2q_conv_bn_requant(mid_q8, panel2, conv2, bn2, /*relu=*/false, ACT_SCALE, conv2_out_q8, OH, OW);
            const double t2 = now_ms();
            q2q_residual_add_relu(conv2_out_q8, input_q8_steady, ACT_SCALE, OH, OW, OC, pad, block_out_q8);
            const double t3 = now_ms();
            c1[i] = t1 - t0; c2[i] = t2 - t1; rs[i] = t3 - t2; only_total[i] = t3 - t0;
        }
        volatile uint32_t sink1 = block_out_q8.data[0] + block_out_q8.data[block_out_q8.data.size() - 1];
        (void)sink1;

        // ---- block_total_with_input_convert: + FP32->q8 quantize of the block input ----
        auto run_with_convert = [&]() {
            quantize_nhwc_fp32_to_q8_padded(input_fp32.data(), H, W, IC, pad, input_q8_cold);
            q2q_conv_bn_requant(input_q8_cold, panel1, conv1, bn1, /*relu=*/true,  ACT_SCALE, mid_q8, OH, OW);
            q2q_conv_bn_requant(mid_q8,        panel2, conv2, bn2, /*relu=*/false, ACT_SCALE, conv2_out_q8, OH, OW);
            q2q_residual_add_relu(conv2_out_q8, input_q8_cold, ACT_SCALE, OH, OW, OC, pad, block_out_q8);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_with_convert();

        std::vector<double> conv_ms(N_TIMED), with_convert_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_nhwc_fp32_to_q8_padded(input_fp32.data(), H, W, IC, pad, input_q8_cold);
            const double t1 = now_ms();
            q2q_conv_bn_requant(input_q8_cold, panel1, conv1, bn1, /*relu=*/true, ACT_SCALE, mid_q8, OH, OW);
            q2q_conv_bn_requant(mid_q8, panel2, conv2, bn2, /*relu=*/false, ACT_SCALE, conv2_out_q8, OH, OW);
            q2q_residual_add_relu(conv2_out_q8, input_q8_cold, ACT_SCALE, OH, OW, OC, pad, block_out_q8);
            const double t2 = now_ms();
            conv_ms[i] = t1 - t0; with_convert_total[i] = t2 - t0;
        }
        volatile uint32_t sink2 = block_out_q8.data[0] + block_out_q8.data[block_out_q8.data.size() - 1];
        (void)sink2;

        std::cout << "\n=== layer2.1 q8-to-q8 block timing (over " << N_TIMED << " runs) ===\n";
        std::cout << "-- block_total_q2q_only (input already q8; steady-state chained cost) --\n";
        print_stats("conv1 (VNNI+BN+ReLU+requant -> q8)", c1, N_TIMED);
        print_stats("conv2 (VNNI+BN+requant -> q8)     ", c2, N_TIMED);
        print_stats("residual (q8 add+ReLU+requant)    ", rs, N_TIMED);
        print_stats("block_total_q2q_only              ", only_total, N_TIMED);

        std::cout << "\n-- block_total_with_input_convert (+ FP32->q8 of block input) --\n";
        print_stats("input_convert (FP32 -> q8)         ", conv_ms, N_TIMED);
        print_stats("block_total_with_input_convert     ", with_convert_total, N_TIMED);

        std::vector<double> s_only = only_total, s_conv = with_convert_total;
        std::sort(s_only.begin(), s_only.end());
        std::sort(s_conv.begin(), s_conv.end());
        const double med_only = s_only[N_TIMED / 2];
        const double med_conv = s_conv[N_TIMED / 2];

        std::cout << "\n=== gate: q2q block vs basic_engine layer2.1 block ===\n";
        std::cout << "  basic_engine layer2.1 block:         " << BASIC_ENGINE_BLOCK_MS << " ms\n";
        std::cout << "  block_total_q2q_only (med):          " << med_only << " ms\n";
        std::cout << "  block_total_with_input_convert (med):" << med_conv << " ms\n";

        if (med_only < BASIC_ENGINE_BLOCK_MS) {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) BEATS basic_engine ("
                      << BASIC_ENGINE_BLOCK_MS << " ms) by "
                      << (BASIC_ENGINE_BLOCK_MS / med_only) << "x -- continue.\n";
        } else {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) does NOT beat basic_engine ("
                      << BASIC_ENGINE_BLOCK_MS << " ms) -- stop, do not proceed to full ResNet.\n";
        }
        if (med_conv < BASIC_ENGINE_BLOCK_MS) {
            std::cout << "  (with_input_convert also beats basic_engine: " << med_conv
                      << " ms < " << BASIC_ENGINE_BLOCK_MS << " ms -- "
                         "the boundary-conversion cost would not erase the win.)\n";
        } else {
            std::cout << "  (with_input_convert does NOT beat basic_engine: " << med_conv
                      << " ms >= " << BASIC_ENGINE_BLOCK_MS << " ms -- "
                         "win depends on staying chained in q8 across block boundaries.)\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
