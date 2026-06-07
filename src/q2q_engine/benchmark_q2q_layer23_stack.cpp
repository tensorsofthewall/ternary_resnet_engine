// benchmark_q2q_layer23_stack.cpp — Q2Q/VNNI: combined layer2+layer3
//
// Chains the two independently-validated stacks end-to-end in QTensorNHWC,
// with NO FP32 materialization at the layer2/layer3 boundary — layer2's q8
// block output (l21_out) is fed directly as layer3.0's input AND, via the
// SAME tensor, as layer3.0's downsample/residual operand:
//
//   FP32 (56x56x64) -> q8 NHWC  [one-time boundary conversion]
//   layer2.0 -> layer2.1                          (28x28x128 q8 out)
//   layer3.0 -> layer3.1                          (14x14x256 q8 out)
//
// Both stacks individually beat basic_engine (1.48x and ~2.05x respectively);
// this phase tests whether that holds when chained — i.e. whether anything
// about crossing the stack boundary in q8 (rather than materializing FP32
// and re-quantizing) erodes the win. Every kernel here
// (q2q_conv_bn_requant / vnni_oc4_dots / q2q_residual_add_relu) is reused
// byte-for-byte from the layer2 and layer3 stack benchmarks — no new
// microkernel code, only orchestration.
//
// Do not touch basic_engine. Do not implement full ResNet until this combined
// stack beats basic_engine's combined layer2+layer3 cost.
//
// Usage: ./benchmark_q2q_layer23_stack <model.bin>

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

// Fixed tensor border for every QTensorNHWC in this benchmark — sufficient
// for any 3x3/pad=1 conv; 1x1/pad=0 convs read with read_offset=+1 from the
// same buffer (see q2q_conv_bn_requant). Carried over unchanged from the
// layer2/layer3 stack benchmarks.
constexpr int PAD = 1;

// Static placeholder activation scale shared by every q8 tensor end-to-end
// (calibration accuracy out of scope — only timing matters).
constexpr float ACT_SCALE = 0.05f;

// Measured basic_engine costs (q8 OCBlock2 bucket path, 1T) — from the
// layer2 stack and layer3 stack benchmarks' own gate baselines:
constexpr double BASIC_ENGINE_L2_MS = 10.28;
constexpr double BASIC_ENGINE_L3_MS = 12.50;
constexpr double BASIC_ENGINE_COMBINED_MS = BASIC_ENGINE_L2_MS + BASIC_ENGINE_L3_MS;

// ---- Quantized NHWC activation tensor — the type the conv path consumes and
// produces directly end-to-end. Always allocated with border = PAD. ----
struct QTensorNHWC {
    int H = 0, W = 0, C = 0;   // padded (allocated) spatial dims
    float scale = 1.0f;
    std::vector<uint8_t> data; // [H][W][C], C contiguous, zero-filled border
};

static QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, float scale) {
    QTensorNHWC t;
    t.H = H_logical + 2 * PAD; t.W = W_logical + 2 * PAD; t.C = C; t.scale = scale;
    t.data.assign((size_t)t.H * t.W * C, uint8_t(0));
    return t;
}

static void quantize_nhwc_fp32_to_q8(const float* src, int H, int W, int C, QTensorNHWC& dst) {
    const float inv_scale = 1.0f / ACT_SCALE;
    for (int h = 0; h < H; ++h) {
        uint8_t* row = dst.data.data() + ((size_t)(h + PAD) * dst.W + PAD) * C;
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
// Generic over Conv2DParams — unchanged from layer2/layer3 stack benchmarks.
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

// VNNI dpbusd accumulation core — byte-identical to the layer2/layer3 stack
// benchmarks. `stride`/`read_offset` generalize indexing without touching
// the accumulation itself: ih = oh*stride + ky + read_offset.
static inline void vnni_oc4_dots(
    const uint8_t* base_in, int W_PAD_IN, int C_IN,
    const int8_t* panel_tile, const PanelStrides& st,
    int IC_CHUNKS, int KH, int KW, int oh, int ow, int stride, int read_offset,
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (int ky = 0; ky < KH; ++ky) {
        const int ih = oh * stride + ky + read_offset;
        const int8_t* panel_ky = panel_tile + (size_t)ky * st.st_ky;
        for (int kx = 0; kx < KW; ++kx) {
            const int iw = ow * stride + kx + read_offset;
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
// ONE kernel for all ten convs across both stacks — byte-identical to the
// layer2/layer3 stack benchmarks' kernel. `relu`/`out_scale` are call-site
// knobs; stride/pad reconciliation derives from `conv` itself.
static void q2q_conv_bn_requant(
    const QTensorNHWC& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    bool relu, float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const int stride = conv.stride_h;
    const int read_offset = PAD - conv.pad_h;
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
            uint8_t* out_row = base_out + ((size_t)(oh + PAD) * y.W + PAD) * y.C + oc_base;
            for (int ow = 0; ow < OW; ++ow) {
                int32_t d0, d1, d2, d3;
                vnni_oc4_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW,
                              oh, ow, stride, read_offset, d0, d1, d2, d3);

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

// q8 residual add + ReLU — byte-identical to layer2/layer3 stack benchmarks.
static void q2q_residual_add_relu(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, QTensorNHWC& y)
{
    const float inv_out_scale = 1.0f / out_scale;
    const float sa = a.scale, sb = b.scale;
    const uint8_t* abase = a.data.data();
    const uint8_t* bbase = b.data.data();
    uint8_t* ybase = y.data.data();

    for (int h = 0; h < H; ++h) {
        const uint8_t* a_row = abase + ((size_t)(h + PAD) * a.W + PAD) * C;
        const uint8_t* b_row = bbase + ((size_t)(h + PAD) * b.W + PAD) * C;
        uint8_t*       y_row = ybase + ((size_t)(h + PAD) * y.W + PAD) * C;
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

        // layer2.0 / layer2.1
        const Conv2DParams& c20_conv1 = model.convs.at("layer2.0.conv1");
        const Conv2DParams& c20_conv2 = model.convs.at("layer2.0.conv2");
        const Conv2DParams& c20_ds    = model.convs.at("layer2.0.downsample.0");
        const BNFusedParams& bn20_1   = model.fused_bns.at("layer2.0.bn1");
        const BNFusedParams& bn20_2   = model.fused_bns.at("layer2.0.bn2");
        const BNFusedParams& bn20_ds  = model.fused_bns.at("layer2.0.downsample.1");
        const Conv2DParams& c21_conv1 = model.convs.at("layer2.1.conv1");
        const Conv2DParams& c21_conv2 = model.convs.at("layer2.1.conv2");
        const BNFusedParams& bn21_1   = model.fused_bns.at("layer2.1.bn1");
        const BNFusedParams& bn21_2   = model.fused_bns.at("layer2.1.bn2");
        // layer3.0 / layer3.1
        const Conv2DParams& c30_conv1 = model.convs.at("layer3.0.conv1");
        const Conv2DParams& c30_conv2 = model.convs.at("layer3.0.conv2");
        const Conv2DParams& c30_ds    = model.convs.at("layer3.0.downsample.0");
        const BNFusedParams& bn30_1   = model.fused_bns.at("layer3.0.bn1");
        const BNFusedParams& bn30_2   = model.fused_bns.at("layer3.0.bn2");
        const BNFusedParams& bn30_ds  = model.fused_bns.at("layer3.0.downsample.1");
        const Conv2DParams& c31_conv1 = model.convs.at("layer3.1.conv1");
        const Conv2DParams& c31_conv2 = model.convs.at("layer3.1.conv2");
        const BNFusedParams& bn31_1   = model.fused_bns.at("layer3.1.bn1");
        const BNFusedParams& bn31_2   = model.fused_bns.at("layer3.1.bn2");

        const int IC0 = c20_conv1.in_channels;   // 64   (input to layer2: 56x56x64)
        const int OC2 = c20_conv1.out_channels;  // 128  (layer2 output: 28x28x128)
        const int OC3 = c30_conv1.out_channels;  // 256  (layer3 output: 14x14x256)
        const int H_IN = 56, W_IN = 56;
        const int H2 = 28, W2 = 28;
        const int H3 = 14, W3 = 14;

        if (IC0 % K_CHUNK != 0 || OC2 % OC_TILE != 0 || OC3 % OC_TILE != 0)
            throw std::runtime_error("channel counts must be divisible by 32 (IC) / 4 (OC)");

        std::cout << "=== benchmark_q2q_layer23_stack: combined layer2+layer3 q8-to-q8 (VNNI) ===\n";
        std::cout << "  input: FP32 NCHW " << IC0 << "x" << H_IN << "x" << W_IN
                  << " -> q8 NHWC (one-time boundary conversion)\n";
        std::cout << "  layer2: 56x56x" << IC0 << " -> 28x28x" << OC2 << "\n";
        std::cout << "  layer3: 28x28x" << OC2 << " -> 14x14x" << OC3 << "\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";
        std::cout << "  basic_engine baselines (measured, 1T, q8 OCBlock2 bucket):\n"
                  << "    layer2   = " << BASIC_ENGINE_L2_MS << " ms\n"
                  << "    layer3   = " << BASIC_ENGINE_L3_MS << " ms\n"
                  << "    combined = " << BASIC_ENGINE_COMBINED_MS << " ms\n";

        // Build all ten panels up front (reusing the generic builder).
        const std::vector<int8_t> p20_c1 = build_oc4_k32_panel(c20_conv1);
        const std::vector<int8_t> p20_c2 = build_oc4_k32_panel(c20_conv2);
        const std::vector<int8_t> p20_ds = build_oc4_k32_panel(c20_ds);
        const std::vector<int8_t> p21_c1 = build_oc4_k32_panel(c21_conv1);
        const std::vector<int8_t> p21_c2 = build_oc4_k32_panel(c21_conv2);
        const std::vector<int8_t> p30_c1 = build_oc4_k32_panel(c30_conv1);
        const std::vector<int8_t> p30_c2 = build_oc4_k32_panel(c30_conv2);
        const std::vector<int8_t> p30_ds = build_oc4_k32_panel(c30_ds);
        const std::vector<int8_t> p31_c1 = build_oc4_k32_panel(c31_conv1);
        const std::vector<int8_t> p31_c2 = build_oc4_k32_panel(c31_conv2);

        // Synthetic FP32 activation: stand-in for the layer2-input dump
        // (64x56x56) — the ONLY FP32 tensor anywhere in this benchmark,
        // touched only at the combined stack's entry boundary.
        std::vector<float> input_fp32((size_t)H_IN * W_IN * IC0);
        fill_random_fp32_nonneg(input_fp32.data(), input_fp32.size(), 0x9E3779B9u, 1.0f);

        QTensorNHWC input_q8_steady = make_q8_tensor(H_IN, W_IN, IC0, ACT_SCALE);
        quantize_nhwc_fp32_to_q8(input_fp32.data(), H_IN, W_IN, IC0, input_q8_steady);
        QTensorNHWC input_q8_cold = make_q8_tensor(H_IN, W_IN, IC0, ACT_SCALE);

        // layer2.0 / layer2.1 scratch
        QTensorNHWC l20_mid       = make_q8_tensor(H2, W2, OC2, ACT_SCALE);
        QTensorNHWC l20_conv2_out = make_q8_tensor(H2, W2, OC2, ACT_SCALE);
        QTensorNHWC l20_identity  = make_q8_tensor(H2, W2, OC2, ACT_SCALE);
        QTensorNHWC l20_out       = make_q8_tensor(H2, W2, OC2, ACT_SCALE);
        QTensorNHWC l21_mid       = make_q8_tensor(H2, W2, OC2, ACT_SCALE);
        QTensorNHWC l21_conv2_out = make_q8_tensor(H2, W2, OC2, ACT_SCALE);
        QTensorNHWC l21_out       = make_q8_tensor(H2, W2, OC2, ACT_SCALE);  // -> layer3 input
        // layer3.0 / layer3.1 scratch
        QTensorNHWC l30_mid       = make_q8_tensor(H3, W3, OC3, ACT_SCALE);
        QTensorNHWC l30_conv2_out = make_q8_tensor(H3, W3, OC3, ACT_SCALE);
        QTensorNHWC l30_identity  = make_q8_tensor(H3, W3, OC3, ACT_SCALE);
        QTensorNHWC l30_out       = make_q8_tensor(H3, W3, OC3, ACT_SCALE);
        QTensorNHWC l31_mid       = make_q8_tensor(H3, W3, OC3, ACT_SCALE);
        QTensorNHWC l31_conv2_out = make_q8_tensor(H3, W3, OC3, ACT_SCALE);
        QTensorNHWC l31_out       = make_q8_tensor(H3, W3, OC3, ACT_SCALE);  // combined output

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        auto run_layer2 = [&](const QTensorNHWC& in) {
            q2q_conv_bn_requant(in,      p20_c1, c20_conv1, bn20_1, true,  ACT_SCALE, l20_mid, H2, W2);
            q2q_conv_bn_requant(l20_mid, p20_c2, c20_conv2, bn20_2, false, ACT_SCALE, l20_conv2_out, H2, W2);
            q2q_conv_bn_requant(in,      p20_ds, c20_ds,    bn20_ds,false, ACT_SCALE, l20_identity, H2, W2);
            q2q_residual_add_relu(l20_conv2_out, l20_identity, ACT_SCALE, H2, W2, OC2, l20_out);
            q2q_conv_bn_requant(l20_out, p21_c1, c21_conv1, bn21_1, true,  ACT_SCALE, l21_mid, H2, W2);
            q2q_conv_bn_requant(l21_mid, p21_c2, c21_conv2, bn21_2, false, ACT_SCALE, l21_conv2_out, H2, W2);
            q2q_residual_add_relu(l21_conv2_out, l20_out, ACT_SCALE, H2, W2, OC2, l21_out);
        };
        auto run_layer3 = [&](const QTensorNHWC& in) {
            q2q_conv_bn_requant(in,      p30_c1, c30_conv1, bn30_1, true,  ACT_SCALE, l30_mid, H3, W3);
            q2q_conv_bn_requant(l30_mid, p30_c2, c30_conv2, bn30_2, false, ACT_SCALE, l30_conv2_out, H3, W3);
            q2q_conv_bn_requant(in,      p30_ds, c30_ds,    bn30_ds,false, ACT_SCALE, l30_identity, H3, W3);
            q2q_residual_add_relu(l30_conv2_out, l30_identity, ACT_SCALE, H3, W3, OC3, l30_out);
            q2q_conv_bn_requant(l30_out, p31_c1, c31_conv1, bn31_1, true,  ACT_SCALE, l31_mid, H3, W3);
            q2q_conv_bn_requant(l31_mid, p31_c2, c31_conv2, bn31_2, false, ACT_SCALE, l31_conv2_out, H3, W3);
            q2q_residual_add_relu(l31_conv2_out, l30_out, ACT_SCALE, H3, W3, OC3, l31_out);
        };

        // ---- combined_q2q_only: input already q8, no FP32 at the boundary ----
        auto run_q2q_only = [&]() {
            run_layer2(input_q8_steady);
            run_layer3(l21_out);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_q2q_only();

        std::vector<double> t_layer2(N_TIMED), t_layer3(N_TIMED), t_combined_only(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            run_layer2(input_q8_steady);
            const double t1 = now_ms();
            run_layer3(l21_out);
            const double t2 = now_ms();
            t_layer2[i] = t1 - t0;
            t_layer3[i] = t2 - t1;
            t_combined_only[i] = t2 - t0;
        }
        volatile uint32_t sink1 = l31_out.data[0] + l31_out.data[l31_out.data.size() - 1];
        (void)sink1;

        // ---- combined_with_input_convert: + FP32->q8 of the layer2 input ----
        auto run_with_convert = [&]() {
            quantize_nhwc_fp32_to_q8(input_fp32.data(), H_IN, W_IN, IC0, input_q8_cold);
            run_layer2(input_q8_cold);
            run_layer3(l21_out);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_with_convert();

        std::vector<double> t_convert(N_TIMED), t_combined_convert(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_nhwc_fp32_to_q8(input_fp32.data(), H_IN, W_IN, IC0, input_q8_cold);
            const double t1 = now_ms();
            run_layer2(input_q8_cold);
            run_layer3(l21_out);
            const double t2 = now_ms();
            t_convert[i] = t1 - t0;
            t_combined_convert[i] = t2 - t0;
        }
        volatile uint32_t sink2 = l31_out.data[0] + l31_out.data[l31_out.data.size() - 1];
        (void)sink2;

        std::cout << "\n=== combined layer2+layer3 q8-to-q8 timing (over " << N_TIMED << " runs) ===\n";
        print_stats("layer2_ms                          ", t_layer2, N_TIMED);
        print_stats("layer3_ms                          ", t_layer3, N_TIMED);
        print_stats("layer23_q2q_only_ms                ", t_combined_only, N_TIMED);
        std::cout << "\n-- with input-convert boundary --\n";
        print_stats("input_convert (FP32 -> q8)         ", t_convert, N_TIMED);
        print_stats("layer23_with_input_convert_ms      ", t_combined_convert, N_TIMED);

        std::vector<double> s_l2 = t_layer2, s_l3 = t_layer3,
                            s_only = t_combined_only, s_conv = t_combined_convert;
        std::sort(s_l2.begin(), s_l2.end());
        std::sort(s_l3.begin(), s_l3.end());
        std::sort(s_only.begin(), s_only.end());
        std::sort(s_conv.begin(), s_conv.end());
        const double med_l2   = s_l2[N_TIMED / 2];
        const double med_l3   = s_l3[N_TIMED / 2];
        const double med_only = s_only[N_TIMED / 2];
        const double med_conv = s_conv[N_TIMED / 2];

        std::cout << "\n=== gate: combined q2q layer2+layer3 vs basic_engine ===\n";
        std::cout << "  basic_engine combined (l2+l3):       " << BASIC_ENGINE_COMBINED_MS << " ms"
                  << "  (layer2=" << BASIC_ENGINE_L2_MS << ", layer3=" << BASIC_ENGINE_L3_MS << ")\n";
        std::cout << "  layer2_ms (med):                     " << med_l2 << " ms\n";
        std::cout << "  layer3_ms (med):                     " << med_l3 << " ms\n";
        std::cout << "  layer23_q2q_only_ms (med):           " << med_only << " ms\n";
        std::cout << "  layer23_with_input_convert_ms (med): " << med_conv << " ms\n";

        if (med_only <= 13.5) {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) <= 13.5 ms -- STRONG, "
                         "speedup of " << (BASIC_ENGINE_COMBINED_MS / med_only)
                      << "x over basic_engine (" << BASIC_ENGINE_COMBINED_MS << " ms). Continue.\n";
        } else if (med_only <= 15.0) {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) <= 15.0 ms -- gate passes "
                         "(speedup " << (BASIC_ENGINE_COMBINED_MS / med_only)
                      << "x over basic_engine " << BASIC_ENGINE_COMBINED_MS << " ms). Continue.\n";
        } else if (med_only < BASIC_ENGINE_COMBINED_MS) {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) beats basic_engine ("
                      << BASIC_ENGINE_COMBINED_MS << " ms) by " << (BASIC_ENGINE_COMBINED_MS / med_only)
                      << "x but exceeds the 15.0 ms gate threshold -- review before continuing.\n";
        } else {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) does NOT beat basic_engine ("
                      << BASIC_ENGINE_COMBINED_MS << " ms) -- stop and debug; the chained win did "
                         "not survive crossing the layer2/layer3 boundary.\n";
        }
        if (med_conv < BASIC_ENGINE_COMBINED_MS) {
            std::cout << "  (with_input_convert also wins: " << med_conv << " ms < "
                      << BASIC_ENGINE_COMBINED_MS << " ms.)\n";
        } else {
            std::cout << "  (with_input_convert does NOT win: " << med_conv << " ms >= "
                      << BASIC_ENGINE_COMBINED_MS << " ms.)\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
