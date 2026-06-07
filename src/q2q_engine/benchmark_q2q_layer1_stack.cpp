// benchmark_q2q_layer1_stack.cpp — Q2Q/VNNI: full layer1 stack (q8-to-q8)
//
// Extends the validated layer2/3/4 trunk (32.37ms -> ~17.3ms, ~1.85-1.88x) to
// layer1 — the largest remaining ResNet stage outside that trunk. Layer1 is
// structurally SIMPLER than layer2/3/4: both blocks are pure identity
// shortcuts (IC==OC==64, stride=1 throughout, no downsample projection), so
// each block is just conv1 -> conv2 -> residual_add(conv2_out, block_input):
//
//   layer1.0: conv1(3x3,s1) IC=64 OC=64 + BN + ReLU  -> q8
//             conv2(3x3,s1) IC=64 OC=64 + BN         -> q8
//             q8 residual add + ReLU (operand = block input, no projection) -> q8
//   layer1.1: conv1(3x3,s1) IC=64 OC=64 + BN + ReLU  -> q8
//             conv2(3x3,s1) IC=64 OC=64 + BN         -> q8
//             q8 residual add + ReLU (operand = block input, no projection) -> q8
//
// Spatially the largest stage (56x56 = 3136 output pixels per conv, 4x more
// than layer2's 28x28) but with the smallest K (IC=OC=64, vs 128/256/512
// downstream) — the inverse of layer4's regime. Empirically tests whether
// VNNI's throughput edge (favored by large K) still wins when K is at its
// network minimum and per-pixel work dominates.
//
// Kernel reuse ("do not tune the microkernel"): q2q_conv_bn_requant /
// vnni_oc4_dots / q2q_residual_add_relu are reused byte-for-byte from every
// earlier stack benchmark — only the orchestration (no downsample branch)
// and shapes differ.
//
// Do not implement the full model yet. Do not retune layer2-4. Do not touch
// basic_engine.
//
// Usage: ./benchmark_q2q_layer1_stack <model.bin>

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

// Every QTensorNHWC in this benchmark is allocated with this fixed border.
// Sufficient for any 3x3/pad=1 conv (the only kind in layer1 — no 1x1
// downsample exists here, so read_offset is always 0).
constexpr int PAD = 1;

// Fixed placeholder activation scale shared by every q8 tensor in the stack.
constexpr float ACT_SCALE = 0.05f;

// Measured basic_engine layer1 stack cost (q8 OCBlock2 bucket path, 1T):
//   layer1.0: conv1 ~2.25 + conv2 ~2.30 + add_relu ~0.05 = ~4.60 ms
//   layer1.1: conv1 ~2.31 + conv2 ~2.30 + add_relu ~0.05 = ~4.66 ms
constexpr double BASIC_ENGINE_L1_0_MS = 2.25 + 2.30 + 0.05;
constexpr double BASIC_ENGINE_L1_1_MS = 2.31 + 2.30 + 0.05;
constexpr double BASIC_ENGINE_STACK_MS = BASIC_ENGINE_L1_0_MS + BASIC_ENGINE_L1_1_MS;

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

// ---- Dense ternary weight panel — generic over Conv2DParams, unchanged. ----
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

// VNNI dpbusd accumulation core — byte-identical to every earlier stack
// benchmark. ih = oh*stride + ky + read_offset, read_offset = PAD - conv.pad_h
// (always 0 here: every layer1 conv is 3x3/stride1/pad1, matching PAD=1).
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
// Byte-identical to every earlier stack benchmark's kernel — ONE kernel
// serves both convs in both blocks here too (relu=true for conv1, false for
// conv2; stride=1, read_offset=0 throughout).
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

// q8 residual add + ReLU — byte-identical to every earlier stack benchmark.
// Here `b` is always the BLOCK INPUT directly (identity shortcut, no
// downsample projection — IC==OC==64, stride=1 means the shapes already
// match, so no projection conv is needed).
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

        // layer1.0
        const Conv2DParams& c10_conv1 = model.convs.at("layer1.0.conv1");
        const Conv2DParams& c10_conv2 = model.convs.at("layer1.0.conv2");
        const BNFusedParams& bn10_1   = model.fused_bns.at("layer1.0.bn1");
        const BNFusedParams& bn10_2   = model.fused_bns.at("layer1.0.bn2");
        // layer1.1
        const Conv2DParams& c11_conv1 = model.convs.at("layer1.1.conv1");
        const Conv2DParams& c11_conv2 = model.convs.at("layer1.1.conv2");
        const BNFusedParams& bn11_1   = model.fused_bns.at("layer1.1.bn1");
        const BNFusedParams& bn11_2   = model.fused_bns.at("layer1.1.bn2");

        const int IC0 = c10_conv1.in_channels;   // 64 (input to layer1.0: 56x56x64)
        const int OC  = c10_conv1.out_channels;  // 64 (identity shortcut: IC==OC, stride=1)
        const int H_IN = 56, W_IN = 56;
        const int OH = 56, OW = 56;

        if (IC0 % K_CHUNK != 0 || OC % OC_TILE != 0)
            throw std::runtime_error("channel counts must be divisible by 32 (IC) / 4 (OC)");
        if (IC0 != OC || c10_conv1.stride_h != 1)
            throw std::runtime_error("layer1 expected to be an identity-shortcut stage (IC==OC, stride=1)");

        std::cout << "=== benchmark_q2q_layer1_stack: full layer1 stack q8-to-q8 (VNNI) ===\n";
        std::cout << "  layer1.0: conv1 IC=" << c10_conv1.in_channels << " OC=" << c10_conv1.out_channels
                  << " | conv2 IC=" << c10_conv2.in_channels << " OC=" << c10_conv2.out_channels
                  << "  (identity shortcut -- no downsample projection)\n";
        std::cout << "  layer1.1: conv1 IC=" << c11_conv1.in_channels << " OC=" << c11_conv1.out_channels
                  << " | conv2 IC=" << c11_conv2.in_channels << " OC=" << c11_conv2.out_channels << "\n";
        std::cout << "  spatial: " << H_IN << "x" << W_IN << " -> " << OH << "x" << OW
                  << "  (" << (OH * OW) << " output pixels per conv -- largest in the network; "
                     "K=" << IC0 << " -- smallest)\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";
        std::cout << "  basic_engine layer1 stack (measured, 1T, q8 OCBlock2 bucket):\n"
                  << "    layer1.0 = conv1(2.25)+conv2(2.30)+add_relu(0.05) = "
                  << BASIC_ENGINE_L1_0_MS << " ms\n"
                  << "    layer1.1 = conv1(2.31)+conv2(2.30)+add_relu(0.05) = "
                  << BASIC_ENGINE_L1_1_MS << " ms\n"
                  << "    stack total                                       = "
                  << BASIC_ENGINE_STACK_MS << " ms\n";

        const std::vector<int8_t> p10_c1 = build_oc4_k32_panel(c10_conv1);
        const std::vector<int8_t> p10_c2 = build_oc4_k32_panel(c10_conv2);
        const std::vector<int8_t> p11_c1 = build_oc4_k32_panel(c11_conv1);
        const std::vector<int8_t> p11_c2 = build_oc4_k32_panel(c11_conv2);

        // Synthetic FP32 activation: stand-in for the conv1+maxpool output
        // feeding layer1 (56x56x64) — the only FP32 tensor anywhere in this
        // benchmark, touched only at the stack's entry boundary.
        std::vector<float> input_fp32((size_t)H_IN * W_IN * IC0);
        fill_random_fp32_nonneg(input_fp32.data(), input_fp32.size(), 0x9E3779B9u, 1.0f);

        QTensorNHWC input_q8_steady = make_q8_tensor(H_IN, W_IN, IC0, ACT_SCALE);
        quantize_nhwc_fp32_to_q8(input_fp32.data(), H_IN, W_IN, IC0, input_q8_steady);
        QTensorNHWC input_q8_cold = make_q8_tensor(H_IN, W_IN, IC0, ACT_SCALE);

        // layer1.0 scratch
        QTensorNHWC l10_mid       = make_q8_tensor(OH, OW, OC, ACT_SCALE);  // conv1 out
        QTensorNHWC l10_conv2_out = make_q8_tensor(OH, OW, OC, ACT_SCALE);
        QTensorNHWC l10_out       = make_q8_tensor(OH, OW, OC, ACT_SCALE);  // block out (-> l1.1 input)
        // layer1.1 scratch
        QTensorNHWC l11_mid       = make_q8_tensor(OH, OW, OC, ACT_SCALE);
        QTensorNHWC l11_conv2_out = make_q8_tensor(OH, OW, OC, ACT_SCALE);
        QTensorNHWC l11_out       = make_q8_tensor(OH, OW, OC, ACT_SCALE);  // stack out

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        // ---- stack_total_q2q_only: full stack, input already q8 ----
        auto run_q2q_only = [&]() {
            // layer1.0 -- residual operand is the BLOCK INPUT directly (identity)
            q2q_conv_bn_requant(input_q8_steady, p10_c1, c10_conv1, bn10_1, /*relu=*/true,  ACT_SCALE, l10_mid, OH, OW);
            q2q_conv_bn_requant(l10_mid,         p10_c2, c10_conv2, bn10_2, /*relu=*/false, ACT_SCALE, l10_conv2_out, OH, OW);
            q2q_residual_add_relu(l10_conv2_out, input_q8_steady, ACT_SCALE, OH, OW, OC, l10_out);
            // layer1.1
            q2q_conv_bn_requant(l10_out, p11_c1, c11_conv1, bn11_1, /*relu=*/true,  ACT_SCALE, l11_mid, OH, OW);
            q2q_conv_bn_requant(l11_mid, p11_c2, c11_conv2, bn11_2, /*relu=*/false, ACT_SCALE, l11_conv2_out, OH, OW);
            q2q_residual_add_relu(l11_conv2_out, l10_out, ACT_SCALE, OH, OW, OC, l11_out);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_q2q_only();

        std::vector<double> t_c10_c1(N_TIMED), t_c10_c2(N_TIMED), t_c10_res(N_TIMED),
                            t_c11_c1(N_TIMED), t_c11_c2(N_TIMED), t_c11_res(N_TIMED),
                            t_l10_total(N_TIMED), t_l11_total(N_TIMED), t_stack_only(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q_conv_bn_requant(input_q8_steady, p10_c1, c10_conv1, bn10_1, true,  ACT_SCALE, l10_mid, OH, OW);
            const double t1 = now_ms();
            q2q_conv_bn_requant(l10_mid, p10_c2, c10_conv2, bn10_2, false, ACT_SCALE, l10_conv2_out, OH, OW);
            const double t2 = now_ms();
            q2q_residual_add_relu(l10_conv2_out, input_q8_steady, ACT_SCALE, OH, OW, OC, l10_out);
            const double t3 = now_ms();
            q2q_conv_bn_requant(l10_out, p11_c1, c11_conv1, bn11_1, true, ACT_SCALE, l11_mid, OH, OW);
            const double t4 = now_ms();
            q2q_conv_bn_requant(l11_mid, p11_c2, c11_conv2, bn11_2, false, ACT_SCALE, l11_conv2_out, OH, OW);
            const double t5 = now_ms();
            q2q_residual_add_relu(l11_conv2_out, l10_out, ACT_SCALE, OH, OW, OC, l11_out);
            const double t6 = now_ms();

            t_c10_c1[i] = t1 - t0; t_c10_c2[i] = t2 - t1; t_c10_res[i] = t3 - t2;
            t_c11_c1[i] = t4 - t3; t_c11_c2[i] = t5 - t4; t_c11_res[i] = t6 - t5;
            t_l10_total[i] = t3 - t0; t_l11_total[i] = t6 - t3; t_stack_only[i] = t6 - t0;
        }
        volatile uint32_t sink1 = l11_out.data[0] + l11_out.data[l11_out.data.size() - 1];
        (void)sink1;

        // ---- stack_total_with_input_convert: + FP32->q8 of stack input ----
        auto run_with_convert = [&]() {
            quantize_nhwc_fp32_to_q8(input_fp32.data(), H_IN, W_IN, IC0, input_q8_cold);
            q2q_conv_bn_requant(input_q8_cold, p10_c1, c10_conv1, bn10_1, true,  ACT_SCALE, l10_mid, OH, OW);
            q2q_conv_bn_requant(l10_mid, p10_c2, c10_conv2, bn10_2, false, ACT_SCALE, l10_conv2_out, OH, OW);
            q2q_residual_add_relu(l10_conv2_out, input_q8_cold, ACT_SCALE, OH, OW, OC, l10_out);
            q2q_conv_bn_requant(l10_out, p11_c1, c11_conv1, bn11_1, true, ACT_SCALE, l11_mid, OH, OW);
            q2q_conv_bn_requant(l11_mid, p11_c2, c11_conv2, bn11_2, false, ACT_SCALE, l11_conv2_out, OH, OW);
            q2q_residual_add_relu(l11_conv2_out, l10_out, ACT_SCALE, OH, OW, OC, l11_out);
        };
        for (int i = 0; i < N_WARMUP; ++i) run_with_convert();

        std::vector<double> t_convert(N_TIMED), t_stack_with_convert(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_nhwc_fp32_to_q8(input_fp32.data(), H_IN, W_IN, IC0, input_q8_cold);
            const double t1 = now_ms();
            q2q_conv_bn_requant(input_q8_cold, p10_c1, c10_conv1, bn10_1, true,  ACT_SCALE, l10_mid, OH, OW);
            q2q_conv_bn_requant(l10_mid, p10_c2, c10_conv2, bn10_2, false, ACT_SCALE, l10_conv2_out, OH, OW);
            q2q_residual_add_relu(l10_conv2_out, input_q8_cold, ACT_SCALE, OH, OW, OC, l10_out);
            q2q_conv_bn_requant(l10_out, p11_c1, c11_conv1, bn11_1, true, ACT_SCALE, l11_mid, OH, OW);
            q2q_conv_bn_requant(l11_mid, p11_c2, c11_conv2, bn11_2, false, ACT_SCALE, l11_conv2_out, OH, OW);
            q2q_residual_add_relu(l11_conv2_out, l10_out, ACT_SCALE, OH, OW, OC, l11_out);
            const double t2 = now_ms();
            t_convert[i] = t1 - t0; t_stack_with_convert[i] = t2 - t0;
        }
        volatile uint32_t sink2 = l11_out.data[0] + l11_out.data[l11_out.data.size() - 1];
        (void)sink2;

        std::cout << "\n=== layer1 stack q8-to-q8 timing (over " << N_TIMED << " runs) ===\n";
        std::cout << "-- layer1.0 breakdown --\n";
        print_stats("conv1 (3x3 s1, VNNI+BN+ReLU+requant)", t_c10_c1, N_TIMED);
        print_stats("conv2 (3x3 s1, VNNI+BN+requant)     ", t_c10_c2, N_TIMED);
        print_stats("residual (q8 add+ReLU+requant)      ", t_c10_res, N_TIMED);
        print_stats("layer1.0 total                      ", t_l10_total, N_TIMED);
        std::cout << "-- layer1.1 breakdown --\n";
        print_stats("conv1 (3x3 s1, VNNI+BN+ReLU+requant)", t_c11_c1, N_TIMED);
        print_stats("conv2 (3x3 s1, VNNI+BN+requant)     ", t_c11_c2, N_TIMED);
        print_stats("residual (q8 add+ReLU+requant)      ", t_c11_res, N_TIMED);
        print_stats("layer1.1 total                      ", t_l11_total, N_TIMED);

        std::cout << "\n-- stack_total_q2q_only (input already q8) --\n";
        print_stats("stack_total_q2q_only                ", t_stack_only, N_TIMED);
        std::cout << "\n-- stack_total_with_input_convert (+ FP32->q8 of stack input) --\n";
        print_stats("input_convert (FP32 -> q8)          ", t_convert, N_TIMED);
        print_stats("stack_total_with_input_convert      ", t_stack_with_convert, N_TIMED);

        std::vector<double> s_only = t_stack_only, s_conv = t_stack_with_convert;
        std::sort(s_only.begin(), s_only.end());
        std::sort(s_conv.begin(), s_conv.end());
        const double med_only = s_only[N_TIMED / 2];
        const double med_conv = s_conv[N_TIMED / 2];

        std::cout << "\n=== gate: q2q layer1 stack vs basic_engine layer1 stack ===\n";
        std::cout << "  basic_engine layer1 stack (l1.0+l1.1):  " << BASIC_ENGINE_STACK_MS << " ms\n";
        std::cout << "  stack_total_q2q_only (med):             " << med_only << " ms\n";
        std::cout << "  stack_total_with_input_convert (med):   " << med_conv << " ms\n";

        if (med_only < BASIC_ENGINE_STACK_MS) {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) BEATS basic_engine ("
                      << BASIC_ENGINE_STACK_MS << " ms) by "
                      << (BASIC_ENGINE_STACK_MS / med_only) << "x -- full layer1 stack wins.\n";
        } else {
            std::cout << "  RESULT: q2q_only (" << med_only << " ms) does NOT beat basic_engine ("
                      << BASIC_ENGINE_STACK_MS << " ms) -- layer1's small K (" << IC0
                      << ") does not give VNNI enough per-pixel work to amortize overhead at "
                         "this large spatial extent. Stop here, do not integrate layer1.\n";
        }
        if (med_conv < BASIC_ENGINE_STACK_MS) {
            std::cout << "  (with_input_convert also wins: " << med_conv << " ms < "
                      << BASIC_ENGINE_STACK_MS << " ms.)\n";
        } else {
            std::cout << "  (with_input_convert does NOT win: " << med_conv << " ms >= "
                      << BASIC_ENGINE_STACK_MS << " ms.)\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
