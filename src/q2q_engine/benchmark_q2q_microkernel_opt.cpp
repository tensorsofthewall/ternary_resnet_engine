// benchmark_q2q_microkernel_opt.cpp — microkernel tuning for the dense Q2Q convs
//
// q2q_conv_bn_requant / vnni_oc4_dots / q2q_residual_add_relu had been kept
// byte-identical through earlier development; this benchmark explores tuning
// them. It builds THREE proposed optimizations and times each one PAIRED
// against its baseline (same data, same panel, back-to-back in one loop) so
// the comparison is direct and noise-free:
//
//   (1) "A-panel interior fast path": vnni_oc4_n2_dots register-blocks TWO
//       adjacent output columns per weight-panel pass for stride=1 dense
//       convs (the "interior" / non-downsample majority of layer1-4). Adjacent
//       output columns share overlapping receptive fields offset by exactly
//       C_IN bytes, so ONE set of w0..w3 loads serves BOTH -- halving weight-
//       load traffic versus the per-pixel vnni_oc4_dots. 8 acc + 4 weight + 2
//       activation = 14 YMM, fits AVX2's 16 with no spill.
//   (2) q2q_residual_add_relu_avx2: the scalar per-channel
//       dequant->add->relu->requantize loop, vectorized 8-wide with
//       _mm256_cvtepu8_epi32 / cvtepi32_ps / fmadd_ps / cvtps_epi32 / pack.
//   (3) epilogue_oc4_simd: the BN-scale+bias+ReLU+requantize epilogue
//       (4 scalar std::lround+clamp+byte-stores per output pixel), replaced
//       with one SSE float pipeline + one 4-byte store.
//
// q2q_conv_bn_requant_apanel = vnni_oc4_n2_dots (1, when applicable) +
// epilogue_oc4_simd (3) -- both land in the same kernel since the
// epilogue runs once per accumulator group regardless of tiling.
//
// Representative shape: layer3.1.conv2 (256ch, 14x14, 3x3 stride=1) -- the
// same dense-stride-1 family the "Beat ORT" packed-kernel exploration used as
// its baseline target, and large enough (K=256) to amortize pack/epilogue
// overhead while small enough to iterate quickly.
//
// Do not touch basic_engine.
//
// Usage: ./benchmark_q2q_microkernel_opt <model.bin>

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"

#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int OC_TILE = 4;
constexpr int K_CHUNK = 32;
constexpr int PAD = 1;
constexpr float ACT_SCALE = 0.05f;

struct QTensorNHWC {
    int H = 0, W = 0, C = 0;
    float scale = 1.0f;
    std::vector<uint8_t> data;
};

static QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, float scale) {
    QTensorNHWC t;
    t.H = H_logical + 2 * PAD; t.W = W_logical + 2 * PAD; t.C = C; t.scale = scale;
    t.data.assign((size_t)t.H * t.W * C, uint8_t(0));
    return t;
}

static void fill_random_q8(QTensorNHWC& t, uint32_t seed) {
    uint32_t rng = seed;
    for (int h = 0; h < t.H - 2 * PAD; ++h) {
        uint8_t* row = t.data.data() + ((size_t)(h + PAD) * t.W + PAD) * t.C;
        for (int i = 0; i < (t.W - 2 * PAD) * t.C; ++i) {
            rng = rng * 1664525u + 1013904223u;
            row[i] = static_cast<uint8_t>(rng >> 24);
        }
    }
}

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

// =====================================================================
// BASELINE kernels -- byte-identical to every earlier stack benchmark.
// =====================================================================
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

// =====================================================================
// OPTIMIZED kernels (1 / 2 / 3 from the header comment above)
// =====================================================================

// (3) Vectorized BN-scale+bias + optional ReLU + requantize epilogue:
// 4 lanes (OC_TILE) at once via one SSE float pipeline + one 4-byte store,
// replacing 4x scalar std::lround + clamp + per-byte store.
static inline void epilogue_oc4_simd(
    int32_t d0, int32_t d1, int32_t d2, int32_t d3,
    const float s[OC_TILE], const float b[OC_TILE],
    bool relu, float inv_out_scale, uint8_t* out_px)
{
    const __m128i dvec = _mm_setr_epi32(d0, d1, d2, d3);
    const __m128  sv   = _mm_loadu_ps(s);
    const __m128  bv   = _mm_loadu_ps(b);
    __m128 vf = _mm_fmadd_ps(_mm_cvtepi32_ps(dvec), sv, bv);
    if (relu) vf = _mm_max_ps(vf, _mm_setzero_ps());
    vf = _mm_mul_ps(vf, _mm_set1_ps(inv_out_scale));
    __m128i qi = _mm_cvtps_epi32(vf);                 // round-to-nearest-even (vs lround's half-away-from-zero -- fine, correctness doesn't matter)
    qi = _mm_max_epi32(qi, _mm_setzero_si128());
    qi = _mm_min_epi32(qi, _mm_set1_epi32(255));
    const __m128i packed16 = _mm_packus_epi32(qi, qi);
    const __m128i packed8  = _mm_packus_epi16(packed16, packed16);
    const uint32_t out32 = static_cast<uint32_t>(_mm_cvtsi128_si32(packed8));
    std::memcpy(out_px, &out32, sizeof(out32));
}

// (1) "A-panel interior fast path": register-blocks TWO adjacent output
// columns (ow, ow+1) per weight-panel pass for stride=1 convs. They share
// overlapping receptive fields offset by exactly C_IN bytes -- one w0..w3
// load set serves both, halving weight-load traffic vs. vnni_oc4_dots.
static inline void vnni_oc4_n2_dots(
    const uint8_t* base_in, int W_PAD_IN, int C_IN,
    const int8_t* panel_tile, const PanelStrides& st,
    int IC_CHUNKS, int KH, int KW, int oh, int ow, int read_offset,
    int32_t d[2][OC_TILE])
{
    __m256i acc[2][OC_TILE];
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < OC_TILE; ++j) acc[n][j] = _mm256_setzero_si256();

    for (int ky = 0; ky < KH; ++ky) {
        const int ih = oh + ky + read_offset; // stride == 1
        const int8_t* panel_ky = panel_tile + (size_t)ky * st.st_ky;
        for (int kx = 0; kx < KW; ++kx) {
            const int iw = ow + kx + read_offset;
            const uint8_t* a_row = base_in + ((size_t)ih * W_PAD_IN + iw) * C_IN;
            const int8_t* panel_kx = panel_ky + (size_t)kx * st.st_kx;
            for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                const size_t off = (size_t)icc * K_CHUNK;
                const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_row + off));
                const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_row + C_IN + off));
                const int8_t* wp = panel_kx + (size_t)icc * st.st_icc;
                const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 0*K_CHUNK));
                const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 1*K_CHUNK));
                const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 2*K_CHUNK));
                const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 3*K_CHUNK));
                acc[0][0] = _mm256_dpbusd_epi32(acc[0][0], a0, w0);
                acc[0][1] = _mm256_dpbusd_epi32(acc[0][1], a0, w1);
                acc[0][2] = _mm256_dpbusd_epi32(acc[0][2], a0, w2);
                acc[0][3] = _mm256_dpbusd_epi32(acc[0][3], a0, w3);
                acc[1][0] = _mm256_dpbusd_epi32(acc[1][0], a1, w0);
                acc[1][1] = _mm256_dpbusd_epi32(acc[1][1], a1, w1);
                acc[1][2] = _mm256_dpbusd_epi32(acc[1][2], a1, w2);
                acc[1][3] = _mm256_dpbusd_epi32(acc[1][3], a1, w3);
            }
        }
    }
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < OC_TILE; ++j) d[n][j] = hsum_epi32(acc[n][j]);
}

// q2q_conv_bn_requant_apanel = vnni_oc4_n2_dots (1, stride=1 + even OW)
// or vnni_oc4_dots fallback (stride=2 / odd-OW tail) + epilogue_oc4_simd
// (3, always). This is the proposed drop-in replacement for
// q2q_conv_bn_requant across the Q2Q trunk.
static void q2q_conv_bn_requant_apanel(
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
    const bool use_n2 = (stride == 1) && (OW % 2 == 0);

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
            if (use_n2) {
                for (int ow = 0; ow < OW; ow += 2) {
                    int32_t d[2][OC_TILE];
                    vnni_oc4_n2_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW,
                                     oh, ow, read_offset, d);
                    epilogue_oc4_simd(d[0][0], d[0][1], d[0][2], d[0][3], s, b, relu, inv_out_scale,
                                      out_row + (size_t)ow * y.C);
                    epilogue_oc4_simd(d[1][0], d[1][1], d[1][2], d[1][3], s, b, relu, inv_out_scale,
                                      out_row + (size_t)(ow + 1) * y.C);
                }
            } else {
                for (int ow = 0; ow < OW; ++ow) {
                    int32_t d0, d1, d2, d3;
                    vnni_oc4_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW,
                                  oh, ow, stride, read_offset, d0, d1, d2, d3);
                    epilogue_oc4_simd(d0, d1, d2, d3, s, b, relu, inv_out_scale,
                                      out_row + (size_t)ow * y.C);
                }
            }
        }
    }
}

// (2) AVX2 8-wide vectorized residual add + ReLU + requantize. C is
// always a multiple of 32 for every q8 tensor in this engine -- no tail.
static void q2q_residual_add_relu_avx2(
    const QTensorNHWC& a, const QTensorNHWC& b,
    float out_scale, int H, int W, int C, QTensorNHWC& y)
{
    const float inv_out_scale = 1.0f / out_scale;
    const __m256 sa_v   = _mm256_set1_ps(a.scale);
    const __m256 sb_v   = _mm256_set1_ps(b.scale);
    const __m256 inv_os = _mm256_set1_ps(inv_out_scale);
    const __m256 zero   = _mm256_setzero_ps();
    const __m256i max255 = _mm256_set1_epi32(255);
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
            for (int c = 0; c < C; c += 8) {
                const __m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ap + c));
                const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp + c));
                const __m256 af = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a8));
                const __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8));
                __m256 sum = _mm256_fmadd_ps(af, sa_v, _mm256_mul_ps(bf, sb_v));
                sum = _mm256_max_ps(sum, zero);
                const __m256 scaled = _mm256_mul_ps(sum, inv_os);
                __m256i qi = _mm256_cvtps_epi32(scaled);
                qi = _mm256_max_epi32(qi, _mm256_setzero_si256());
                qi = _mm256_min_epi32(qi, max255);
                const __m128i lo = _mm256_castsi256_si128(qi);
                const __m128i hi = _mm256_extracti128_si256(qi, 1);
                const __m128i packed16 = _mm_packus_epi32(lo, hi);
                const __m128i packed8  = _mm_packus_epi16(packed16, packed16);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(yp + c), packed8);
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

        // Representative dense stride=1 3x3 conv -- layer3.1.conv2
        // (256ch, 14x14): the same family/shape the "Beat ORT" packed-kernel
        // exploration benchmarked as its baseline target (~2.5ms in basic_engine).
        const Conv2DParams& conv  = model.convs.at("layer3.1.conv2");
        const BNFusedParams& bn   = model.fused_bns.at("layer3.1.bn2");
        const int C = conv.in_channels;   // 256 (== out_channels, identity block)
        const int H = 14, W = 14;

        if (C % K_CHUNK != 0 || conv.out_channels % OC_TILE != 0 || conv.stride_h != 1)
            throw std::runtime_error("expected layer3.1.conv2 to be dense IC=OC, stride=1, 3x3");

        std::cout << "=== benchmark_q2q_microkernel_opt: dense-conv/residual microkernel tuning (paired A/B, same data) ===\n";
        std::cout << "  representative conv: layer3.1.conv2  IC=" << C << " OC=" << conv.out_channels
                  << "  " << H << "x" << W << "  (3x3 stride=1 -- 'interior' dense family)\n";
        std::cout << "  ACT_SCALE: " << ACT_SCALE << "\n";

        const std::vector<int8_t> panel = build_oc4_k32_panel(conv);

        QTensorNHWC input = make_q8_tensor(H, W, C, ACT_SCALE);
        fill_random_q8(input, 0xC0FFEEu);
        QTensorNHWC out_orig   = make_q8_tensor(H, W, C, ACT_SCALE);
        QTensorNHWC out_apanel = make_q8_tensor(H, W, C, ACT_SCALE);

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 200;

        // ---- (1)+(3) paired: q2q_conv_bn_requant vs q2q_conv_bn_requant_apanel ----
        for (int i = 0; i < N_WARMUP; ++i) {
            q2q_conv_bn_requant(input, panel, conv, bn, true, ACT_SCALE, out_orig, H, W);
            q2q_conv_bn_requant_apanel(input, panel, conv, bn, true, ACT_SCALE, out_apanel, H, W);
        }
        std::vector<double> t_orig_conv(N_TIMED), t_apanel_conv(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q_conv_bn_requant(input, panel, conv, bn, true, ACT_SCALE, out_orig, H, W);
            const double t1 = now_ms();
            q2q_conv_bn_requant_apanel(input, panel, conv, bn, true, ACT_SCALE, out_apanel, H, W);
            const double t2 = now_ms();
            t_orig_conv[i]   = t1 - t0;
            t_apanel_conv[i] = t2 - t1;
        }
        size_t mismatches_conv = 0;
        for (size_t i = 0; i < out_orig.data.size(); ++i)
            if (out_orig.data[i] != out_apanel.data[i]) ++mismatches_conv;

        std::cout << "\n--- conv kernel (orig vs A-panel N_TILE=2 + SIMD epilogue), "
                  << N_TIMED << " paired runs ---\n";
        print_stats("q2q_conv_bn_requant         (baseline)        ", t_orig_conv, N_TIMED);
        print_stats("q2q_conv_bn_requant_apanel  (tuned)           ", t_apanel_conv, N_TIMED);
        {
            std::vector<double> so = t_orig_conv, sa = t_apanel_conv;
            std::sort(so.begin(), so.end()); std::sort(sa.begin(), sa.end());
            const double mo = so[N_TIMED/2], ma = sa[N_TIMED/2];
            std::cout << "  output bytes differing (rounding-mode delta only, expected nonzero): "
                      << mismatches_conv << " / " << out_orig.data.size() << "\n";
            if (ma < mo)
                std::cout << "  RESULT: apanel (" << ma << " ms) BEATS baseline (" << mo
                          << " ms) by " << (mo / ma) << "x -- integrate.\n";
            else
                std::cout << "  RESULT: apanel (" << ma << " ms) does NOT beat baseline (" << mo
                          << " ms) -- " << (ma / mo) << "x slower; keep baseline.\n";
        }

        // ---- (2) paired: q2q_residual_add_relu vs q2q_residual_add_relu_avx2 ----
        QTensorNHWC ra = make_q8_tensor(H, W, C, ACT_SCALE);
        QTensorNHWC rb = make_q8_tensor(H, W, C, ACT_SCALE);
        fill_random_q8(ra, 0x9E3779B9u);
        fill_random_q8(rb, 0x2545F491u);
        QTensorNHWC res_orig  = make_q8_tensor(H, W, C, ACT_SCALE);
        QTensorNHWC res_avx2  = make_q8_tensor(H, W, C, ACT_SCALE);

        for (int i = 0; i < N_WARMUP; ++i) {
            q2q_residual_add_relu(ra, rb, ACT_SCALE, H, W, C, res_orig);
            q2q_residual_add_relu_avx2(ra, rb, ACT_SCALE, H, W, C, res_avx2);
        }
        std::vector<double> t_orig_res(N_TIMED), t_avx2_res(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            q2q_residual_add_relu(ra, rb, ACT_SCALE, H, W, C, res_orig);
            const double t1 = now_ms();
            q2q_residual_add_relu_avx2(ra, rb, ACT_SCALE, H, W, C, res_avx2);
            const double t2 = now_ms();
            t_orig_res[i] = t1 - t0;
            t_avx2_res[i] = t2 - t1;
        }
        size_t mismatches_res = 0;
        for (size_t i = 0; i < res_orig.data.size(); ++i)
            if (res_orig.data[i] != res_avx2.data[i]) ++mismatches_res;

        std::cout << "\n--- residual add+ReLU (scalar vs AVX2 8-wide), "
                  << N_TIMED << " paired runs ---\n";
        print_stats("q2q_residual_add_relu       (baseline)        ", t_orig_res, N_TIMED);
        print_stats("q2q_residual_add_relu_avx2  (tuned)           ", t_avx2_res, N_TIMED);
        {
            std::vector<double> so = t_orig_res, sa = t_avx2_res;
            std::sort(so.begin(), so.end()); std::sort(sa.begin(), sa.end());
            const double mo = so[N_TIMED/2], ma = sa[N_TIMED/2];
            std::cout << "  output bytes differing (rounding-mode delta only, expected nonzero): "
                      << mismatches_res << " / " << res_orig.data.size() << "\n";
            if (ma < mo)
                std::cout << "  RESULT: avx2 (" << ma << " ms) BEATS baseline (" << mo
                          << " ms) by " << (mo / ma) << "x -- integrate.\n";
            else
                std::cout << "  RESULT: avx2 (" << ma << " ms) does NOT beat baseline (" << mo
                          << " ms) -- " << (ma / mo) << "x slower; keep baseline.\n";
        }

        volatile uint32_t sink = out_orig.data[0] + out_apanel.data[0]
                               + res_orig.data[0] + res_avx2.data[0];
        (void)sink;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
