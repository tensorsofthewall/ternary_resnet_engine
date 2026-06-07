// benchmark_q2q_full_prefix_optimized.cpp — Q2Q Phase 10, item 6: re-run with
// items 3-5 wired in
//
// Same all-Q2Q forward prefix as benchmark_q2q_full_prefix.cpp (Phase 10b):
//   FP32 input -> quantize -> conv1(Q2Q) -> maxpool(q8) -> layer1-4(Q2Q)
//   -> dequant -> basic_engine avgpool+fc
// but with the THREE microkernel optimizations from Phase 10 items 3-5
// (validated standalone in benchmark_q2q_microkernel_opt.cpp, ~2.08x and
// ~17x paired wins respectively) wired into every layer1-4 conv/residual call:
//
//   q2q_conv_bn_requant        -> q2q_conv_bn_requant_apanel   (items 3+5)
//   q2q_residual_add_relu      -> q2q_residual_add_relu_avx2   (item 4)
//
// conv1 (im2col VNNI) and maxpool (q8-direct) are UNCHANGED -- items 3-5 only
// target the dense layer1-4 family (the validated ~1.85-1.88x trunk), so this
// is where the wins land.
//
// To keep the comparison live and fair, BOTH pipelines run in this one binary
// back-to-back (original-kernel "baseline_total" replicating Phase 10b, then
// optimized-kernel "optimized_total"), sharing one scratch-tensor set across
// the two self-contained warmup+timed phases -- the same pattern
// benchmark_q2q_layer1_stack used for its "q2q_only" vs "with_convert" phases.
// Both are gated against a freshly, live-measured basic_engine full-model cost.
//
// Generic run_layerN lambdas are parameterized over (conv_fn, res_fn) via
// C++14 generic lambdas (auto&& params) -- zero overhead, the compiler
// generates a distinct instantiation per kernel pair, so orchestration code
// is written once and shared by both phases.
//
// Do not touch basic_engine.
//
// Usage: ./benchmark_q2q_full_prefix_optimized <model.bin>
//   (reads artifacts/golden_input_fp32.bin as the model input)

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"
#include "resnet18_engine.hpp"
#include "ops.hpp"

#include <immintrin.h>
#include <omp.h>
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
constexpr int PAD_Q8 = 1;
constexpr float ACT_SCALE = 0.05f;

struct QTensorNHWC {
    int H = 0, W = 0, C = 0;
    int border = 0;
    float scale = 1.0f;
    std::vector<uint8_t> data;
};

static QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, int border, float scale) {
    QTensorNHWC t;
    t.H = H_logical + 2 * border; t.W = W_logical + 2 * border; t.C = C;
    t.border = border; t.scale = scale;
    t.data.assign((size_t)t.H * t.W * C, uint8_t(0));
    return t;
}

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

static void dequantize_q8_nhwc_to_fp32_nchw(const QTensorNHWC& src, int H, int W, int C, Tensor& dst) {
    const int b = src.border;
    for (int y = 0; y < H; ++y) {
        const uint8_t* row = src.data.data() + ((size_t)(y + b) * src.W + b) * C;
        for (int x = 0; x < W; ++x) {
            const uint8_t* px = row + (size_t)x * C;
            for (int c = 0; c < C; ++c)
                dst.data[((size_t)c * H + y) * W + x] = static_cast<float>(px[c]) * src.scale;
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

// Vectorized BN-scale + bias + (optional) ReLU + requantize-to-u8 for an
// OC_TILE=4 group, replacing 4x scalar requantize_u8/std::lround with one
// SSE pipeline + a single 4-byte store. Validated standalone (~2.08x paired
// win in benchmark_q2q_microkernel_opt; ~4.6x for conv1's larger pixel count
// in benchmark_q2q_conv1_opt) -- shared by both the conv1 kernel and the
// dense apanel conv below.
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
    __m128i qi = _mm_cvtps_epi32(vf);
    qi = _mm_max_epi32(qi, _mm_setzero_si128());
    qi = _mm_min_epi32(qi, _mm_set1_epi32(255));
    const __m128i packed16 = _mm_packus_epi32(qi, qi);
    const __m128i packed8  = _mm_packus_epi16(packed16, packed16);
    const uint32_t out32 = static_cast<uint32_t>(_mm_cvtsi128_si32(packed8));
    std::memcpy(out_px, &out32, sizeof(out32));
}

// =====================================================================
// conv1 im2col VNNI + q8 maxpool -- conv1 optimized via paired A/B/C
// validation in benchmark_q2q_conv1_opt (~4.6x standalone win, 0/831744
// byte mismatches): panel flattening reordered from k=ic*49+ky*7+kx to
// k=(ky*7+kx)*IC+ic so each (ky,kx) position's IC=3 source channels are
// contiguous in NHWC, enabling a 4-byte load+store gather (49 simple
// mov pairs) instead of 147 strided scalar byte copies; scalar
// requantize_u8 epilogue replaced by epilogue_oc4_simd. Bit-exact with
// the original im2col kernel -- only memory layout/ordering changed.
// maxpool is q8-direct and untouched.
// =====================================================================
struct Conv1Panel {
    int K_PAD = 0, IC_CHUNKS = 0, OC_TILES = 0;
    std::vector<int8_t> data;
};

static Conv1Panel build_conv1_im2col_panel(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int K_FULL = IC * KH * KW;
    const int IC_CHUNKS = (K_FULL + K_CHUNK - 1) / K_CHUNK;
    const int K_PAD = IC_CHUNKS * K_CHUNK;
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
                    const int kg = icc * K_CHUNK + k;
                    int8_t w = 0;
                    if (kg < K_FULL) {
                        const int ky = kg / (KW * IC);
                        const int rem = kg % (KW * IC);
                        const int kx = rem / IC;
                        const int ic = rem % IC;
                        w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                    }
                    const size_t idx = (((size_t)oc_tile * IC_CHUNKS + icc) * OC_TILE + j) * K_CHUNK + k;
                    panel.data[idx] = w;
                }
            }
    return panel;
}

// Gathers im2col columns for one output pixel. For each (ky,kx), the IC=3
// source channels are contiguous in NHWC and the destination slot is
// contiguous in the panel-B layout above -- one 4-byte load + 4-byte store
// per (ky,kx) (49 total) instead of 3 strided scalar byte copies x 49 (147
// total). The 4th (garbage) byte written by iteration i is overwritten by
// iteration i+1's valid byte 0, except the LAST iteration's spill into
// a_panel[K_FULL], which the zero-fill below (always run after the gather)
// resets to 0. Reads stay in-bounds: max (iw0+KW-1)*C+4 = 228*3+4 = 688
// < x.W*x.C = 230*3 = 690.
static inline void conv1_gather_fast(
    const uint8_t* base_in, int W, int C, int ih0, int iw0,
    int KH, int KW, int IC, uint8_t* a_panel)
{
    for (int ky = 0; ky < KH; ++ky) {
        const uint8_t* row = base_in + ((size_t)(ih0 + ky) * W + iw0) * C;
        uint8_t* dst_row = a_panel + (size_t)ky * KW * IC;
        for (int kx = 0; kx < KW; ++kx) {
            uint32_t v;
            std::memcpy(&v, row + (size_t)kx * C, sizeof(v));
            std::memcpy(dst_row + (size_t)kx * IC, &v, sizeof(v));
        }
    }
}

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

    // Precompute per-OC scale/bias once -- shared, read-only across threads.
    std::vector<float> s_all((size_t)OC_TILES * OC_TILE), b_all((size_t)OC_TILES * OC_TILE);
    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        const int oc_base = oc_tile * OC_TILE;
        for (int j = 0; j < OC_TILE; ++j) {
            s_all[oc_base + j] = bn.scale[oc_base + j] * dq;
            b_all[oc_base + j] = bn.bias[oc_base + j];
        }
    }

    // Phase 12: parallelize over output rows (M = OH*OW = 12544 >> 196, so a
    // row-granularity m_tile gives every thread plenty of independent work).
    // Each thread packs its OWN a_panel buffer once (outside the #pragma omp
    // for) and reuses it across all rows it's assigned -- "thread-local A-panel,
    // not shared, no malloc inside the timed loop" per spec: this allocates
    // K_PAD+4 (~164) bytes once per thread per call, not per pixel.
    #pragma omp parallel
    {
        std::vector<uint8_t> a_panel(K_PAD + 4, uint8_t(0));  // +4 slack for the 4-byte gather tail write

        #pragma omp for schedule(static)
        for (int oh = 0; oh < OH; ++oh) {
            const int ih0 = oh * stride - conv.pad_h + b;
            uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C;
            for (int ow = 0; ow < OW; ++ow) {
                const int iw0 = ow * stride - conv.pad_w + b;
                conv1_gather_fast(base_in, x.W, x.C, ih0, iw0, KH, KW, IC, a_panel.data());
                for (int kg = K_FULL; kg < K_PAD; ++kg) a_panel[kg] = 0;

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

                    epilogue_oc4_simd(d0, d1, d2, d3, &s_all[oc_base], &b_all[oc_base],
                                      /*relu=*/true, inv_out_scale, out_px + oc_base);
                }
            }
        }
    }
}

static void q8_maxpool_3x3_s2_p1(const QTensorNHWC& x, int OH, int OW, QTensorNHWC& y) {
    constexpr int stride = 2, pad = 1;
    const int C = x.C;
    const int read_offset = x.border - pad;
    const uint8_t* base_in  = x.data.data();
    uint8_t*       base_out = y.data.data();

    // Phase 12: parallelize over output rows -- each row reads an overlapping
    // (read-only, safe) 3-row band of the input and writes a disjoint output
    // row, so there are no races.
    #pragma omp parallel for schedule(static)
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
}

// =====================================================================
// Dense panel + PanelStrides -- generic, unchanged.
// =====================================================================
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
// BASELINE dense conv + residual -- byte-for-byte (mod border-generalization,
// a no-op here) from benchmark_q2q_full_prefix.cpp / *_layer234_stack.cpp.
// Replicates the Phase 10b numbers in this binary's "baseline" phase.
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
    const int read_offset = x.border - conv.pad_h;
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
            uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C + oc_base;
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
        const uint8_t* a_row = abase + ((size_t)(h + a.border) * a.W + a.border) * C;
        const uint8_t* b_row = bbase + ((size_t)(h + b.border) * b.W + b.border) * C;
        uint8_t*       y_row = ybase + ((size_t)(h + y.border) * y.W + y.border) * C;
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
// OPTIMIZED dense conv + residual -- Phase 10 items 3/4/5, byte-for-byte
// from benchmark_q2q_microkernel_opt.cpp (validated there: ~2.08x / ~17x
// paired wins). border-generic (q2q_conv_bn_requant_apanel uses x.border /
// y.border instead of a fixed PAD, matching the baseline's generalization;
// every tensor it touches here has border=PAD_Q8=1, so this is a no-op).
// epilogue_oc4_simd is shared with conv1 -- defined earlier in this file.
// =====================================================================
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
        const int ih = oh + ky + read_offset;
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
    const int read_offset = x.border - conv.pad_h;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const PanelStrides st = PanelStrides::make(IC, KH, KW);
    const bool use_n2 = (stride == 1) && (OW % 2 == 0);

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data();
    uint8_t*       base_out   = y.data.data();

    // Precompute per-OC scale/bias once -- tiny (OC floats each), read-only,
    // shared across threads. This keeps the (oc_tile, oh) loop nest perfectly
    // collapsible (OpenMP collapse() requires no code between the collapsed
    // loop heads).
    std::vector<float> s_all((size_t)OC), b_all((size_t)OC);
    for (int oc = 0; oc < OC; ++oc) {
        s_all[oc] = bn.scale[oc] * dq;
        b_all[oc] = bn.bias[oc];
    }

    // Phase 12: parallelize over the flattened (oc_tile, oh) task space via
    // collapse(2). Each task writes a disjoint OC_TILE-channel x one-output-row
    // region (oc_base..oc_base+3, row oh) -- no overlap, no races. Flattening
    // both dimensions (rather than just oh) gives every thread count plenty of
    // independent units even for layer4's small spatial extent (OH=7,
    // OC_TILES=128 -> 896 tasks vs. just 7 if parallelizing over oh alone).
    #pragma omp parallel for collapse(2) schedule(static)
    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        for (int oh = 0; oh < OH; ++oh) {
            const int oc_base = oc_tile * OC_TILE;
            const float* s = &s_all[oc_base];
            const float* b = &b_all[oc_base];
            const int8_t* panel_tile = base_panel + (size_t)oc_tile * st.st_oct;
            uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C + oc_base;
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

    // Phase 12: parallelize over the flattened (h, w) element-tile space --
    // collapsing both spatial dims (rather than just rows) keeps every thread
    // fed even for layer4's small H*W=49; each (h,w) tile writes a disjoint
    // C-channel run of the output, so there are no races.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const uint8_t* ap = abase + ((size_t)(h + a.border) * a.W + a.border) * C + (size_t)w * C;
            const uint8_t* bp = bbase + ((size_t)(h + b.border) * b.W + b.border) * C + (size_t)w * C;
            uint8_t*       yp = ybase + ((size_t)(h + y.border) * y.W + y.border) * C + (size_t)w * C;
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

static std::vector<float> read_f32_file(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Could not open file: " + path);
    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));
    if (!f) throw std::runtime_error("Failed to read expected float count from: " + path);
    return data;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> [--threads N]\n";
        return 1;
    }

    int threads = 1;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }
    if (threads < 1) {
        std::cerr << "--threads must be >= 1\n";
        return 1;
    }
    omp_set_num_threads(threads);

    try {
        const Model model = load_model(argv[1]);

        const Conv2DParams& conv1 = model.convs.at("conv1");
        const BNFusedParams& bn1  = model.fused_bns.at("bn1");
        const Conv2DParams& c10_conv1 = model.convs.at("layer1.0.conv1");
        const Conv2DParams& c10_conv2 = model.convs.at("layer1.0.conv2");
        const BNFusedParams& bn10_1   = model.fused_bns.at("layer1.0.bn1");
        const BNFusedParams& bn10_2   = model.fused_bns.at("layer1.0.bn2");
        const Conv2DParams& c11_conv1 = model.convs.at("layer1.1.conv1");
        const Conv2DParams& c11_conv2 = model.convs.at("layer1.1.conv2");
        const BNFusedParams& bn11_1   = model.fused_bns.at("layer1.1.bn1");
        const BNFusedParams& bn11_2   = model.fused_bns.at("layer1.1.bn2");
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
        const Conv2DParams& c40_conv1 = model.convs.at("layer4.0.conv1");
        const Conv2DParams& c40_conv2 = model.convs.at("layer4.0.conv2");
        const Conv2DParams& c40_ds    = model.convs.at("layer4.0.downsample.0");
        const BNFusedParams& bn40_1   = model.fused_bns.at("layer4.0.bn1");
        const BNFusedParams& bn40_2   = model.fused_bns.at("layer4.0.bn2");
        const BNFusedParams& bn40_ds  = model.fused_bns.at("layer4.0.downsample.1");
        const Conv2DParams& c41_conv1 = model.convs.at("layer4.1.conv1");
        const Conv2DParams& c41_conv2 = model.convs.at("layer4.1.conv2");
        const BNFusedParams& bn41_1   = model.fused_bns.at("layer4.1.bn1");
        const BNFusedParams& bn41_2   = model.fused_bns.at("layer4.1.bn2");
        const LinearParams& fc        = model.linears.at("fc");

        const int IC_C1 = conv1.in_channels, OC_C1 = conv1.out_channels;
        const int H_IN = 224, W_IN = 224;
        const int OH1 = (H_IN + 2 * conv1.pad_h - conv1.kh) / conv1.stride_h + 1;
        const int OW1 = (W_IN + 2 * conv1.pad_w - conv1.kw) / conv1.stride_w + 1;
        const int OHp = (OH1 + 2 - 3) / 2 + 1;
        const int OWp = (OW1 + 2 - 3) / 2 + 1;
        const int OC1 = c10_conv1.out_channels;
        const int OC2 = c20_conv1.out_channels;
        const int OC3 = c30_conv1.out_channels;
        const int OC4 = c40_conv1.out_channels;
        const int H1 = OHp,  W1 = OWp;
        const int H2 = 28,   W2 = 28;
        const int H3 = 14,   W3 = 14;
        const int H4 = 7,    W4 = 7;

        std::cout << "=== benchmark_q2q_full_prefix_optimized: Phase 10 item 6 -- re-run with "
                     "items 3-5 wired in ===\n";
        std::cout << "  pipeline: FP32 -> quantize -> conv1(Q2Q) -> maxpool(q8) -> layer1-4(Q2Q) "
                     "-> dequant -> avgpool+fc\n";
        std::cout << "  baseline kernels  : q2q_conv_bn_requant      + q2q_residual_add_relu      (Phase 10b)\n";
        std::cout << "  optimized kernels : q2q_conv_bn_requant_apanel + q2q_residual_add_relu_avx2 "
                     "(items 3+4+5, ~2.08x / ~17x paired wins standalone)\n";
        std::cout << "  conv1 uses the fast-gather + SIMD-epilogue kernel in Phase B "
                     "(~4.6x standalone win, validated in benchmark_q2q_conv1_opt); "
                     "maxpool stays q8-direct/unchanged\n";
        std::cout << "  ACT_SCALE: " << ACT_SCALE << "\n";
        std::cout << "  Phase 12: optimized-path kernels (conv1, layer1-4 conv/residual, maxpool) "
                     "now run multi-threaded via OpenMP -- requested threads: " << threads
                  << " (omp_get_max_threads() = " << omp_get_max_threads() << ")\n";

        // ---- live basic_engine full-model baseline ----
        Tensor golden_input(3, 224, 224);
        golden_input.data = read_f32_file("artifacts/golden_input_fp32.bin", 3 * 224 * 224);

        EngineWorkspace ws;
        constexpr int N_WARMUP_BASIC = 10;
        constexpr int N_TIMED_BASIC  = 30;
        for (int i = 0; i < N_WARMUP_BASIC; ++i) {
            std::vector<TimingEntry> t;
            forward_resnet18_workspace_timed(model, golden_input, ws, t);
        }
        std::vector<double> t_basic_full(N_TIMED_BASIC);
        for (int i = 0; i < N_TIMED_BASIC; ++i) {
            std::vector<TimingEntry> t;
            forward_resnet18_workspace_timed(model, golden_input, ws, t);
            double full = 0.0;
            for (const auto& e : t) full += e.ms;
            t_basic_full[i] = full;
        }
        std::vector<double> s_basic = t_basic_full;
        std::sort(s_basic.begin(), s_basic.end());
        const double basic_full_ms = s_basic[N_TIMED_BASIC / 2];
        std::cout << "\n=== basic_engine full-model cost (measured live, " << N_TIMED_BASIC << " runs) ===\n";
        print_stats("basic_full_ms", t_basic_full, N_TIMED_BASIC);

        // ---- panels ----
        const Conv1Panel p_c1 = build_conv1_im2col_panel(conv1);
        const std::vector<int8_t> p10_c1 = build_oc4_k32_panel(c10_conv1);
        const std::vector<int8_t> p10_c2 = build_oc4_k32_panel(c10_conv2);
        const std::vector<int8_t> p11_c1 = build_oc4_k32_panel(c11_conv1);
        const std::vector<int8_t> p11_c2 = build_oc4_k32_panel(c11_conv2);
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
        const std::vector<int8_t> p40_c1 = build_oc4_k32_panel(c40_conv1);
        const std::vector<int8_t> p40_c2 = build_oc4_k32_panel(c40_conv2);
        const std::vector<int8_t> p40_ds = build_oc4_k32_panel(c40_ds);
        const std::vector<int8_t> p41_c1 = build_oc4_k32_panel(c41_conv1);
        const std::vector<int8_t> p41_c2 = build_oc4_k32_panel(c41_conv2);

        // ---- shared scratch tensors (safe to share across the two
        // sequential, self-contained phases -- no cross-phase dependency) ----
        QTensorNHWC input_q8   = make_q8_tensor(H_IN, W_IN, IC_C1, conv1.pad_h, ACT_SCALE);
        QTensorNHWC conv1_out  = make_q8_tensor(OH1, OW1, OC_C1, PAD_Q8, ACT_SCALE);
        QTensorNHWC pool_out   = make_q8_tensor(OHp, OWp, OC_C1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l10_mid       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l10_conv2_out = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l10_out       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l11_mid       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l11_conv2_out = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l11_out       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_mid       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_conv2_out = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_identity  = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_out       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l21_mid       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l21_conv2_out = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l21_out       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_mid       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_conv2_out = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_identity  = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_out       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l31_mid       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l31_conv2_out = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l31_out       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_mid       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_conv2_out = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_identity  = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_out       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l41_mid       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l41_conv2_out = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l41_out       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);

        Tensor dequant_out(OC4, H4, W4);
        Tensor avgpool_out(OC4, 1, 1);
        Tensor fc_out(1000, 1, 1);

        // ---- generic orchestration: parameterized over (conv_fn, res_fn).
        // C++14 generic lambdas -- the compiler instantiates one specialization
        // per (conv_fn, res_fn) pair, so baseline and optimized phases share
        // this code with zero indirection overhead. ----
        auto run_layer1 = [&](auto&& conv_fn, auto&& res_fn, const QTensorNHWC& in) {
            conv_fn(in,      p10_c1, c10_conv1, bn10_1, true,  ACT_SCALE, l10_mid, H1, W1);
            conv_fn(l10_mid, p10_c2, c10_conv2, bn10_2, false, ACT_SCALE, l10_conv2_out, H1, W1);
            res_fn(l10_conv2_out, in, ACT_SCALE, H1, W1, OC1, l10_out);
            conv_fn(l10_out, p11_c1, c11_conv1, bn11_1, true,  ACT_SCALE, l11_mid, H1, W1);
            conv_fn(l11_mid, p11_c2, c11_conv2, bn11_2, false, ACT_SCALE, l11_conv2_out, H1, W1);
            res_fn(l11_conv2_out, l10_out, ACT_SCALE, H1, W1, OC1, l11_out);
        };
        auto run_layer2 = [&](auto&& conv_fn, auto&& res_fn, const QTensorNHWC& in) {
            conv_fn(in,      p20_c1, c20_conv1, bn20_1, true,  ACT_SCALE, l20_mid, H2, W2);
            conv_fn(l20_mid, p20_c2, c20_conv2, bn20_2, false, ACT_SCALE, l20_conv2_out, H2, W2);
            conv_fn(in,      p20_ds, c20_ds,    bn20_ds,false, ACT_SCALE, l20_identity, H2, W2);
            res_fn(l20_conv2_out, l20_identity, ACT_SCALE, H2, W2, OC2, l20_out);
            conv_fn(l20_out, p21_c1, c21_conv1, bn21_1, true,  ACT_SCALE, l21_mid, H2, W2);
            conv_fn(l21_mid, p21_c2, c21_conv2, bn21_2, false, ACT_SCALE, l21_conv2_out, H2, W2);
            res_fn(l21_conv2_out, l20_out, ACT_SCALE, H2, W2, OC2, l21_out);
        };
        auto run_layer3 = [&](auto&& conv_fn, auto&& res_fn, const QTensorNHWC& in) {
            conv_fn(in,      p30_c1, c30_conv1, bn30_1, true,  ACT_SCALE, l30_mid, H3, W3);
            conv_fn(l30_mid, p30_c2, c30_conv2, bn30_2, false, ACT_SCALE, l30_conv2_out, H3, W3);
            conv_fn(in,      p30_ds, c30_ds,    bn30_ds,false, ACT_SCALE, l30_identity, H3, W3);
            res_fn(l30_conv2_out, l30_identity, ACT_SCALE, H3, W3, OC3, l30_out);
            conv_fn(l30_out, p31_c1, c31_conv1, bn31_1, true,  ACT_SCALE, l31_mid, H3, W3);
            conv_fn(l31_mid, p31_c2, c31_conv2, bn31_2, false, ACT_SCALE, l31_conv2_out, H3, W3);
            res_fn(l31_conv2_out, l30_out, ACT_SCALE, H3, W3, OC3, l31_out);
        };
        auto run_layer4 = [&](auto&& conv_fn, auto&& res_fn, const QTensorNHWC& in) {
            conv_fn(in,      p40_c1, c40_conv1, bn40_1, true,  ACT_SCALE, l40_mid, H4, W4);
            conv_fn(l40_mid, p40_c2, c40_conv2, bn40_2, false, ACT_SCALE, l40_conv2_out, H4, W4);
            conv_fn(in,      p40_ds, c40_ds,    bn40_ds,false, ACT_SCALE, l40_identity, H4, W4);
            res_fn(l40_conv2_out, l40_identity, ACT_SCALE, H4, W4, OC4, l40_out);
            conv_fn(l40_out, p41_c1, c41_conv1, bn41_1, true,  ACT_SCALE, l41_mid, H4, W4);
            conv_fn(l41_mid, p41_c2, c41_conv2, bn41_2, false, ACT_SCALE, l41_conv2_out, H4, W4);
            res_fn(l41_conv2_out, l40_out, ACT_SCALE, H4, W4, OC4, l41_out);
        };
        auto run_full = [&](auto&& conv_fn, auto&& res_fn) {
            quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), H_IN, W_IN, IC_C1, input_q8);
            q2q_conv1_im2col_bn_relu_requant(input_q8, p_c1, conv1, bn1, ACT_SCALE, conv1_out, OH1, OW1);
            q8_maxpool_3x3_s2_p1(conv1_out, OHp, OWp, pool_out);
            run_layer1(conv_fn, res_fn, pool_out);
            run_layer2(conv_fn, res_fn, l11_out);
            run_layer3(conv_fn, res_fn, l21_out);
            run_layer4(conv_fn, res_fn, l31_out);
            dequantize_q8_nhwc_to_fp32_nchw(l41_out, H4, W4, OC4, dequant_out);
            adaptive_avgpool_1x1_into(dequant_out, avgpool_out);
            linear_into(avgpool_out, fc, fc_out);
        };

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        // ---- Phase A: baseline dense-conv/residual kernels + optimized conv1.
        // (conv1 is now the fast-gather+SIMD kernel in BOTH phases -- it's a
        // strict win, so there is no reason to keep a slower copy around just
        // to replicate Phase 10b's exact number; this isolates items 3+4+5's
        // contribution on a like-for-like conv1.) ----
        for (int i = 0; i < N_WARMUP; ++i) run_full(q2q_conv_bn_requant, q2q_residual_add_relu);
        std::vector<double> t_baseline(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            run_full(q2q_conv_bn_requant, q2q_residual_add_relu);
            t_baseline[i] = now_ms() - t0;
        }
        volatile float sink_a = fc_out.data[0];
        (void)sink_a;

        // ---- Phase B: optimized kernels (items 3+4+5), with per-stage breakdown ----
        for (int i = 0; i < N_WARMUP; ++i) run_full(q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2);
        std::vector<double> t_quant(N_TIMED), t_conv1(N_TIMED), t_maxpool(N_TIMED),
                            t_layer1(N_TIMED), t_layer2(N_TIMED), t_layer3(N_TIMED), t_layer4(N_TIMED),
                            t_dequant(N_TIMED), t_avgpool_fc(N_TIMED), t_optimized(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), H_IN, W_IN, IC_C1, input_q8);
            const double t1 = now_ms();
            q2q_conv1_im2col_bn_relu_requant(input_q8, p_c1, conv1, bn1, ACT_SCALE, conv1_out, OH1, OW1);
            const double t2 = now_ms();
            q8_maxpool_3x3_s2_p1(conv1_out, OHp, OWp, pool_out);
            const double t3 = now_ms();
            run_layer1(q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, pool_out);
            const double t4 = now_ms();
            run_layer2(q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, l11_out);
            const double t5 = now_ms();
            run_layer3(q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, l21_out);
            const double t6 = now_ms();
            run_layer4(q2q_conv_bn_requant_apanel, q2q_residual_add_relu_avx2, l31_out);
            const double t7 = now_ms();
            dequantize_q8_nhwc_to_fp32_nchw(l41_out, H4, W4, OC4, dequant_out);
            const double t8 = now_ms();
            adaptive_avgpool_1x1_into(dequant_out, avgpool_out);
            linear_into(avgpool_out, fc, fc_out);
            const double t9 = now_ms();

            t_quant[i] = t1 - t0; t_conv1[i] = t2 - t1; t_maxpool[i] = t3 - t2;
            t_layer1[i] = t4 - t3; t_layer2[i] = t5 - t4; t_layer3[i] = t6 - t5; t_layer4[i] = t7 - t6;
            t_dequant[i] = t8 - t7; t_avgpool_fc[i] = t9 - t8; t_optimized[i] = t9 - t0;
        }
        volatile float sink_b = fc_out.data[0] + dequant_out.data[0];
        (void)sink_b;

        std::cout << "\n=== Phase A: baseline dense-conv/residual + optimized conv1 (" << N_TIMED << " runs) ===\n";
        print_stats("baseline_total_ms (q2q_conv_bn_requant + q2q_residual_add_relu)", t_baseline, N_TIMED);

        std::cout << "\n=== Phase B: optimized-kernel full prefix (" << N_TIMED << " runs, items 3+4+5 wired in) ===\n";
        print_stats("quantize_ms     (unchanged)                     ", t_quant, N_TIMED);
        print_stats("conv1_ms        (fast-gather + simd epilogue)   ", t_conv1, N_TIMED);
        print_stats("maxpool_ms      (unchanged, q8-direct)          ", t_maxpool, N_TIMED);
        print_stats("layer1_ms       (apanel conv + avx2 residual)   ", t_layer1, N_TIMED);
        print_stats("layer2_ms       (apanel conv + avx2 residual)   ", t_layer2, N_TIMED);
        print_stats("layer3_ms       (apanel conv + avx2 residual)   ", t_layer3, N_TIMED);
        print_stats("layer4_ms       (apanel conv + avx2 residual)   ", t_layer4, N_TIMED);
        print_stats("dequant_ms      (unchanged)                     ", t_dequant, N_TIMED);
        print_stats("avgpool_fc_ms   (unchanged, basic_engine ops)   ", t_avgpool_fc, N_TIMED);
        print_stats("optimized_total_ms                              ", t_optimized, N_TIMED);

        std::vector<double> sb = t_baseline, so = t_optimized;
        std::sort(sb.begin(), sb.end());
        std::sort(so.begin(), so.end());
        const double med_baseline  = sb[N_TIMED / 2];
        const double med_optimized = so[N_TIMED / 2];

        std::cout << "\n=== gate: optimized full prefix vs baseline full prefix vs basic_engine ===\n";
        std::cout << "  basic_engine full model (measured live, med) : " << basic_full_ms << " ms\n";
        std::cout << "  baseline full prefix    (med, Phase 10b kernels): " << med_baseline << " ms\n";
        std::cout << "  optimized full prefix   (med, items 3+4+5)      : " << med_optimized << " ms\n";
        std::cout << "  items 3+4+5 speedup over Phase 10b prefix       : "
                  << (med_baseline / med_optimized) << "x\n";
        std::cout << "  optimized prefix speedup over basic_engine      : "
                  << (basic_full_ms / med_optimized) << "x\n";

        if (med_optimized < med_baseline) {
            std::cout << "  RESULT: items 3+4+5 IMPROVE the full prefix -- " << med_optimized
                      << " ms vs " << med_baseline << " ms (" << (med_baseline / med_optimized)
                      << "x). The microkernel-tuning reversal paid off end-to-end, not just standalone.\n";
        } else {
            std::cout << "  RESULT: items 3+4+5 do NOT improve the full prefix end-to-end ("
                      << med_optimized << " ms vs " << med_baseline << " ms) despite standalone wins "
                         "-- some interaction (cache pressure, dispatch overhead, generic-lambda "
                         "codegen) erases the microkernel-level gains at full-pipeline scale.\n";
        }
        if (med_optimized < basic_full_ms) {
            std::cout << "  (optimized prefix also beats basic_engine: " << med_optimized << " ms < "
                      << basic_full_ms << " ms, " << (basic_full_ms / med_optimized) << "x)\n";
        } else {
            std::cout << "  (optimized prefix does NOT beat basic_engine: " << med_optimized
                      << " ms >= " << basic_full_ms << " ms)\n";
        }

        std::cout << "\nC++ engine latency\n";
        std::cout << "  runs:      " << N_TIMED << "\n";
        std::cout << "  min_ms:    " << so.front() << "\n";
        std::cout << "  median_ms: " << med_optimized << "\n";
        std::cout << "  mean_ms:   " << (std::accumulate(so.begin(), so.end(), 0.0) / N_TIMED) << "\n";
        std::cout << "  p90_ms:    " << so[N_TIMED * 9 / 10] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
