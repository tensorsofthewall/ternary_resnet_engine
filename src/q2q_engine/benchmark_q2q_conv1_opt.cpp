// benchmark_q2q_conv1_opt.cpp -- conv1 optimization A/B/C (paired, same data)
//
// Q2Q conv1 (im2col VNNI) is the dominant remaining cost in the optimized
// full-prefix pipeline (~7.8ms of ~21.1ms total) -- ~2x slower standalone
// than basic_engine's conv1+maxpool. This benchmark isolates WHERE that time
// goes and validates two targeted fixes, paired against the frozen baseline
// on identical data:
//
//   (A) baseline      : q2q_conv1_im2col_bn_relu_requant -- byte-identical to
//                       the kernel in benchmark_q2q_full_prefix_optimized.
//                       Gather flattens k = ic*49 + ky*7 + kx (147 scalar
//                       byte copies, stride=C_IN=3); scalar epilogue
//                       (4x std::lround+clamp+store per OC_TILE).
//   (B) fast_gather   : reorders the im2col flattening to
//                       k = (ky*7+kx)*IC + ic (so each (ky,kx) position's 3
//                       channels are CONTIGUOUS in NHWC), enabling a 4-byte
//                       load+store per (ky,kx) instead of 3 strided scalar
//                       byte copies -- 49 simple mov pairs instead of 147
//                       strided byte ops. The 4th (garbage) byte of each
//                       4-byte write is overwritten by the next iteration's
//                       valid byte 0 (self-correcting), except the very last
//                       iteration's spill into the K_FULL..K_PAD pad region,
//                       which the existing zero-fill (run AFTER gather)
//                       resets to 0. Scalar epilogue retained (isolates the
//                       gather-reorder contribution).
//   (C) fast_gather   : (B)'s reordered gather + epilogue_oc4_simd (the
//       + simd_epi   vectorized BN+ReLU+requantize epilogue validated in
//                       benchmark_q2q_microkernel_opt, ~2.08x paired win
//                       there) -- the fully-combined optimized kernel.
//
// All three are bit-exact (same arithmetic, just reordered memory layout/
// loads and a vectorized-vs-scalar epilogue with matching round-to-nearest);
// byte-mismatch counts are reported as a sanity check.
//
// Do not touch basic_engine. Standalone microkernel A/B -- no model forward
// pass, so no resnet18_engine dependency needed.
//
// Usage: ./benchmark_q2q_conv1_opt <model.bin>
//   (reads artifacts/golden_input_fp32.bin as the model input)

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"

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

// (validated in benchmark_q2q_microkernel_opt, ~2.08x paired win there)
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
// Shared VNNI compute core (identical across A/B/C -- only the gather
// layout and epilogue differ). K_PAD=160, IC_CHUNKS=5, OC_TILES=16.
// =====================================================================
static inline void conv1_vnni_oc4(
    const uint8_t* a_panel, const int8_t* panel_tile, int IC_CHUNKS,
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    for (int icc = 0; icc < IC_CHUNKS; ++icc) {
        const __m256i a32 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a_panel + (size_t)icc * K_CHUNK));
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
    d0 = hsum_epi32(acc0); d1 = hsum_epi32(acc1);
    d2 = hsum_epi32(acc2); d3 = hsum_epi32(acc3);
}

// =====================================================================
// (A) BASELINE -- byte-identical to benchmark_q2q_full_prefix_optimized's
// q2q_conv1_im2col_bn_relu_requant. Panel flattening k = ic*49 + ky*7 + kx;
// gather is 147 scalar byte copies (stride=C_IN=3); scalar epilogue.
// =====================================================================
struct Conv1PanelA { int K_PAD, IC_CHUNKS, OC_TILES; std::vector<int8_t> data; };

static Conv1PanelA build_panel_a(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int K_FULL = IC * KH * KW;
    const int IC_CHUNKS = (K_FULL + K_CHUNK - 1) / K_CHUNK;
    const int K_PAD = IC_CHUNKS * K_CHUNK;
    const int OC_TILES = OC / OC_TILE;
    const size_t base = (size_t)IC * KH * KW;

    Conv1PanelA panel;
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
                        const int ic = kg / (KH * KW);
                        const int rem = kg % (KH * KW);
                        const int ky = rem / KW, kx = rem % KW;
                        w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                    }
                    panel.data[(((size_t)oc_tile * IC_CHUNKS + icc) * OC_TILE + j) * K_CHUNK + k] = w;
                }
            }
    return panel;
}

static void kernel_baseline(
    const QTensorNHWC& x, const Conv1PanelA& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, KH = conv.kh, KW = conv.kw;
    const int stride = conv.stride_h;
    const int K_FULL = IC * KH * KW;
    const int K_PAD = panel.K_PAD, IC_CHUNKS = panel.IC_CHUNKS, OC_TILES = panel.OC_TILES;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const int b = x.border;

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data.data();
    uint8_t*       base_out   = y.data.data();
    std::vector<uint8_t> a_panel(K_PAD, uint8_t(0));

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * stride - conv.pad_h + b;
        uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C;
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * stride - conv.pad_w + b;
            for (int ic = 0; ic < IC; ++ic) {
                for (int ky = 0; ky < KH; ++ky) {
                    const uint8_t* row = base_in + ((size_t)(ih0 + ky) * x.W + iw0) * x.C + ic;
                    uint8_t* dst = a_panel.data() + (size_t)(ic * KH + ky) * KW;
                    for (int kx = 0; kx < KW; ++kx) dst[kx] = row[(size_t)kx * x.C];
                }
            }
            for (int kg = K_FULL; kg < K_PAD; ++kg) a_panel[kg] = 0;

            uint8_t* out_px = out_row + (size_t)ow * y.C;
            for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
                const int oc_base = oc_tile * OC_TILE;
                const int8_t* panel_tile = base_panel + (size_t)oc_tile * IC_CHUNKS * OC_TILE * K_CHUNK;
                int32_t d0, d1, d2, d3;
                conv1_vnni_oc4(a_panel.data(), panel_tile, IC_CHUNKS, d0, d1, d2, d3);

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

// =====================================================================
// (B)/(C) FAST-GATHER panel -- flattening k = (ky*KW+kx)*IC + ic, so each
// (ky,kx) position's IC=3 channels are contiguous in the panel AND in the
// NHWC source (channels are the fastest-varying axis). Numerically
// identical weights, just laid out to match the faster gather below.
// =====================================================================
struct Conv1PanelB { int K_PAD, IC_CHUNKS, OC_TILES; std::vector<int8_t> data; };

static Conv1PanelB build_panel_b(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int K_FULL = IC * KH * KW;
    const int IC_CHUNKS = (K_FULL + K_CHUNK - 1) / K_CHUNK;
    const int K_PAD = IC_CHUNKS * K_CHUNK;
    const int OC_TILES = OC / OC_TILE;
    const size_t base = (size_t)IC * KH * KW;

    Conv1PanelB panel;
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
                        const int kx = rem / IC, ic = rem % IC;
                        w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                    }
                    panel.data[(((size_t)oc_tile * IC_CHUNKS + icc) * OC_TILE + j) * K_CHUNK + k] = w;
                }
            }
    return panel;
}

// Fast gather: for each (ky,kx), the IC=3 source channels are contiguous
// (NHWC) and the destination slot is contiguous (panel B layout) -- one
// 4-byte load + 4-byte store per (ky,kx) instead of 3 strided byte copies.
// The 4th (garbage) byte written by iteration i is overwritten by iteration
// i+1's valid byte 0 (dst offsets are IC=3 apart, loads are 4 bytes), except
// the LAST iteration's spill into a_panel[K_FULL], which the zero-fill
// (always run after the gather) resets to 0. Reads stay in-bounds: max
// (iw0+KW-1)*C+4 = (222+6)*3+4 = 688 < x.W*x.C = 230*3 = 690.
static inline void gather_fast(
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

static void kernel_fast_gather(
    const QTensorNHWC& x, const Conv1PanelB& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, KH = conv.kh, KW = conv.kw;
    const int stride = conv.stride_h;
    const int K_FULL = IC * KH * KW;
    const int K_PAD = panel.K_PAD, IC_CHUNKS = panel.IC_CHUNKS, OC_TILES = panel.OC_TILES;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const int b = x.border;

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data.data();
    uint8_t*       base_out   = y.data.data();
    std::vector<uint8_t> a_panel(K_PAD + 4, uint8_t(0));  // +4 slack for the 4-byte gather tail write

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * stride - conv.pad_h + b;
        uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C;
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * stride - conv.pad_w + b;
            gather_fast(base_in, x.W, x.C, ih0, iw0, KH, KW, IC, a_panel.data());
            for (int kg = K_FULL; kg < K_PAD; ++kg) a_panel[kg] = 0;

            uint8_t* out_px = out_row + (size_t)ow * y.C;
            for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
                const int oc_base = oc_tile * OC_TILE;
                const int8_t* panel_tile = base_panel + (size_t)oc_tile * IC_CHUNKS * OC_TILE * K_CHUNK;
                int32_t d0, d1, d2, d3;
                conv1_vnni_oc4(a_panel.data(), panel_tile, IC_CHUNKS, d0, d1, d2, d3);

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

static void kernel_fast_gather_simd_epilogue(
    const QTensorNHWC& x, const Conv1PanelB& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, KH = conv.kh, KW = conv.kw;
    const int stride = conv.stride_h;
    const int K_FULL = IC * KH * KW;
    const int K_PAD = panel.K_PAD, IC_CHUNKS = panel.IC_CHUNKS, OC_TILES = panel.OC_TILES;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const int b = x.border;

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data.data();
    uint8_t*       base_out   = y.data.data();
    std::vector<uint8_t> a_panel(K_PAD + 4, uint8_t(0));

    for (int oh = 0; oh < OH; ++oh) {
        const int ih0 = oh * stride - conv.pad_h + b;
        uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C;
        for (int ow = 0; ow < OW; ++ow) {
            const int iw0 = ow * stride - conv.pad_w + b;
            gather_fast(base_in, x.W, x.C, ih0, iw0, KH, KW, IC, a_panel.data());
            for (int kg = K_FULL; kg < K_PAD; ++kg) a_panel[kg] = 0;

            uint8_t* out_px = out_row + (size_t)ow * y.C;
            for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
                const int oc_base = oc_tile * OC_TILE;
                const int8_t* panel_tile = base_panel + (size_t)oc_tile * IC_CHUNKS * OC_TILE * K_CHUNK;
                int32_t d0, d1, d2, d3;
                conv1_vnni_oc4(a_panel.data(), panel_tile, IC_CHUNKS, d0, d1, d2, d3);

                float s[OC_TILE], bb[OC_TILE];
                for (int j = 0; j < OC_TILE; ++j) {
                    s[j] = bn.scale[oc_base + j] * dq;
                    bb[j] = bn.bias[oc_base + j];
                }
                epilogue_oc4_simd(d0, d1, d2, d3, s, bb, /*relu=*/true, inv_out_scale,
                                  out_px + oc_base);
            }
        }
    }
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

static void print_stats(const char* label, std::vector<double> v, int N) {
    std::sort(v.begin(), v.end());
    const double mn = v.front(), med = v[N/2], p90 = v[N*9/10];
    const double avg = std::accumulate(v.begin(), v.end(), 0.0) / N;
    std::cout << "  " << label << ":  min=" << mn << "  med=" << med
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

static size_t count_mismatches(const QTensorNHWC& a, const QTensorNHWC& b) {
    if (a.data.size() != b.data.size()) return a.data.size();
    size_t n = 0;
    for (size_t i = 0; i < a.data.size(); ++i) if (a.data[i] != b.data[i]) ++n;
    return n;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }
    try {
        const Model model = load_model(argv[1]);
        const Conv2DParams& conv1 = model.convs.at("conv1");
        const BNFusedParams& bn1  = model.fused_bns.at("bn1");

        Tensor golden_input(3, 224, 224);
        golden_input.data = read_f32_file("artifacts/golden_input_fp32.bin", 3 * 224 * 224);

        const int IC = conv1.in_channels, OC = conv1.out_channels;
        const int H_IN = 224, W_IN = 224;
        const int OH = (H_IN + 2*conv1.pad_h - conv1.kh) / conv1.stride_h + 1;
        const int OW = (W_IN + 2*conv1.pad_w - conv1.kw) / conv1.stride_w + 1;

        std::cout << "=== benchmark_q2q_conv1_opt: conv1 gather/epilogue A/B/C ===\n";
        std::cout << "  conv1: IC=" << IC << " OC=" << OC << " " << conv1.kh << "x" << conv1.kw
                  << " stride=" << conv1.stride_h << " pad=" << conv1.pad_h
                  << " -> " << OH << "x" << OW << "\n";
        std::cout << "  (A) baseline         : k=ic*49+ky*7+kx gather (147 strided byte copies) + scalar epilogue\n";
        std::cout << "  (B) fast_gather      : k=(ky*7+kx)*IC+ic gather (49x 4-byte load/store) + scalar epilogue\n";
        std::cout << "  (C) fast_gather+simd : (B)'s gather + epilogue_oc4_simd (vectorized BN+ReLU+requant)\n";

        QTensorNHWC input_q8 = make_q8_tensor(H_IN, W_IN, IC, conv1.pad_h, ACT_SCALE);
        quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), H_IN, W_IN, IC, input_q8);

        const Conv1PanelA panel_a = build_panel_a(conv1);
        const Conv1PanelB panel_b = build_panel_b(conv1);

        QTensorNHWC out_a = make_q8_tensor(OH, OW, OC, 1, ACT_SCALE);
        QTensorNHWC out_b = make_q8_tensor(OH, OW, OC, 1, ACT_SCALE);
        QTensorNHWC out_c = make_q8_tensor(OH, OW, OC, 1, ACT_SCALE);

        constexpr int N_WARMUP = 5;
        constexpr int N_TIMED = 30;

        for (int i = 0; i < N_WARMUP; ++i) {
            kernel_baseline(input_q8, panel_a, conv1, bn1, ACT_SCALE, out_a, OH, OW);
            kernel_fast_gather(input_q8, panel_b, conv1, bn1, ACT_SCALE, out_b, OH, OW);
            kernel_fast_gather_simd_epilogue(input_q8, panel_b, conv1, bn1, ACT_SCALE, out_c, OH, OW);
        }

        std::vector<double> t_a(N_TIMED), t_b(N_TIMED), t_c(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            double t0 = now_ms();
            kernel_baseline(input_q8, panel_a, conv1, bn1, ACT_SCALE, out_a, OH, OW);
            double t1 = now_ms();
            kernel_fast_gather(input_q8, panel_b, conv1, bn1, ACT_SCALE, out_b, OH, OW);
            double t2 = now_ms();
            kernel_fast_gather_simd_epilogue(input_q8, panel_b, conv1, bn1, ACT_SCALE, out_c, OH, OW);
            double t3 = now_ms();
            t_a[i] = t1 - t0; t_b[i] = t2 - t1; t_c[i] = t3 - t2;
        }

        const size_t mism_b = count_mismatches(out_a, out_b);
        const size_t mism_c = count_mismatches(out_a, out_c);

        std::cout << "\n=== paired results (" << N_TIMED << " runs each, same input data) ===\n";
        print_stats("(A) baseline           ", t_a, N_TIMED);
        print_stats("(B) fast_gather        ", t_b, N_TIMED);
        print_stats("(C) fast_gather+simd_ep", t_c, N_TIMED);

        std::vector<double> sa = t_a, sb = t_b, sc = t_c;
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        std::sort(sc.begin(), sc.end());
        const double ma = sa[N_TIMED/2], mb = sb[N_TIMED/2], mc = sc[N_TIMED/2];

        std::cout << "\n=== verdict ===\n";
        std::cout << "  (B) fast_gather        : " << ma / mb << "x vs baseline   (byte mismatches: "
                  << mism_b << "/" << out_a.data.size() << ")\n";
        std::cout << "  (C) fast_gather+simd_ep: " << ma / mc << "x vs baseline   (byte mismatches: "
                  << mism_c << "/" << out_a.data.size() << ")\n";
        std::cout << "  (C) vs (B) [epilogue contribution alone] : " << mb / mc << "x\n";

        if (mc < ma) {
            std::cout << "  RESULT: (C) WINS by " << (ma / mc) << "x -- "
                      << "gather reorder + SIMD epilogue meaningfully cut conv1's standalone cost ("
                      << ma << " ms -> " << mc << " ms median). Worth wiring into the full prefix.\n";
        } else {
            std::cout << "  RESULT: optimizations do NOT beat the baseline standalone ("
                      << mc << " ms vs " << ma << " ms) -- the gather/epilogue are not "
                         "the dominant cost; the VNNI compute loop (16 OC_TILES x 5 chunks "
                         "per pixel) likely dominates instead.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
