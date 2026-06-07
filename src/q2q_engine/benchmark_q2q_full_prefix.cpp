// benchmark_q2q_full_prefix.cpp — Q2Q Phase 10b: full Q2Q-ish forward prefix
//
//   FP32 input -> quantize -> conv1 (Q2Q im2col VNNI) -> maxpool (q8 direct)
//     -> layer1 (Q2Q) -> layer2-4 (Q2Q) -> dequant -> basic_engine avgpool+fc
//
// The first "all-Q2Q-trunk" benchmark: every conv from conv1 through layer4
// runs in q8/VNNI end to end -- no FP32 mid-stack, no basic_engine conv
// anywhere in the trunk. Only the input quantization (FP32 -> q8, a one-time
// boundary op) and the final avgpool+fc (trivial ops with no Q2Q replacement)
// touch basic_engine, and only through its public, read-only ops API -- the
// same pattern used by benchmark_q2q_hybrid_trunk.
//
// Structural note -- generalized border: every earlier stack benchmark fixed
// QTensorNHWC's halo at a compile-time PAD=1 (sufficient for the 3x3/pad<=1
// convs that make up layer1-4). conv1 is a 7x7/stride=2/pad=3 conv, which
// needs a wider halo on its INPUT only. So QTensorNHWC now carries a per-
// tensor `border` field, and q2q_conv_bn_requant / q2q_residual_add_relu index
// through `x.border`/`y.border` instead of the fixed constant. Every tensor
// downstream of conv1 is still allocated with border=PAD_Q8=1 -- identical to
// every earlier benchmark's convention -- so this generalization changes
// nothing numerically for layer1-4; only the initial quantized image (which
// conv1 reads via its own dedicated im2col kernel, not q2q_conv_bn_requant)
// uses border=conv1.pad_h=3.
//
// Kernel provenance (reused, not retuned):
//   - conv1 im2col panel + kernel, q8 maxpool: byte-for-byte from
//     benchmark_q2q_conv1_maxpool.cpp (Phase 10a)
//   - q2q_conv_bn_requant / vnni_oc4_dots / q2q_residual_add_relu /
//     build_oc4_k32_panel: byte-for-byte from benchmark_q2q_layer234_stack.cpp,
//     modulo the PAD -> {x,y}.border generalization described above (a no-op
//     for every border=1 tensor, i.e. everything but the input image)
//   - layer1/layer2/layer3/layer4 orchestration: byte-for-byte from
//     benchmark_q2q_layer1_stack.cpp / benchmark_q2q_hybrid_trunk.cpp
//
// Do not touch basic_engine.
//
// Usage: ./benchmark_q2q_full_prefix <model.bin>
//   (reads artifacts/golden_input_fp32.bin as the model input)

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"
#include "resnet18_engine.hpp"
#include "ops.hpp"

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

// Standard halo for every q8 tensor downstream of conv1 -- identical to the
// fixed PAD=1 used throughout every earlier layer1/layer2-4 stack benchmark.
constexpr int PAD_Q8 = 1;

constexpr float ACT_SCALE = 0.05f;

// ---- Quantized NHWC activation tensor, now with a per-tensor border (see
// header comment for why this generalization is needed and why it is a no-op
// for every tensor except the initial quantized image). ----
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
// tensor's interior (border stays zero-filled). Border-generic.
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

// q8 NHWC -> FP32 NCHW [C][H][W] -- the exit boundary back into basic_engine's
// Tensor layout for avgpool+fc. Border-generic.
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

// =====================================================================
// conv1: im2col "A-panel" + VNNI -- byte-for-byte from
// benchmark_q2q_conv1_maxpool.cpp (Phase 10a). IC=3 breaks the
// IC % K_CHUNK == 0 assumption behind every dense Q2Q panel, so the
// receptive field (IC*KH*KW = 147, padded to K_PAD=160 = 5*32) is
// flattened into a single K axis at pack time; the VNNI core then just
// streams 5 flat 32-wide dpbusd chunks per OC_TILE (no ky/kx loop).
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
        const int ih0 = oh * stride - conv.pad_h + b; // == oh*stride when b == conv.pad_h
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

// q8 maxpool 3x3/stride=2/pad=1, direct on uint8 -- byte-for-byte from
// benchmark_q2q_conv1_maxpool.cpp. One static ACT_SCALE everywhere makes
// max(uint8) track max(fp32) under the shared monotonic affine map.
static void q8_maxpool_3x3_s2_p1(const QTensorNHWC& x, int OH, int OW, QTensorNHWC& y) {
    constexpr int stride = 2, pad = 1;
    const int C = x.C;
    const int read_offset = x.border - pad;
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
}

// =====================================================================
// layer1-4 dense Q2Q kernels -- byte-for-byte from benchmark_q2q_layer234_stack.cpp,
// generalized PAD -> {x,y}.border (a no-op: every tensor here has border=PAD_Q8=1).
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
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

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
        const int OH1 = (H_IN + 2 * conv1.pad_h - conv1.kh) / conv1.stride_h + 1; // 112
        const int OW1 = (W_IN + 2 * conv1.pad_w - conv1.kw) / conv1.stride_w + 1; // 112
        const int OHp = (OH1 + 2 - 3) / 2 + 1; // maxpool: 56
        const int OWp = (OW1 + 2 - 3) / 2 + 1; // 56
        const int OC1 = c10_conv1.out_channels;  // 64  (layer1, identity shortcut)
        const int OC2 = c20_conv1.out_channels;  // 128 (layer2 -> 28x28)
        const int OC3 = c30_conv1.out_channels;  // 256 (layer3 -> 14x14)
        const int OC4 = c40_conv1.out_channels;  // 512 (layer4 -> 7x7)
        const int H1 = OHp,  W1 = OWp;            // 56
        const int H2 = 28,   W2 = 28;
        const int H3 = 14,   W3 = 14;
        const int H4 = 7,    W4 = 7;

        if (IC_C1 != 3 || OC_C1 % OC_TILE != 0)
            throw std::runtime_error("conv1 expected IC=3, OC % 4 == 0");
        if (OC1 % K_CHUNK != 0 || OC2 % OC_TILE != 0 || OC3 % OC_TILE != 0 || OC4 % OC_TILE != 0)
            throw std::runtime_error("channel counts must be divisible by 32 (IC) / 4 (OC)");

        std::cout << "=== benchmark_q2q_full_prefix: FP32 -> quantize -> conv1(Q2Q) -> "
                     "maxpool(q8) -> layer1(Q2Q) -> layer2-4(Q2Q) -> dequant -> avgpool+fc ===\n";
        std::cout << "  conv1   : 3x224x224 -> " << OC_C1 << "x" << OH1 << "x" << OW1
                  << "  (k=" << conv1.kh << "x" << conv1.kw << " s=" << conv1.stride_h << " p=" << conv1.pad_h
                  << ", im2col VNNI)\n";
        std::cout << "  maxpool : -> " << OC_C1 << "x" << OHp << "x" << OWp << "  (k=3x3 s=2 p=1, q8-direct)\n";
        std::cout << "  layer1  : -> " << OC1 << "x" << H1 << "x" << W1 << "  (identity shortcut x2, Q2Q)\n";
        std::cout << "  layer2  : -> " << OC2 << "x" << H2 << "x" << W2 << "  (downsample + identity, Q2Q)\n";
        std::cout << "  layer3  : -> " << OC3 << "x" << H3 << "x" << W3 << "  (downsample + identity, Q2Q)\n";
        std::cout << "  layer4  : -> " << OC4 << "x" << H4 << "x" << W4 << "  (downsample + identity, Q2Q)\n";
        std::cout << "  tail    : dequant -> adaptive_avgpool_1x1 -> fc (-> 1000)  [basic_engine, read-only]\n";
        std::cout << "  ACT_SCALE (static, all q8 tensors): " << ACT_SCALE << "\n";

        // ---- live basic_engine full-model baseline (gate) ----
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
        std::vector<double> s_basic_full = t_basic_full;
        std::sort(s_basic_full.begin(), s_basic_full.end());
        const double basic_full_ms = s_basic_full[N_TIMED_BASIC / 2];

        std::cout << "\n=== basic_engine full-model cost (measured live, workspace-timed path, "
                  << N_TIMED_BASIC << " runs) ===\n";
        print_stats("basic_full_ms", t_basic_full, N_TIMED_BASIC);

        // ---- build all panels up front ----
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

        std::cout << "\n  conv1 im2col panel: K_full=" << (IC_C1 * conv1.kh * conv1.kw)
                  << "  K_pad=" << p_c1.K_PAD << "  IC_CHUNKS=" << p_c1.IC_CHUNKS
                  << "  OC_TILES=" << p_c1.OC_TILES << "\n";

        // ---- scratch tensors: input image needs border=conv1.pad_h (wide halo
        // for the 7x7 receptive field); every tensor downstream uses border=PAD_Q8=1 ----
        QTensorNHWC input_q8   = make_q8_tensor(H_IN, W_IN, IC_C1, conv1.pad_h, ACT_SCALE);
        QTensorNHWC conv1_out  = make_q8_tensor(OH1, OW1, OC_C1, PAD_Q8, ACT_SCALE);
        QTensorNHWC pool_out   = make_q8_tensor(OHp, OWp, OC_C1, PAD_Q8, ACT_SCALE);

        QTensorNHWC l10_mid       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l10_conv2_out = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l10_out       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l11_mid       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l11_conv2_out = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);
        QTensorNHWC l11_out       = make_q8_tensor(H1, W1, OC1, PAD_Q8, ACT_SCALE);  // -> layer2 input

        QTensorNHWC l20_mid       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_conv2_out = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_identity  = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l20_out       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l21_mid       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l21_conv2_out = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);
        QTensorNHWC l21_out       = make_q8_tensor(H2, W2, OC2, PAD_Q8, ACT_SCALE);  // -> layer3 input

        QTensorNHWC l30_mid       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_conv2_out = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_identity  = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l30_out       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l31_mid       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l31_conv2_out = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);
        QTensorNHWC l31_out       = make_q8_tensor(H3, W3, OC3, PAD_Q8, ACT_SCALE);  // -> layer4 input

        QTensorNHWC l40_mid       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_conv2_out = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_identity  = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l40_out       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l41_mid       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l41_conv2_out = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);
        QTensorNHWC l41_out       = make_q8_tensor(H4, W4, OC4, PAD_Q8, ACT_SCALE);  // Q2Q trunk output

        Tensor dequant_out(OC4, H4, W4);
        Tensor avgpool_out(OC4, 1, 1);
        Tensor fc_out(1000, 1, 1);

        auto run_layer1 = [&](const QTensorNHWC& in) {
            q2q_conv_bn_requant(in,      p10_c1, c10_conv1, bn10_1, true,  ACT_SCALE, l10_mid, H1, W1);
            q2q_conv_bn_requant(l10_mid, p10_c2, c10_conv2, bn10_2, false, ACT_SCALE, l10_conv2_out, H1, W1);
            q2q_residual_add_relu(l10_conv2_out, in, ACT_SCALE, H1, W1, OC1, l10_out);
            q2q_conv_bn_requant(l10_out, p11_c1, c11_conv1, bn11_1, true,  ACT_SCALE, l11_mid, H1, W1);
            q2q_conv_bn_requant(l11_mid, p11_c2, c11_conv2, bn11_2, false, ACT_SCALE, l11_conv2_out, H1, W1);
            q2q_residual_add_relu(l11_conv2_out, l10_out, ACT_SCALE, H1, W1, OC1, l11_out);
        };
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
        auto run_layer4 = [&](const QTensorNHWC& in) {
            q2q_conv_bn_requant(in,      p40_c1, c40_conv1, bn40_1, true,  ACT_SCALE, l40_mid, H4, W4);
            q2q_conv_bn_requant(l40_mid, p40_c2, c40_conv2, bn40_2, false, ACT_SCALE, l40_conv2_out, H4, W4);
            q2q_conv_bn_requant(in,      p40_ds, c40_ds,    bn40_ds,false, ACT_SCALE, l40_identity, H4, W4);
            q2q_residual_add_relu(l40_conv2_out, l40_identity, ACT_SCALE, H4, W4, OC4, l40_out);
            q2q_conv_bn_requant(l40_out, p41_c1, c41_conv1, bn41_1, true,  ACT_SCALE, l41_mid, H4, W4);
            q2q_conv_bn_requant(l41_mid, p41_c2, c41_conv2, bn41_2, false, ACT_SCALE, l41_conv2_out, H4, W4);
            q2q_residual_add_relu(l41_conv2_out, l40_out, ACT_SCALE, H4, W4, OC4, l41_out);
        };

        auto run_full_q2q = [&]() {
            quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), H_IN, W_IN, IC_C1, input_q8);
            q2q_conv1_im2col_bn_relu_requant(input_q8, p_c1, conv1, bn1, ACT_SCALE, conv1_out, OH1, OW1);
            q8_maxpool_3x3_s2_p1(conv1_out, OHp, OWp, pool_out);
            run_layer1(pool_out);
            run_layer2(l11_out);
            run_layer3(l21_out);
            run_layer4(l31_out);
            dequantize_q8_nhwc_to_fp32_nchw(l41_out, H4, W4, OC4, dequant_out);
            adaptive_avgpool_1x1_into(dequant_out, avgpool_out);
            linear_into(avgpool_out, fc, fc_out);
        };

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;
        for (int i = 0; i < N_WARMUP; ++i) run_full_q2q();

        std::vector<double> t_quant(N_TIMED), t_conv1(N_TIMED), t_maxpool(N_TIMED),
                            t_layer1(N_TIMED), t_layer2(N_TIMED), t_layer3(N_TIMED), t_layer4(N_TIMED),
                            t_dequant(N_TIMED), t_avgpool_fc(N_TIMED), t_total(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            quantize_nchw_fp32_to_q8_nhwc(golden_input.data.data(), H_IN, W_IN, IC_C1, input_q8);
            const double t1 = now_ms();
            q2q_conv1_im2col_bn_relu_requant(input_q8, p_c1, conv1, bn1, ACT_SCALE, conv1_out, OH1, OW1);
            const double t2 = now_ms();
            q8_maxpool_3x3_s2_p1(conv1_out, OHp, OWp, pool_out);
            const double t3 = now_ms();
            run_layer1(pool_out);
            const double t4 = now_ms();
            run_layer2(l11_out);
            const double t5 = now_ms();
            run_layer3(l21_out);
            const double t6 = now_ms();
            run_layer4(l31_out);
            const double t7 = now_ms();
            dequantize_q8_nhwc_to_fp32_nchw(l41_out, H4, W4, OC4, dequant_out);
            const double t8 = now_ms();
            adaptive_avgpool_1x1_into(dequant_out, avgpool_out);
            linear_into(avgpool_out, fc, fc_out);
            const double t9 = now_ms();

            t_quant[i]      = t1 - t0;
            t_conv1[i]      = t2 - t1;
            t_maxpool[i]    = t3 - t2;
            t_layer1[i]     = t4 - t3;
            t_layer2[i]     = t5 - t4;
            t_layer3[i]     = t6 - t5;
            t_layer4[i]     = t7 - t6;
            t_dequant[i]    = t8 - t7;
            t_avgpool_fc[i] = t9 - t8;
            t_total[i]      = t9 - t0;
        }
        volatile float sink = fc_out.data[0] + dequant_out.data[0];
        (void)sink;

        std::cout << "\n=== full Q2Q-ish prefix timing (FP32 in -> logits out, over " << N_TIMED << " runs) ===\n";
        print_stats("quantize_ms     (FP32 NCHW -> q8 NHWC)         ", t_quant, N_TIMED);
        print_stats("conv1_ms        (Q2Q im2col VNNI+BN+ReLU+req.) ", t_conv1, N_TIMED);
        print_stats("maxpool_ms      (q8-direct)                    ", t_maxpool, N_TIMED);
        print_stats("layer1_ms       (Q2Q, identity x2)             ", t_layer1, N_TIMED);
        print_stats("layer2_ms       (Q2Q, downsample + identity)   ", t_layer2, N_TIMED);
        print_stats("layer3_ms       (Q2Q, downsample + identity)   ", t_layer3, N_TIMED);
        print_stats("layer4_ms       (Q2Q, downsample + identity)   ", t_layer4, N_TIMED);
        print_stats("dequant_ms      (q8 NHWC -> FP32 NCHW)         ", t_dequant, N_TIMED);
        print_stats("avgpool_fc_ms   (basic_engine ops)             ", t_avgpool_fc, N_TIMED);
        print_stats("full_prefix_total_ms                           ", t_total, N_TIMED);

        std::vector<double> s_total = t_total;
        std::sort(s_total.begin(), s_total.end());
        const double med_total = s_total[N_TIMED / 2];

        std::cout << "\n=== gate: full Q2Q-ish prefix vs basic_engine full model ===\n";
        std::cout << "  basic_engine full model (measured live, med): " << basic_full_ms << " ms\n";
        std::cout << "  full Q2Q-ish prefix (med)                   : " << med_total << " ms\n";
        if (med_total < basic_full_ms) {
            std::cout << "  RESULT: full Q2Q-ish prefix (" << med_total << " ms) BEATS basic_engine ("
                      << basic_full_ms << " ms) by " << (basic_full_ms / med_total)
                      << "x -- an all-Q2Q trunk (conv1 through layer4) is a net win end-to-end.\n";
        } else {
            std::cout << "  RESULT: full Q2Q-ish prefix (" << med_total << " ms) does NOT beat basic_engine ("
                      << basic_full_ms << " ms) -- " << (med_total / basic_full_ms)
                      << "x slower. The conv1 im2col kernel's ~2x deficit (Phase 10a) and layer1's "
                         "marginal Q2Q result (Phase 8, ~1.03x) drag down the otherwise-winning "
                         "layer2-4 trunk; the hybrid split (basic conv1/maxpool/layer1 + Q2Q layer2-4, "
                         "Phase 9) remains the better measured strategy.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
