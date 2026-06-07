// benchmark_w16w32.cpp — runtime-only benchmark: wider-tile blocked sparse kernels
// Layout: [CB][H_PAD][WB][CI][WI_EXT], C_BLOCK=16
// Target: layer2.1.conv2 (IC=128, OC=128, H=28, W=28, K=3×3, S=1, P=1)
//
// Variant A: W_BLOCK=16, WI_EXT=18, OC_TILE=2, OW_TILE=16 (2 YMM lanes/OC)
// Variant B: W_BLOCK=32, WI_EXT=34, OC_TILE=2, OW_TILE=32 (4 YMM lanes/OC)
//
// Both: branch-free OC2 sparse pattern lists (pp/pn/np/nn/p0/n0/zp/zn).
// No dense weight branches, no sign_epi32. Benchmark-only; does not touch
// basic_engine or the existing blocked_engine forward pass.
//
// Usage: ./benchmark_w16w32 <model.bin>

#include "model_loader.hpp"
#include "bn_fusion.hpp"
#include "model.hpp"
#include "tensor.hpp"

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

static constexpr int C_BLOCK = 16;

// ---- Generic blocked layout [CB][H_PAD][WB][CI][WI_EXT] ----
struct Q8Blocked {
    int CB = 0, H = 0, WB = 0;
    int W_BLOCK = 0, WI_EXT = 0;
    float scale = 1.0f;
    std::vector<int8_t> data;
};

struct KAction {
    uint16_t ic;
    uint8_t  ky, kx;
};

struct OC2PairLists {
    std::vector<KAction> pp, pn, np, nn, p0, n0, zp, zn;
};

struct BlockedOC2Plan {
    std::vector<OC2PairLists> oc_pairs;
};

// ---- Quantize FP32 NCHW → Q8Blocked, parameterized by W_BLOCK/WI_EXT ----
static void convert_fp32_nchw_to_q8_blocked(
    const float* src, int C, int H, int W, int pad,
    int W_BLOCK, int WI_EXT, Q8Blocked& dst)
{
    const int H_PAD = H + 2 * pad;
    const int W_PAD = W + 2 * pad;
    const int CB    = C / C_BLOCK;
    const int WB    = (W_PAD + W_BLOCK - 1) / W_BLOCK;

    float max_abs = 0.0f;
    for (int i = 0; i < C * H * W; ++i)
        max_abs = std::max(max_abs, std::abs(src[i]));
    dst.scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    const float inv_scale = 1.0f / dst.scale;

    dst.CB = CB; dst.H = H_PAD; dst.WB = WB;
    dst.W_BLOCK = W_BLOCK; dst.WI_EXT = WI_EXT;
    const size_t total = (size_t)CB * H_PAD * WB * C_BLOCK * WI_EXT;
    dst.data.assign(total, int8_t(0));

    for (int c = 0; c < C; ++c) {
        const int cb = c / C_BLOCK;
        const int ci = c % C_BLOCK;
        for (int h_pad = 0; h_pad < H_PAD; ++h_pad) {
            const int h_src = h_pad - pad;
            if (h_src < 0 || h_src >= H) continue;
            for (int wb = 0; wb < WB; ++wb) {
                for (int wi = 0; wi < WI_EXT; ++wi) {
                    const int w_pad = wb * W_BLOCK + wi;
                    const int w_src = w_pad - pad;
                    if (w_src < 0 || w_src >= W) continue;
                    const float fval = src[(c * H + h_src) * W + w_src];
                    int32_t ival = static_cast<int32_t>(std::roundf(fval * inv_scale));
                    ival = std::max(-127, std::min(127, ival));
                    const size_t idx = ((size_t)(cb * H_PAD + h_pad) * WB + wb)
                                     * C_BLOCK * WI_EXT
                                     + ci * WI_EXT + wi;
                    dst.data[idx] = static_cast<int8_t>(ival);
                }
            }
        }
    }
}

static BlockedOC2Plan build_plan(const Conv2DParams& conv) {
    const int IC = conv.in_channels;
    const int OC = conv.out_channels;
    const int KH = conv.kh;
    const int KW = conv.kw;

    BlockedOC2Plan plan;
    plan.oc_pairs.resize(OC / 2);

    for (int oc_pair = 0; oc_pair < OC / 2; ++oc_pair) {
        const int oc0 = oc_pair * 2;
        const int oc1 = oc_pair * 2 + 1;
        OC2PairLists& pl = plan.oc_pairs[oc_pair];

        for (int ic = 0; ic < IC; ++ic) {
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    const size_t base = (size_t)IC * KH * KW;
                    const size_t off  = (size_t)ic * KH * KW + ky * KW + kx;
                    const int w0 = conv.weight[oc0 * base + off];
                    const int w1 = conv.weight[oc1 * base + off];
                    if (w0 == 0 && w1 == 0) continue;

                    const KAction a = {
                        static_cast<uint16_t>(ic),
                        static_cast<uint8_t>(ky),
                        static_cast<uint8_t>(kx)
                    };

                    if      (w0 > 0 && w1 > 0) pl.pp.push_back(a);
                    else if (w0 > 0 && w1 < 0) pl.pn.push_back(a);
                    else if (w0 < 0 && w1 > 0) pl.np.push_back(a);
                    else if (w0 < 0 && w1 < 0) pl.nn.push_back(a);
                    else if (w0 > 0)            pl.p0.push_back(a);
                    else if (w0 < 0)            pl.n0.push_back(a);
                    else if (w1 > 0)            pl.zp.push_back(a);
                    else                        pl.zn.push_back(a);
                }
            }
        }
    }
    return plan;
}

static inline __m256i ymm_lo(__m128i v) { return _mm256_cvtepi8_epi32(v); }
static inline __m256i ymm_hi(__m128i v) { return _mm256_cvtepi8_epi32(_mm_bsrli_si128(v, 8)); }

// ===========================================================================
// Variant A — W_BLOCK=16, WI_EXT=18, OC_TILE=2, OW_TILE=16 (2 YMM lanes/OC)
// ===========================================================================
static void kernel_variantA_oc2_ow16(
    const Q8Blocked& x, const BlockedOC2Plan& plan, const BNFusedParams& bn,
    float* y_out, int OC, int OH, int OWB, bool relu)
{
    const int H_PAD = x.H, WB_IN = x.WB, WI_EXT = x.WI_EXT, W_BLOCK = x.W_BLOCK;
    const int stride_wb = C_BLOCK * WI_EXT;
    const int stride_h  = WB_IN * stride_wb;
    const int stride_cb = H_PAD * stride_h;
    const float dq = x.scale;
    const int OW_ROUND = OWB * W_BLOCK;
    const int8_t* base = x.data.data();

    for (int oc_pair = 0; oc_pair < OC / 2; ++oc_pair) {
        const int oc0 = oc_pair * 2, oc1 = oc_pair * 2 + 1;
        const OC2PairLists& pl = plan.oc_pairs[oc_pair];
        const float s0 = bn.scale[oc0] * dq, b0 = bn.bias[oc0];
        const float s1 = bn.scale[oc1] * dq, b1 = bn.bias[oc1];

        for (int oh = 0; oh < OH; ++oh) {
            float* row0 = y_out + oc0 * OH * OW_ROUND + oh * OW_ROUND;
            float* row1 = y_out + oc1 * OH * OW_ROUND + oh * OW_ROUND;

            for (int owb = 0; owb < OWB; ++owb) {
                __m256i a0_lo = _mm256_setzero_si256(), a0_hi = _mm256_setzero_si256();
                __m256i a1_lo = _mm256_setzero_si256(), a1_hi = _mm256_setzero_si256();
                const int owb_off = owb * stride_wb;

#define LOAD16(act, lo, hi) \
    const int _iy##act  = oh + (act).ky; \
    const int _cb##act  = (act).ic >> 4; \
    const int _ci##act  = (act).ic & 0xF; \
    const int _idx##act = _cb##act*stride_cb + _iy##act*stride_h + owb_off \
                        + _ci##act*WI_EXT + (act).kx; \
    const __m128i _v##act = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + _idx##act)); \
    const __m256i lo = ymm_lo(_v##act); \
    const __m256i hi = ymm_hi(_v##act);

                for (const KAction& a : pl.pp) { LOAD16(a, lo, hi)
                    a0_lo=_mm256_add_epi32(a0_lo,lo); a0_hi=_mm256_add_epi32(a0_hi,hi);
                    a1_lo=_mm256_add_epi32(a1_lo,lo); a1_hi=_mm256_add_epi32(a1_hi,hi); }
                for (const KAction& a : pl.pn) { LOAD16(a, lo, hi)
                    a0_lo=_mm256_add_epi32(a0_lo,lo); a0_hi=_mm256_add_epi32(a0_hi,hi);
                    a1_lo=_mm256_sub_epi32(a1_lo,lo); a1_hi=_mm256_sub_epi32(a1_hi,hi); }
                for (const KAction& a : pl.np) { LOAD16(a, lo, hi)
                    a0_lo=_mm256_sub_epi32(a0_lo,lo); a0_hi=_mm256_sub_epi32(a0_hi,hi);
                    a1_lo=_mm256_add_epi32(a1_lo,lo); a1_hi=_mm256_add_epi32(a1_hi,hi); }
                for (const KAction& a : pl.nn) { LOAD16(a, lo, hi)
                    a0_lo=_mm256_sub_epi32(a0_lo,lo); a0_hi=_mm256_sub_epi32(a0_hi,hi);
                    a1_lo=_mm256_sub_epi32(a1_lo,lo); a1_hi=_mm256_sub_epi32(a1_hi,hi); }
                for (const KAction& a : pl.p0) { LOAD16(a, lo, hi)
                    a0_lo=_mm256_add_epi32(a0_lo,lo); a0_hi=_mm256_add_epi32(a0_hi,hi); }
                for (const KAction& a : pl.n0) { LOAD16(a, lo, hi)
                    a0_lo=_mm256_sub_epi32(a0_lo,lo); a0_hi=_mm256_sub_epi32(a0_hi,hi); }
                for (const KAction& a : pl.zp) { LOAD16(a, lo, hi)
                    a1_lo=_mm256_add_epi32(a1_lo,lo); a1_hi=_mm256_add_epi32(a1_hi,hi); }
                for (const KAction& a : pl.zn) { LOAD16(a, lo, hi)
                    a1_lo=_mm256_sub_epi32(a1_lo,lo); a1_hi=_mm256_sub_epi32(a1_hi,hi); }
#undef LOAD16

                alignas(32) int32_t o0[16], o1[16];
                _mm256_store_si256((__m256i*)(o0+0), a0_lo);
                _mm256_store_si256((__m256i*)(o0+8), a0_hi);
                _mm256_store_si256((__m256i*)(o1+0), a1_lo);
                _mm256_store_si256((__m256i*)(o1+8), a1_hi);

                const int base_ow = owb * W_BLOCK;
                for (int w = 0; w < W_BLOCK; ++w) {
                    float v0 = static_cast<float>(o0[w]) * s0 + b0;
                    float v1 = static_cast<float>(o1[w]) * s1 + b1;
                    if (relu) { v0 = v0 < 0.f ? 0.f : v0; v1 = v1 < 0.f ? 0.f : v1; }
                    row0[base_ow + w] = v0;
                    row1[base_ow + w] = v1;
                }
            }
        }
    }
}

// ===========================================================================
// Variant B — W_BLOCK=32, WI_EXT=34, OC_TILE=2, OW_TILE=32 (4 YMM lanes/OC)
// One width block per row (OWB=1).
// ===========================================================================
static void kernel_variantB_oc2_ow32(
    const Q8Blocked& x, const BlockedOC2Plan& plan, const BNFusedParams& bn,
    float* y_out, int OC, int OH, int OWB, bool relu)
{
    const int H_PAD = x.H, WB_IN = x.WB, WI_EXT = x.WI_EXT, W_BLOCK = x.W_BLOCK;
    const int stride_wb = C_BLOCK * WI_EXT;
    const int stride_h  = WB_IN * stride_wb;
    const int stride_cb = H_PAD * stride_h;
    const float dq = x.scale;
    const int OW_ROUND = OWB * W_BLOCK;
    const int8_t* base = x.data.data();

    for (int oc_pair = 0; oc_pair < OC / 2; ++oc_pair) {
        const int oc0 = oc_pair * 2, oc1 = oc_pair * 2 + 1;
        const OC2PairLists& pl = plan.oc_pairs[oc_pair];
        const float s0 = bn.scale[oc0] * dq, b0 = bn.bias[oc0];
        const float s1 = bn.scale[oc1] * dq, b1 = bn.bias[oc1];

        for (int oh = 0; oh < OH; ++oh) {
            float* row0 = y_out + oc0 * OH * OW_ROUND + oh * OW_ROUND;
            float* row1 = y_out + oc1 * OH * OW_ROUND + oh * OW_ROUND;

            for (int owb = 0; owb < OWB; ++owb) {
                __m256i a0_0=_mm256_setzero_si256(), a0_1=_mm256_setzero_si256();
                __m256i a0_2=_mm256_setzero_si256(), a0_3=_mm256_setzero_si256();
                __m256i a1_0=_mm256_setzero_si256(), a1_1=_mm256_setzero_si256();
                __m256i a1_2=_mm256_setzero_si256(), a1_3=_mm256_setzero_si256();
                const int owb_off = owb * stride_wb;

#define LOAD32(act, q0,q1,q2,q3) \
    const int _iy##act  = oh + (act).ky; \
    const int _cb##act  = (act).ic >> 4; \
    const int _ci##act  = (act).ic & 0xF; \
    const int _idx##act = _cb##act*stride_cb + _iy##act*stride_h + owb_off \
                        + _ci##act*WI_EXT + (act).kx; \
    const __m128i _va##act = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + _idx##act)); \
    const __m128i _vb##act = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base + _idx##act + 16)); \
    const __m256i q0 = ymm_lo(_va##act); \
    const __m256i q1 = ymm_hi(_va##act); \
    const __m256i q2 = ymm_lo(_vb##act); \
    const __m256i q3 = ymm_hi(_vb##act);

                for (const KAction& a : pl.pp) { LOAD32(a, q0,q1,q2,q3)
                    a0_0=_mm256_add_epi32(a0_0,q0); a0_1=_mm256_add_epi32(a0_1,q1);
                    a0_2=_mm256_add_epi32(a0_2,q2); a0_3=_mm256_add_epi32(a0_3,q3);
                    a1_0=_mm256_add_epi32(a1_0,q0); a1_1=_mm256_add_epi32(a1_1,q1);
                    a1_2=_mm256_add_epi32(a1_2,q2); a1_3=_mm256_add_epi32(a1_3,q3); }
                for (const KAction& a : pl.pn) { LOAD32(a, q0,q1,q2,q3)
                    a0_0=_mm256_add_epi32(a0_0,q0); a0_1=_mm256_add_epi32(a0_1,q1);
                    a0_2=_mm256_add_epi32(a0_2,q2); a0_3=_mm256_add_epi32(a0_3,q3);
                    a1_0=_mm256_sub_epi32(a1_0,q0); a1_1=_mm256_sub_epi32(a1_1,q1);
                    a1_2=_mm256_sub_epi32(a1_2,q2); a1_3=_mm256_sub_epi32(a1_3,q3); }
                for (const KAction& a : pl.np) { LOAD32(a, q0,q1,q2,q3)
                    a0_0=_mm256_sub_epi32(a0_0,q0); a0_1=_mm256_sub_epi32(a0_1,q1);
                    a0_2=_mm256_sub_epi32(a0_2,q2); a0_3=_mm256_sub_epi32(a0_3,q3);
                    a1_0=_mm256_add_epi32(a1_0,q0); a1_1=_mm256_add_epi32(a1_1,q1);
                    a1_2=_mm256_add_epi32(a1_2,q2); a1_3=_mm256_add_epi32(a1_3,q3); }
                for (const KAction& a : pl.nn) { LOAD32(a, q0,q1,q2,q3)
                    a0_0=_mm256_sub_epi32(a0_0,q0); a0_1=_mm256_sub_epi32(a0_1,q1);
                    a0_2=_mm256_sub_epi32(a0_2,q2); a0_3=_mm256_sub_epi32(a0_3,q3);
                    a1_0=_mm256_sub_epi32(a1_0,q0); a1_1=_mm256_sub_epi32(a1_1,q1);
                    a1_2=_mm256_sub_epi32(a1_2,q2); a1_3=_mm256_sub_epi32(a1_3,q3); }
                for (const KAction& a : pl.p0) { LOAD32(a, q0,q1,q2,q3)
                    a0_0=_mm256_add_epi32(a0_0,q0); a0_1=_mm256_add_epi32(a0_1,q1);
                    a0_2=_mm256_add_epi32(a0_2,q2); a0_3=_mm256_add_epi32(a0_3,q3); }
                for (const KAction& a : pl.n0) { LOAD32(a, q0,q1,q2,q3)
                    a0_0=_mm256_sub_epi32(a0_0,q0); a0_1=_mm256_sub_epi32(a0_1,q1);
                    a0_2=_mm256_sub_epi32(a0_2,q2); a0_3=_mm256_sub_epi32(a0_3,q3); }
                for (const KAction& a : pl.zp) { LOAD32(a, q0,q1,q2,q3)
                    a1_0=_mm256_add_epi32(a1_0,q0); a1_1=_mm256_add_epi32(a1_1,q1);
                    a1_2=_mm256_add_epi32(a1_2,q2); a1_3=_mm256_add_epi32(a1_3,q3); }
                for (const KAction& a : pl.zn) { LOAD32(a, q0,q1,q2,q3)
                    a1_0=_mm256_sub_epi32(a1_0,q0); a1_1=_mm256_sub_epi32(a1_1,q1);
                    a1_2=_mm256_sub_epi32(a1_2,q2); a1_3=_mm256_sub_epi32(a1_3,q3); }
#undef LOAD32

                alignas(32) int32_t o0[32], o1[32];
                _mm256_store_si256((__m256i*)(o0+0),  a0_0);
                _mm256_store_si256((__m256i*)(o0+8),  a0_1);
                _mm256_store_si256((__m256i*)(o0+16), a0_2);
                _mm256_store_si256((__m256i*)(o0+24), a0_3);
                _mm256_store_si256((__m256i*)(o1+0),  a1_0);
                _mm256_store_si256((__m256i*)(o1+8),  a1_1);
                _mm256_store_si256((__m256i*)(o1+16), a1_2);
                _mm256_store_si256((__m256i*)(o1+24), a1_3);

                const int base_ow = owb * W_BLOCK;
                for (int w = 0; w < W_BLOCK; ++w) {
                    float v0 = static_cast<float>(o0[w]) * s0 + b0;
                    float v1 = static_cast<float>(o1[w]) * s1 + b1;
                    if (relu) { v0 = v0 < 0.f ? 0.f : v0; v1 = v1 < 0.f ? 0.f : v1; }
                    row0[base_ow + w] = v0;
                    row1[base_ow + w] = v1;
                }
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

struct VariantResult {
    double convert_med, compute_med, total_med;
};

using KernelFn = void(*)(const Q8Blocked&, const BlockedOC2Plan&, const BNFusedParams&,
                         float*, int, int, int, bool);

static VariantResult run_variant(
    const char* label,
    int W_BLOCK, int WI_EXT, KernelFn kernel,
    const std::vector<float>& src_fp32,
    const BlockedOC2Plan& plan, const BNFusedParams& bn,
    int IC, int OC, int H, int W, int OH, int OW, int pad, bool relu)
{
    const int OWB      = (OW + W_BLOCK - 1) / W_BLOCK;
    const int OW_ROUND = OWB * W_BLOCK;

    std::cout << "\n=== " << label
              << "  (W_BLOCK=" << W_BLOCK << " WI_EXT=" << WI_EXT
              << " OWB=" << OWB << " OW_ROUND=" << OW_ROUND << ") ===\n";

    std::vector<float> y_out((size_t)OC * OH * OW_ROUND, 0.0f);
    Q8Blocked xq;

    constexpr int N_WARMUP = 20;
    constexpr int N_TIMED  = 100;

    for (int i = 0; i < N_WARMUP; ++i) {
        convert_fp32_nchw_to_q8_blocked(src_fp32.data(), IC, H, W, pad, W_BLOCK, WI_EXT, xq);
        kernel(xq, plan, bn, y_out.data(), OC, OH, OWB, relu);
    }

    std::vector<double> ct(N_TIMED), kt(N_TIMED), tt(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        const double t0 = now_ms();
        convert_fp32_nchw_to_q8_blocked(src_fp32.data(), IC, H, W, pad, W_BLOCK, WI_EXT, xq);
        const double t1 = now_ms();
        kernel(xq, plan, bn, y_out.data(), OC, OH, OWB, relu);
        const double t2 = now_ms();
        ct[i] = t1 - t0; kt[i] = t2 - t1; tt[i] = t2 - t0;
    }

    volatile float sink = y_out[0] + y_out[(size_t)OC * OH * OW_ROUND - 1];
    (void)sink;

    print_stats("convert_ms", ct, N_TIMED);
    print_stats("compute_ms", kt, N_TIMED);
    print_stats("total_ms  ", tt, N_TIMED);

    std::vector<double> sc = ct, sk = kt, st = tt;
    std::sort(sc.begin(), sc.end());
    std::sort(sk.begin(), sk.end());
    std::sort(st.begin(), st.end());
    return { sc[N_TIMED/2], sk[N_TIMED/2], st[N_TIMED/2] };
}

static void gate(const char* label, double compute_med, double total_med) {
    std::cout << "  --- gate (" << label << ") ---\n";
    if (compute_med > 2.67)
        std::cout << "  STOP: compute_med=" << compute_med << " ms > bucket 2.67 ms.\n";
    else if (compute_med <= 2.2)
        std::cout << "  CONTINUE: compute_med=" << compute_med << " ms <= 2.2 ms.\n";
    else
        std::cout << "  CLOSE: compute_med=" << compute_med << " ms (2.2..2.67 ms).\n";

    if (total_med <= 2.13)
        std::cout << "  STRONG: total_med=" << total_med << " ms <= 2.13 ms (beats packed sparse).\n";
    else if (total_med <= 2.0)
        std::cout << "  STRONG: total_med=" << total_med << " ms <= 2.0 ms.\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);

        const char* conv_name  = "layer2.1.conv2";
        const char* bn_name    = "layer2.1.bn2";
        const bool  apply_relu = false;

        const Conv2DParams& conv = model.convs.at(conv_name);
        const BNFusedParams& bn  = model.fused_bns.at(bn_name);

        const int IC = conv.in_channels, OC = conv.out_channels;
        const int H = 28, W = 28, OH = 28, OW = 28;
        const int pad = conv.pad_h;

        if (IC % C_BLOCK != 0 || OC % 2 != 0)
            throw std::runtime_error("IC must be divisible by 16, OC by 2");

        std::cout << "=== benchmark_w16w32: wide-tile blocked sparse kernels ===\n";
        std::cout << "  Target: " << conv_name << "  IC=" << IC << "  OC=" << OC
                  << "  H=" << H << "  W=" << W << "  pad=" << pad << "\n";

        const BlockedOC2Plan plan = build_plan(conv);
        {
            size_t total_act = 0;
            for (auto& pl : plan.oc_pairs)
                total_act += pl.pp.size()+pl.pn.size()+pl.np.size()+pl.nn.size()
                           + pl.p0.size()+pl.n0.size()+pl.zp.size()+pl.zn.size();
            const size_t max_act = (size_t)(OC/2) * IC * conv.kh * conv.kw;
            std::cout << "  Non-zero actions: " << total_act << " / " << max_act
                      << " (" << 100.0*total_act/max_act << "% density)\n";
        }

        std::vector<float> src_fp32((size_t)IC * H * W);
        {
            uint32_t rng = 0xDEADBEEFu;
            for (float& v : src_fp32) {
                rng = rng * 1664525u + 1013904223u;
                v = (rng >> 31) ? 0.0f : static_cast<float>((rng >> 8) & 0xFFFF) / 65535.0f;
            }
        }

        std::cout << "\n--- baselines ---\n";
        std::cout << "  bucket (q8 ocblock2):  ~2.67 ms\n";
        std::cout << "  packed sparse:         ~2.13 ms\n";
        std::cout << "  NCHWc16 mullo:         ~13.8 ms\n";
        std::cout << "  CB_H_WB_CI_WI OW8:     ~5.99 ms (compute)\n";

        const auto rA = run_variant("Variant A: W_BLOCK=16 OW_TILE=16",
            16, 18, kernel_variantA_oc2_ow16,
            src_fp32, plan, bn, IC, OC, H, W, OH, OW, pad, apply_relu);
        gate("Variant A", rA.compute_med, rA.total_med);

        if (rA.compute_med > 2.67) {
            std::cout << "\nVariant A failed gate (compute > 2.67 ms bucket).\n";
            std::cout << "Trying Variant B anyway — kernel structure is a trivial extension.\n";
        }

        const auto rB = run_variant("Variant B: W_BLOCK=32 OW_TILE=32",
            32, 34, kernel_variantB_oc2_ow32,
            src_fp32, plan, bn, IC, OC, H, W, OH, OW, pad, apply_relu);
        gate("Variant B", rB.compute_med, rB.total_med);

        std::cout << "\n=== summary ===\n";
        std::cout << "  Variant A (W16): compute_med=" << rA.compute_med
                  << " ms  total_med=" << rA.total_med << " ms\n";
        std::cout << "  Variant B (W32): compute_med=" << rB.compute_med
                  << " ms  total_med=" << rB.total_med << " ms\n";
        if (rA.compute_med > 2.67 && rB.compute_med > 2.67)
            std::cout << "  Both variants fail compute > bucket. Close blocked-width path.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
