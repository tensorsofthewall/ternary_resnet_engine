// benchmark_cbhwb.cpp — runtime-only benchmark: CB_H_WB_CI_WI_EXT sparse OW8 kernel
// Layout: [CB][H_PAD][WB][CI][WI_EXT], C_BLOCK=16, W_BLOCK=8, WI_EXT=10
// Target: layer2.1.conv2  (IC=128, OC=128, H=28, W=28, K=3×3, S=1, P=1)
// OC2 sparse pattern lists; no branch on weight in hot loop; no sign_epi32.
//
// Usage: ./benchmark_cbhwb <model.bin>

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
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static constexpr int C_BLOCK = 16;
static constexpr int W_BLOCK = 8;
static constexpr int WI_EXT  = 10;   // W_BLOCK + 2: covers kx=0,1,2 within same block

// ---- Blocked input layout [CB][H_PAD][WB][CI][WI_EXT] ----
// For output tile owb, input channel block cb, input channel within block ci,
// padded row iy, and kernel col kx:
//   ptr = &data[(cb*H_PAD + iy)*WB*CI*WI_EXT + owb*CI*WI_EXT + ci*WI_EXT + kx]
// Loads 8 contiguous i8 → extend to 8×i32, covering OW tile owb*8..owb*8+7.
struct Q8CBHWBCiWi {
    int CB = 0, H = 0, WB = 0;   // H = H_PAD
    float scale = 1.0f;
    std::vector<int8_t> data;
};

// ---- Sparse OC2 pattern action ----
struct KAction {
    uint16_t ic;
    uint8_t  ky, kx;
};

// One OC pair: 8 pattern lists (skip 0,0).
struct OC2PairLists {
    std::vector<KAction> pp, pn, np, nn, p0, n0, zp, zn;
};

struct BlockedOC2Plan {
    std::vector<OC2PairLists> oc_pairs;
};

// ---- Quantize FP32 NCHW to Q8CBHWBCiWi (with padding + halo) ----
static void convert_fp32_nchw_to_q8_cbhwb(
    const float* src, int C, int H, int W, int pad,
    Q8CBHWBCiWi& dst)
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
    const size_t total = (size_t)CB * H_PAD * WB * C_BLOCK * WI_EXT;
    dst.data.assign(total, int8_t(0));

    // WI_EXT=10: each block wb stores padded cols wb*W_BLOCK+0..wb*W_BLOCK+9.
    // Positions beyond W_PAD or H_PAD remain zero (padding).
    // This naturally creates the 2-column halo needed for kx=1,2 reads.
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

// ---- Build OC2 sparse pattern lists from ternary weights ----
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
                    else if (w0 > 0)            pl.p0.push_back(a);  // w0=+1, w1=0
                    else if (w0 < 0)            pl.n0.push_back(a);  // w0=-1, w1=0
                    else if (w1 > 0)            pl.zp.push_back(a);  // w0=0,  w1=+1
                    else                        pl.zn.push_back(a);  // w0=0,  w1=-1
                }
            }
        }
    }
    return plan;
}

// ---- AVX2 OC2×OW8 sparse kernel ----
// For each (oc_pair, oh, owb): load 8-wide i8 lane, extend to i32, add/sub into acc.
// No branch on weight; pattern pre-classified into 8 lists.
static void kernel_oc2_ow8(
    const Q8CBHWBCiWi& x,
    const BlockedOC2Plan& plan,
    const BNFusedParams& bn,
    float* y_out,        // [OC][OH][OW_ROUND], written in NCHW order
    int OC, int OH, int OWB,
    bool relu)
{
    const int H_PAD    = x.H;
    const int WB_IN    = x.WB;
    // Pre-compute strides (all compile-time constants with specialisation)
    const int stride_wb = C_BLOCK * WI_EXT;     // 160
    const int stride_h  = WB_IN * stride_wb;    // 640 for WB=4
    const int stride_cb = H_PAD * stride_h;     // 19200 for H_PAD=30
    const float dq      = x.scale;
    const int OW_ROUND  = OWB * W_BLOCK;

    const int8_t* base = x.data.data();

    for (int oc_pair = 0; oc_pair < OC / 2; ++oc_pair) {
        const int oc0 = oc_pair * 2;
        const int oc1 = oc_pair * 2 + 1;
        const OC2PairLists& pl = plan.oc_pairs[oc_pair];
        const float s0 = bn.scale[oc0] * dq, b0 = bn.bias[oc0];
        const float s1 = bn.scale[oc1] * dq, b1 = bn.bias[oc1];

        for (int oh = 0; oh < OH; ++oh) {
            float* row0 = y_out + oc0 * OH * OW_ROUND + oh * OW_ROUND;
            float* row1 = y_out + oc1 * OH * OW_ROUND + oh * OW_ROUND;

            for (int owb = 0; owb < OWB; ++owb) {

                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();

                // Load 8 i8 → 8 i32 for a given KAction.
                // cb = a.ic >> 4, ci = a.ic & 0xF, iy = oh + a.ky.
                // idx = cb*stride_cb + iy*stride_h + owb*stride_wb + ci*WI_EXT + a.kx
                const int owb_off = owb * stride_wb;

#define LOAD8(a_) \
    [&](const KAction& _a) __attribute__((always_inline)) -> __m256i { \
        const int _iy  = oh + _a.ky; \
        const int _cb  = _a.ic >> 4; \
        const int _ci  = _a.ic & 0xF; \
        const int _idx = _cb * stride_cb + _iy * stride_h + owb_off \
                       + _ci * WI_EXT + _a.kx; \
        return _mm256_cvtepi8_epi32( \
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(base + _idx))); \
    }(a_)

                for (const KAction& a : pl.pp) { const __m256i v=LOAD8(a); acc0=_mm256_add_epi32(acc0,v); acc1=_mm256_add_epi32(acc1,v); }
                for (const KAction& a : pl.pn) { const __m256i v=LOAD8(a); acc0=_mm256_add_epi32(acc0,v); acc1=_mm256_sub_epi32(acc1,v); }
                for (const KAction& a : pl.np) { const __m256i v=LOAD8(a); acc0=_mm256_sub_epi32(acc0,v); acc1=_mm256_add_epi32(acc1,v); }
                for (const KAction& a : pl.nn) { const __m256i v=LOAD8(a); acc0=_mm256_sub_epi32(acc0,v); acc1=_mm256_sub_epi32(acc1,v); }
                for (const KAction& a : pl.p0) { acc0=_mm256_add_epi32(acc0, LOAD8(a)); }
                for (const KAction& a : pl.n0) { acc0=_mm256_sub_epi32(acc0, LOAD8(a)); }
                for (const KAction& a : pl.zp) { acc1=_mm256_add_epi32(acc1, LOAD8(a)); }
                for (const KAction& a : pl.zn) { acc1=_mm256_sub_epi32(acc1, LOAD8(a)); }

#undef LOAD8

                alignas(32) int32_t a0[8], a1[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(a0), acc0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(a1), acc1);

                const int base_ow = owb * W_BLOCK;
                for (int w = 0; w < W_BLOCK; ++w) {
                    float v0 = static_cast<float>(a0[w]) * s0 + b0;
                    float v1 = static_cast<float>(a1[w]) * s1 + b1;
                    if (relu) {
                        v0 = v0 < 0.0f ? 0.0f : v0;
                        v1 = v1 < 0.0f ? 0.0f : v1;
                    }
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
              << ":  min=" << mn
              << "  med=" << med
              << "  p90=" << p90
              << "  avg=" << avg << " ms\n";
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

        const int IC = conv.in_channels;   // 128
        const int OC = conv.out_channels;  // 128
        const int H = 28, W = 28, OH = 28, OW = 28;
        const int pad = conv.pad_h;

        if (IC % C_BLOCK != 0 || OC % 2 != 0)
            throw std::runtime_error("IC must be divisible by 16, OC by 2");

        const int OWB     = (OW + W_BLOCK - 1) / W_BLOCK;  // 4
        const int OW_ROUND = OWB * W_BLOCK;                  // 32

        std::cout << "=== benchmark_cbhwb: CB_H_WB_CI_WI_EXT sparse OW8 ===\n";
        std::cout << "  Target: " << conv_name
                  << "  IC=" << IC << "  OC=" << OC
                  << "  H=" << H << "  W=" << W
                  << "  pad=" << pad << "\n";
        std::cout << "  OWB=" << OWB << "  OW_ROUND=" << OW_ROUND
                  << "  WI_EXT=" << WI_EXT << "\n";

        const BlockedOC2Plan plan = build_plan(conv);
        {
            size_t total_act = 0;
            for (auto& pl : plan.oc_pairs)
                total_act += pl.pp.size() + pl.pn.size() + pl.np.size() + pl.nn.size()
                           + pl.p0.size() + pl.n0.size() + pl.zp.size() + pl.zn.size();
            const int kernel_elems = IC * conv.kh * conv.kw;  // per OC pair
            const size_t max_act   = (size_t)(OC / 2) * kernel_elems;
            std::cout << "  Total non-zero actions: " << total_act
                      << " / " << max_act
                      << " (" << 100.0 * total_act / max_act << "% density)\n";
            const OC2PairLists& p0 = plan.oc_pairs[0];
            std::cout << "  OC pair 0 lists: pp=" << p0.pp.size()
                      << " pn=" << p0.pn.size() << " np=" << p0.np.size()
                      << " nn=" << p0.nn.size() << " p0=" << p0.p0.size()
                      << " n0=" << p0.n0.size() << " zp=" << p0.zp.size()
                      << " zn=" << p0.zn.size() << "\n";
        }

        // Synthetic FP32 NCHW input (post-ReLU: values in [0,1], ~50% zero)
        std::vector<float> src_fp32((size_t)IC * H * W);
        {
            uint32_t rng = 0xDEADBEEFu;
            for (float& v : src_fp32) {
                rng = rng * 1664525u + 1013904223u;
                v = (rng >> 31) ? 0.0f : static_cast<float>((rng >> 8) & 0xFFFF) / 65535.0f;
            }
        }

        std::vector<float> y_out((size_t)OC * OH * OW_ROUND, 0.0f);
        Q8CBHWBCiWi xq;

        constexpr int N_WARMUP = 20;
        constexpr int N_TIMED  = 100;

        // Warmup
        for (int i = 0; i < N_WARMUP; ++i) {
            convert_fp32_nchw_to_q8_cbhwb(src_fp32.data(), IC, H, W, pad, xq);
            kernel_oc2_ow8(xq, plan, bn, y_out.data(), OC, OH, OWB, apply_relu);
        }

        std::vector<double> ct(N_TIMED), kt(N_TIMED), tt(N_TIMED);
        for (int i = 0; i < N_TIMED; ++i) {
            const double t0 = now_ms();
            convert_fp32_nchw_to_q8_cbhwb(src_fp32.data(), IC, H, W, pad, xq);
            const double t1 = now_ms();
            kernel_oc2_ow8(xq, plan, bn, y_out.data(), OC, OH, OWB, apply_relu);
            const double t2 = now_ms();
            ct[i] = t1 - t0;
            kt[i] = t2 - t1;
            tt[i] = t2 - t0;
        }

        // Anti-DCE
        volatile float sink = y_out[0] + y_out[(size_t)OC * OH * OW_ROUND - 1];
        (void)sink;

        std::cout << "\n--- timing (n=" << N_TIMED << ", synthetic post-ReLU input) ---\n";
        print_stats("convert_ms", ct, N_TIMED);
        print_stats("compute_ms", kt, N_TIMED);
        print_stats("total_ms  ", tt, N_TIMED);

        std::cout << "\n--- baselines ---\n";
        std::cout << "  bucket (q8 ocblock2):       ~2.67 ms\n";
        std::cout << "  packed sparse:              ~2.13 ms\n";
        std::cout << "  NCHWc16 AVX2 mullo:         ~13.8 ms\n";

        std::vector<double> sorted_k = kt;
        std::sort(sorted_k.begin(), sorted_k.end());
        const double compute_med = sorted_k[N_TIMED / 2];

        std::cout << "\n--- gate ---\n";
        if (compute_med > 2.67) {
            std::cout << "  STOP: compute_med=" << compute_med
                      << " ms > bucket 2.67 ms. Negative result.\n";
        } else if (compute_med <= 2.2) {
            std::cout << "  INTERESTING: compute_med=" << compute_med
                      << " ms. Check total_ms vs packed sparse.\n";
        } else {
            std::cout << "  CLOSE: compute_med=" << compute_med
                      << " ms (between 2.2 and 2.67 ms).\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
