#include "blocked_conv3x3.hpp"
#include "q8_runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace {

// Shared scalar implementation for any KH/KW/stride/pad combination.
static void blocked_conv_ref_impl(
    const BlockedTensorI8& xq,
    const BlockedConvWeights& bw,
    const BNFusedParams& bn,
    bool relu,
    int pad,
    int stride,
    BlockedTensorFP32& output)
{
    const int KH = bw.kh;
    const int KW = bw.kw;
    const int OH = (xq.H + 2 * pad - KH) / stride + 1;
    const int OW = (xq.W + 2 * pad - KW) / stride + 1;

    output.reuse(bw.oc_blocks, OH, OW);

    // Returns 0 for out-of-bounds — avoids materialising a padded copy.
    auto get_a = [&](int cb, int h, int w, int ci) -> int8_t {
        if (h < 0 || h >= xq.H || w < 0 || w >= xq.W) return 0;
        return xq.pixel(cb, h, w)[ci];
    };

    for (int oc_cb = 0; oc_cb < bw.oc_blocks; ++oc_cb) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc[16] = {};

                for (int ic_cb = 0; ic_cb < bw.ic_blocks; ++ic_cb) {
                    for (int ky = 0; ky < KH; ++ky) {
                        const int ih = oh * stride + ky - pad;
                        for (int kx = 0; kx < KW; ++kx) {
                            const int iw = ow * stride + kx - pad;
                            const int8_t* tile = bw.block(oc_cb, ic_cb, ky, kx);
                            for (int ci_in = 0; ci_in < 16; ++ci_in) {
                                const int32_t a =
                                    static_cast<int32_t>(get_a(ic_cb, ih, iw, ci_in));
                                if (a == 0) continue;
                                for (int ci_out = 0; ci_out < 16; ++ci_out)
                                    acc[ci_out] += static_cast<int32_t>(
                                        tile[ci_in * 16 + ci_out]) * a;
                            }
                        }
                    }
                }

                float* out_pix = output.pixel(oc_cb, oh, ow);
                for (int ci_out = 0; ci_out < 16; ++ci_out) {
                    const int oc  = oc_cb * 16 + ci_out;
                    float val = static_cast<float>(acc[ci_out]) * xq.scale
                                * bn.scale[oc] + bn.bias[oc];
                    if (relu && val < 0.0f) val = 0.0f;
                    out_pix[ci_out] = val;
                }
            }
        }
    }
}

} // namespace

void blocked_conv3x3_s1_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_ref_impl(xq, w, bn, relu, /*pad=*/1, /*stride=*/1, output);
}

void blocked_conv3x3_s2_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_ref_impl(xq, w, bn, relu, /*pad=*/1, /*stride=*/2, output);
}

void blocked_conv1x1_s2_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_ref_impl(xq, w, bn, relu, /*pad=*/0, /*stride=*/2, output);
}

// ---------------------------------------------------------------------------
// Phase 5 — AVX2 sign_epi32 kernel
// sign_epi32(av, w): returns av when w>0, -av when w<0, 0 when w==0.
// Exactly equivalent to mullo(av, w) for w in {-1, 0, +1}. Latency: 1 cycle.
// ---------------------------------------------------------------------------
namespace {

static void blocked_conv_avx2_impl(
    const BlockedTensorI8& xq,
    const BlockedConvWeights& bw,
    const BNFusedParams& bn,
    bool relu,
    int pad,
    int stride,
    BlockedTensorFP32& output)
{
    const int KH = bw.kh;
    const int KW = bw.kw;
    const int OH = (xq.H + 2 * pad - KH) / stride + 1;
    const int OW = (xq.W + 2 * pad - KW) / stride + 1;

    output.reuse(bw.oc_blocks, OH, OW);

    // acc_lo/acc_hi are YMM registers — 16 × i32 accumulator per output pixel.
    // Weight tiles (18 KB per oc_cb for layer2) stay hot in L1 across (oh,ow) loop.
    for (int oc_cb = 0; oc_cb < bw.oc_blocks; ++oc_cb) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {

                __m256i acc_lo = _mm256_setzero_si256();
                __m256i acc_hi = _mm256_setzero_si256();

                for (int ic_cb = 0; ic_cb < bw.ic_blocks; ++ic_cb) {
                    for (int ky = 0; ky < KH; ++ky) {
                        const int ih = oh * stride + ky - pad;
                        for (int kx = 0; kx < KW; ++kx) {
                            const int iw = ow * stride + kx - pad;
                            if (ih < 0 || ih >= xq.H || iw < 0 || iw >= xq.W)
                                continue;

                            const int8_t* W     = bw.block(oc_cb, ic_cb, ky, kx);
                            const int8_t* a_pix = xq.pixel(ic_cb, ih, iw);

                            for (int ci_in = 0; ci_in < 16; ++ci_in) {
                                const int32_t a_val =
                                    static_cast<int32_t>(a_pix[ci_in]);
                                if (a_val == 0) continue;

                                const __m256i av = _mm256_set1_epi32(a_val);
                                const __m128i w128 = _mm_loadu_si128(
                                    reinterpret_cast<const __m128i*>(
                                        W + ci_in * 16));
                                // sign_epi32(av, w) = av*sign(w) for ternary w
                                acc_lo = _mm256_add_epi32(acc_lo,
                                    _mm256_sign_epi32(av,
                                        _mm256_cvtepi8_epi32(w128)));
                                acc_hi = _mm256_add_epi32(acc_hi,
                                    _mm256_sign_epi32(av,
                                        _mm256_cvtepi8_epi32(
                                            _mm_bsrli_si128(w128, 8))));
                            }
                        }
                    }
                }

                alignas(32) int32_t acc[16];
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc),     acc_lo);
                _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 8), acc_hi);

                float* out_pix = output.pixel(oc_cb, oh, ow);
                for (int ci_out = 0; ci_out < 16; ++ci_out) {
                    const int oc = oc_cb * 16 + ci_out;
                    float val = static_cast<float>(acc[ci_out]) * xq.scale
                                * bn.scale[oc] + bn.bias[oc];
                    if (relu && val < 0.0f) val = 0.0f;
                    out_pix[ci_out] = val;
                }
            }
        }
    }
}

} // namespace

void blocked_conv3x3_s1_avx2(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_avx2_impl(xq, w, bn, relu, /*pad=*/1, /*stride=*/1, output);
}

void blocked_conv3x3_s2_avx2(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_avx2_impl(xq, w, bn, relu, /*pad=*/1, /*stride=*/2, output);
}
