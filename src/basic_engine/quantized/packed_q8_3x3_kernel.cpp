// Phase 2: sparse pos/neg packed activation-panel kernel.
//
// For each output row (oh):
//   1. pack_a_panel: copy 14 (layer3) or 7 (layer4) consecutive i8 values per K position
//      into A_panel[k * N_PAD + n], padded to N_PAD=16. Load=16 bytes (layer3) or 8
//      bytes (layer4, safe because padded_w=13 and kx+7<=9<13).
//   2. compute: for each OC, iterate pos_idx then neg_idx and add/sub A_panel rows
//      into two __m256i int32 accumulators (layer3) or one (layer4); apply BN+ReLU.
//
// Int32 overflow check: max |acc| = 127 * K = 127 * 4608 = 585,216 << INT32_MAX. Safe.

#include "quantized/packed_q8_3x3_kernel.hpp"

#include <cstring>
#include <immintrin.h>

// ---------------------------------------------------------------------------
// Plan builder
// ---------------------------------------------------------------------------

PackedSparseQ8Plan build_packed_sparse_q8_plan(const Conv2DParams& conv) {
    const int IC = conv.in_channels;
    const int OC = conv.out_channels;
    const int KH = conv.kh;
    const int KW = conv.kw;

    PackedSparseQ8Plan plan;
    plan.K  = IC * KH * KW;
    plan.OC = OC;
    plan.pos_idx.resize(OC);
    plan.neg_idx.resize(OC);

    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    const int k = (ic * KH + ky) * KW + kx;
                    const int8_t w = conv.weight[
                        ((oc * IC + ic) * KH + ky) * KW + kx
                    ];
                    if (w > 0)
                        plan.pos_idx[oc].push_back(static_cast<uint16_t>(k));
                    else if (w < 0)
                        plan.neg_idx[oc].push_back(static_cast<uint16_t>(k));
                }
            }
        }
    }

    return plan;
}

// ---------------------------------------------------------------------------
// Layer3 pack: IC=256, padded_h=16, padded_w=20, N_TILE=14, N_PAD=16
//
// 16-byte load safety: kx in {0,1,2}, load covers cols kx..kx+15 ≤ 17 < 20. OK.
// Bytes 14-15 of A_panel[k*16] come from the padded input right-pad region (zeros
// or valid data), but they correspond to output cols 14-15 which don't exist and
// are never read from acc_hi in the compute step. No masking needed.
// ---------------------------------------------------------------------------

void pack_a_panel_layer3(const QuantTensorI8& xq, int oh, int8_t* A_panel) {
    const int padded_h = xq.h;
    const int padded_w = xq.w;
    int k = 0;
    for (int ic = 0; ic < 256; ++ic) {
        for (int ky = 0; ky < 3; ++ky) {
            const int8_t* row = xq.data.data() +
                static_cast<size_t>((ic * padded_h + oh + ky) * padded_w);
            for (int kx = 0; kx < 3; ++kx, ++k) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(A_panel + k * 16),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kx)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Layer4 pack: IC=512, padded_h=9, padded_w=13, N_TILE=7, N_PAD=16
//
// Only 8 bytes loaded per k (loadl_epi64 zeroes upper 8 bytes).
// Load safety: kx in {0,1,2}, kx+7 ≤ 9 < 13. OK.
// Byte 7 lands in acc_lo lane 7 but is discarded during the 7-float store.
// ---------------------------------------------------------------------------

void pack_a_panel_layer4(const QuantTensorI8& xq, int oh, int8_t* A_panel) {
    const int padded_h = xq.h;
    const int padded_w = xq.w;
    int k = 0;
    for (int ic = 0; ic < 512; ++ic) {
        for (int ky = 0; ky < 3; ++ky) {
            const int8_t* row = xq.data.data() +
                static_cast<size_t>((ic * padded_h + oh + ky) * padded_w);
            for (int kx = 0; kx < 3; ++kx, ++k) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(A_panel + k * 16),
                    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + kx)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Layer3 compute helpers
//
// acc_lo: 8 int32 for output cols 0..7
// acc_hi: 8 int32 for output cols 8..15, but only 0..5 (output cols 8..13) written
// ---------------------------------------------------------------------------

static inline void store14_f32(float* dst, __m256 flo, __m256 fhi) {
    _mm256_storeu_ps(dst, flo);
    _mm_storeu_ps(dst + 8, _mm256_castps256_ps128(fhi));
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, _mm256_extractf128_ps(fhi, 1));
    dst[12] = tmp[0];
    dst[13] = tmp[1];
}

template <bool HasRelu>
static void compute_layer3_row(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out)
{
    constexpr int OC = 256, OH = 14, OW = 14;
    const __m256 zero_v = _mm256_setzero_ps();

    for (int oc = 0; oc < OC; ++oc) {
        __m256i acc_lo = _mm256_setzero_si256();
        __m256i acc_hi = _mm256_setzero_si256();

        for (uint16_t k : plan.pos_idx[oc]) {
            const __m128i raw = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 16));
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_cvtepi8_epi32(raw));
            acc_hi = _mm256_add_epi32(acc_hi,
                _mm256_cvtepi8_epi32(_mm_srli_si128(raw, 8)));
        }

        for (uint16_t k : plan.neg_idx[oc]) {
            const __m128i raw = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 16));
            acc_lo = _mm256_sub_epi32(acc_lo, _mm256_cvtepi8_epi32(raw));
            acc_hi = _mm256_sub_epi32(acc_hi,
                _mm256_cvtepi8_epi32(_mm_srli_si128(raw, 8)));
        }

        const __m256 sc_v   = _mm256_set1_ps(xq_scale * bn.scale[oc]);
        const __m256 bias_v = _mm256_set1_ps(bn.bias[oc]);
        __m256 flo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_lo), sc_v, bias_v);
        __m256 fhi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_hi), sc_v, bias_v);

        if constexpr (HasRelu) {
            flo = _mm256_max_ps(flo, zero_v);
            fhi = _mm256_max_ps(fhi, zero_v);
        }

        float* dst = y_out.data.data() +
            static_cast<size_t>(oc) * (OH * OW) +
            static_cast<size_t>(oh * OW);
        store14_f32(dst, flo, fhi);
    }
}

void packed_q8_3x3_s1_layer3_bn_avx2(
    const int8_t* A_panel, const PackedSparseQ8Plan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer3_row<false>(A_panel, plan, bn, xq_scale, oh, y_out);
}

void packed_q8_3x3_s1_layer3_bn_relu_avx2(
    const int8_t* A_panel, const PackedSparseQ8Plan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer3_row<true>(A_panel, plan, bn, xq_scale, oh, y_out);
}

// ---------------------------------------------------------------------------
// Layer4 compute helpers
//
// Only acc_lo (8 int32) needed; output cols 0..6, lane 7 discarded.
// ---------------------------------------------------------------------------

static inline void store7_f32(float* dst, __m256 fv) {
    _mm_storeu_ps(dst, _mm256_castps256_ps128(fv));
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, _mm256_extractf128_ps(fv, 1));
    dst[4] = tmp[0];
    dst[5] = tmp[1];
    dst[6] = tmp[2];
}

template <bool HasRelu>
static void compute_layer4_row(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out)
{
    constexpr int OC = 512, OH = 7, OW = 7;
    const __m256 zero_v = _mm256_setzero_ps();

    for (int oc = 0; oc < OC; ++oc) {
        __m256i acc_lo = _mm256_setzero_si256();

        for (uint16_t k : plan.pos_idx[oc]) {
            acc_lo = _mm256_add_epi32(acc_lo,
                _mm256_cvtepi8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(A_panel + k * 16))));
        }

        for (uint16_t k : plan.neg_idx[oc]) {
            acc_lo = _mm256_sub_epi32(acc_lo,
                _mm256_cvtepi8_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(A_panel + k * 16))));
        }

        const __m256 sc_v   = _mm256_set1_ps(xq_scale * bn.scale[oc]);
        const __m256 bias_v = _mm256_set1_ps(bn.bias[oc]);
        __m256 fv = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_lo), sc_v, bias_v);

        if constexpr (HasRelu) {
            fv = _mm256_max_ps(fv, zero_v);
        }

        float* dst = y_out.data.data() +
            static_cast<size_t>(oc) * (OH * OW) +
            static_cast<size_t>(oh * OW);
        store7_f32(dst, fv);
    }
}

void packed_q8_3x3_s1_layer4_bn_avx2(
    const int8_t* A_panel, const PackedSparseQ8Plan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer4_row<false>(A_panel, plan, bn, xq_scale, oh, y_out);
}

void packed_q8_3x3_s1_layer4_bn_relu_avx2(
    const int8_t* A_panel, const PackedSparseQ8Plan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer4_row<true>(A_panel, plan, bn, xq_scale, oh, y_out);
}
