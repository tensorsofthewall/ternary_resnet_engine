// Sparse pos/neg packed activation-panel kernel.
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

// ---------------------------------------------------------------------------
// Layer2 pack: IC=128, padded_h=30, padded_w=34, N_TILE=28, N_PAD=32
//
// Two 16-byte loads per k cover bytes kx..kx+31.
// Safety: kx in {0,1,2}, kx+31 ≤ 33 < padded_w=34. OK.
// Bytes 28-31 land in acc_hi lanes 4-7, which are discarded at store time.
// ---------------------------------------------------------------------------

void pack_a_panel_layer2(const QuantTensorI8& xq, int oh, int8_t* A_panel) {
    const int padded_h = xq.h;
    const int padded_w = xq.w;
    int k = 0;
    for (int ic = 0; ic < 128; ++ic) {
        for (int ky = 0; ky < 3; ++ky) {
            const int8_t* row = xq.data.data() +
                static_cast<size_t>((ic * padded_h + oh + ky) * padded_w);
            for (int kx = 0; kx < 3; ++kx, ++k) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(A_panel + k * 32),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kx)));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(A_panel + k * 32 + 16),
                    _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + kx + 16)));
            }
        }
    }
}

// acc_lo / acc_mid1 / acc_mid2 / acc_hi map to output cols 0-7, 8-15, 16-23, 24-31.
// Only cols 24-27 are valid; store via lower 128 bits of acc_hi (4 floats).

static inline void store28_f32(float* dst,
    __m256 flo, __m256 fmid1, __m256 fmid2, __m256 fhi)
{
    _mm256_storeu_ps(dst,      flo);
    _mm256_storeu_ps(dst + 8,  fmid1);
    _mm256_storeu_ps(dst + 16, fmid2);
    _mm_storeu_ps(dst + 24, _mm256_castps256_ps128(fhi));
}

template <bool HasRelu>
static void compute_layer2_row(
    const int8_t*             A_panel,
    const PackedSparseQ8Plan& plan,
    const BNFusedParams&      bn,
    float                     xq_scale,
    int                       oh,
    Tensor&                   y_out)
{
    constexpr int OC = 128, OH = 28, OW = 28;
    const __m256 zero_v = _mm256_setzero_ps();

    for (int oc = 0; oc < OC; ++oc) {
        __m256i acc_lo   = _mm256_setzero_si256();
        __m256i acc_mid1 = _mm256_setzero_si256();
        __m256i acc_mid2 = _mm256_setzero_si256();
        __m256i acc_hi   = _mm256_setzero_si256();

        for (uint16_t k : plan.pos_idx[oc]) {
            const __m128i raw_lo = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 32));
            const __m128i raw_hi = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 32 + 16));
            acc_lo   = _mm256_add_epi32(acc_lo,   _mm256_cvtepi8_epi32(raw_lo));
            acc_mid1 = _mm256_add_epi32(acc_mid1, _mm256_cvtepi8_epi32(_mm_srli_si128(raw_lo, 8)));
            acc_mid2 = _mm256_add_epi32(acc_mid2, _mm256_cvtepi8_epi32(raw_hi));
            acc_hi   = _mm256_add_epi32(acc_hi,   _mm256_cvtepi8_epi32(_mm_srli_si128(raw_hi, 8)));
        }

        for (uint16_t k : plan.neg_idx[oc]) {
            const __m128i raw_lo = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 32));
            const __m128i raw_hi = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + k * 32 + 16));
            acc_lo   = _mm256_sub_epi32(acc_lo,   _mm256_cvtepi8_epi32(raw_lo));
            acc_mid1 = _mm256_sub_epi32(acc_mid1, _mm256_cvtepi8_epi32(_mm_srli_si128(raw_lo, 8)));
            acc_mid2 = _mm256_sub_epi32(acc_mid2, _mm256_cvtepi8_epi32(raw_hi));
            acc_hi   = _mm256_sub_epi32(acc_hi,   _mm256_cvtepi8_epi32(_mm_srli_si128(raw_hi, 8)));
        }

        const __m256 sc_v   = _mm256_set1_ps(xq_scale * bn.scale[oc]);
        const __m256 bias_v = _mm256_set1_ps(bn.bias[oc]);
        __m256 flo   = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_lo),   sc_v, bias_v);
        __m256 fmid1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_mid1), sc_v, bias_v);
        __m256 fmid2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_mid2), sc_v, bias_v);
        __m256 fhi   = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc_hi),   sc_v, bias_v);

        if constexpr (HasRelu) {
            flo   = _mm256_max_ps(flo,   zero_v);
            fmid1 = _mm256_max_ps(fmid1, zero_v);
            fmid2 = _mm256_max_ps(fmid2, zero_v);
            fhi   = _mm256_max_ps(fhi,   zero_v);
        }

        float* dst = y_out.data.data() +
            static_cast<size_t>(oc) * (OH * OW) +
            static_cast<size_t>(oh * OW);
        store28_f32(dst, flo, fmid1, fmid2, fhi);
    }
}

void packed_q8_3x3_s1_layer2_bn_avx2(
    const int8_t* A_panel, const PackedSparseQ8Plan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer2_row<false>(A_panel, plan, bn, xq_scale, oh, y_out);
}

void packed_q8_3x3_s1_layer2_bn_relu_avx2(
    const int8_t* A_panel, const PackedSparseQ8Plan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer2_row<true>(A_panel, plan, bn, xq_scale, oh, y_out);
}

// ---------------------------------------------------------------------------
// OC_TILE=2 plan builder
//
// For each pair (oc0, oc1): walk K positions in order, emit a PairAction
// whenever at least one weight is nonzero.  Weight arrays are reused across
// pairs to avoid repeated allocation.
// ---------------------------------------------------------------------------

PackedSparsePairPlan build_packed_sparse_pair_plan(const Conv2DParams& conv) {
    const PackedSparseQ8Plan base = build_packed_sparse_q8_plan(conv);

    PackedSparsePairPlan plan;
    plan.K  = base.K;
    plan.OC = base.OC;

    const int n_pairs = base.OC / 2;
    plan.pairs.resize(n_pairs);

    std::vector<int8_t> w0(base.K, 0), w1(base.K, 0);

    for (int p = 0; p < n_pairs; ++p) {
        const int oc0 = p * 2, oc1 = oc0 + 1;

        std::fill(w0.begin(), w0.end(), 0);
        std::fill(w1.begin(), w1.end(), 0);

        for (uint16_t k : base.pos_idx[oc0]) w0[k] = +1;
        for (uint16_t k : base.neg_idx[oc0]) w0[k] = -1;
        for (uint16_t k : base.pos_idx[oc1]) w1[k] = +1;
        for (uint16_t k : base.neg_idx[oc1]) w1[k] = -1;

        auto& out = plan.pairs[p];
        out.clear();
        for (int k = 0; k < base.K; ++k) {
            if (w0[k] != 0 || w1[k] != 0)
                out.push_back({static_cast<uint16_t>(k), w0[k], w1[k]});
        }
    }

    return plan;
}

// ---------------------------------------------------------------------------
// OC_TILE=4 plan builder
// ---------------------------------------------------------------------------

PackedSparseQuadPlan build_packed_sparse_quad_plan(const Conv2DParams& conv) {
    const PackedSparseQ8Plan base = build_packed_sparse_q8_plan(conv);

    PackedSparseQuadPlan plan;
    plan.K  = base.K;
    plan.OC = base.OC;

    const int n_quads = base.OC / 4;
    plan.quads.resize(n_quads);

    std::vector<int8_t> w0(base.K, 0), w1(base.K, 0),
                        w2(base.K, 0), w3(base.K, 0);

    for (int q = 0; q < n_quads; ++q) {
        const int oc0 = q * 4;

        std::fill(w0.begin(), w0.end(), 0);
        std::fill(w1.begin(), w1.end(), 0);
        std::fill(w2.begin(), w2.end(), 0);
        std::fill(w3.begin(), w3.end(), 0);

        for (uint16_t k : base.pos_idx[oc0])   w0[k] = +1;
        for (uint16_t k : base.neg_idx[oc0])   w0[k] = -1;
        for (uint16_t k : base.pos_idx[oc0+1]) w1[k] = +1;
        for (uint16_t k : base.neg_idx[oc0+1]) w1[k] = -1;
        for (uint16_t k : base.pos_idx[oc0+2]) w2[k] = +1;
        for (uint16_t k : base.neg_idx[oc0+2]) w2[k] = -1;
        for (uint16_t k : base.pos_idx[oc0+3]) w3[k] = +1;
        for (uint16_t k : base.neg_idx[oc0+3]) w3[k] = -1;

        auto& out = plan.quads[q];
        out.clear();
        for (int k = 0; k < base.K; ++k) {
            if (w0[k] != 0 || w1[k] != 0 || w2[k] != 0 || w3[k] != 0)
                out.push_back({static_cast<uint16_t>(k),
                               w0[k], w1[k], w2[k], w3[k], {0, 0}});
        }
    }

    return plan;
}

// ---------------------------------------------------------------------------
// OC_TILE=2 compute (layer3, N_TILE=14, N_PAD=16)
//
// sign_epi32(x, y) = x if y>0, 0 if y==0, -x if y<0.
// Used branch-free so the CPU can issue all four accumulator updates
// without branch mis-prediction overhead.
// ---------------------------------------------------------------------------

template <bool HasRelu>
static void compute_layer3_oc2_row(
    const int8_t*               A_panel,
    const PackedSparsePairPlan& plan,
    const BNFusedParams&        bn,
    float                       xq_scale,
    int                         oh,
    Tensor&                     y_out)
{
    constexpr int OC = 256, OH = 14, OW = 14;
    const int n_pairs = OC / 2;
    const __m256 zero_v = _mm256_setzero_ps();

    for (int p = 0; p < n_pairs; ++p) {
        __m256i acc0_lo = _mm256_setzero_si256();
        __m256i acc0_hi = _mm256_setzero_si256();
        __m256i acc1_lo = _mm256_setzero_si256();
        __m256i acc1_hi = _mm256_setzero_si256();

        for (const PairAction& a : plan.pairs[p]) {
            const __m128i raw = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + a.k * 16));
            const __m256i lo = _mm256_cvtepi8_epi32(raw);
            const __m256i hi = _mm256_cvtepi8_epi32(_mm_srli_si128(raw, 8));

            const __m256i sv0 = _mm256_set1_epi32(a.w0);
            acc0_lo = _mm256_add_epi32(acc0_lo, _mm256_sign_epi32(lo, sv0));
            acc0_hi = _mm256_add_epi32(acc0_hi, _mm256_sign_epi32(hi, sv0));

            const __m256i sv1 = _mm256_set1_epi32(a.w1);
            acc1_lo = _mm256_add_epi32(acc1_lo, _mm256_sign_epi32(lo, sv1));
            acc1_hi = _mm256_add_epi32(acc1_hi, _mm256_sign_epi32(hi, sv1));
        }

        const int oc0 = p * 2, oc1 = oc0 + 1;

        const __m256 sc0   = _mm256_set1_ps(xq_scale * bn.scale[oc0]);
        const __m256 bias0 = _mm256_set1_ps(bn.bias[oc0]);
        __m256 flo0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0_lo), sc0, bias0);
        __m256 fhi0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0_hi), sc0, bias0);

        const __m256 sc1   = _mm256_set1_ps(xq_scale * bn.scale[oc1]);
        const __m256 bias1 = _mm256_set1_ps(bn.bias[oc1]);
        __m256 flo1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1_lo), sc1, bias1);
        __m256 fhi1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1_hi), sc1, bias1);

        if constexpr (HasRelu) {
            flo0 = _mm256_max_ps(flo0, zero_v);
            fhi0 = _mm256_max_ps(fhi0, zero_v);
            flo1 = _mm256_max_ps(flo1, zero_v);
            fhi1 = _mm256_max_ps(fhi1, zero_v);
        }

        float* dst0 = y_out.data.data() +
            static_cast<size_t>(oc0) * (OH * OW) + static_cast<size_t>(oh * OW);
        float* dst1 = y_out.data.data() +
            static_cast<size_t>(oc1) * (OH * OW) + static_cast<size_t>(oh * OW);
        store14_f32(dst0, flo0, fhi0);
        store14_f32(dst1, flo1, fhi1);
    }
}

void packed_q8_3x3_s1_layer3_oc2_bn_avx2(
    const int8_t* A_panel, const PackedSparsePairPlan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer3_oc2_row<false>(A_panel, plan, bn, xq_scale, oh, y_out);
}

void packed_q8_3x3_s1_layer3_oc2_bn_relu_avx2(
    const int8_t* A_panel, const PackedSparsePairPlan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer3_oc2_row<true>(A_panel, plan, bn, xq_scale, oh, y_out);
}

// ---------------------------------------------------------------------------
// OC_TILE=4 compute (layer3, N_TILE=14, N_PAD=16)
//
// Eight independent int32 accumulator chains; register budget at inner loop:
// raw(xmm) + lo + hi + acc0_lo/hi + acc1_lo/hi + acc2_lo/hi + acc3_lo/hi
// + one sign broadcast = 13 YMM max live.  No spilling.
// ---------------------------------------------------------------------------

template <bool HasRelu>
static void compute_layer3_oc4_row(
    const int8_t*               A_panel,
    const PackedSparseQuadPlan& plan,
    const BNFusedParams&        bn,
    float                       xq_scale,
    int                         oh,
    Tensor&                     y_out)
{
    constexpr int OC = 256, OH = 14, OW = 14;
    const int n_quads = OC / 4;
    const __m256 zero_v = _mm256_setzero_ps();

    for (int q = 0; q < n_quads; ++q) {
        __m256i acc0_lo = _mm256_setzero_si256();
        __m256i acc0_hi = _mm256_setzero_si256();
        __m256i acc1_lo = _mm256_setzero_si256();
        __m256i acc1_hi = _mm256_setzero_si256();
        __m256i acc2_lo = _mm256_setzero_si256();
        __m256i acc2_hi = _mm256_setzero_si256();
        __m256i acc3_lo = _mm256_setzero_si256();
        __m256i acc3_hi = _mm256_setzero_si256();

        for (const QuadAction& a : plan.quads[q]) {
            const __m128i raw = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(A_panel + a.k * 16));
            const __m256i lo = _mm256_cvtepi8_epi32(raw);
            const __m256i hi = _mm256_cvtepi8_epi32(_mm_srli_si128(raw, 8));

            const __m256i sv0 = _mm256_set1_epi32(a.w0);
            acc0_lo = _mm256_add_epi32(acc0_lo, _mm256_sign_epi32(lo, sv0));
            acc0_hi = _mm256_add_epi32(acc0_hi, _mm256_sign_epi32(hi, sv0));

            const __m256i sv1 = _mm256_set1_epi32(a.w1);
            acc1_lo = _mm256_add_epi32(acc1_lo, _mm256_sign_epi32(lo, sv1));
            acc1_hi = _mm256_add_epi32(acc1_hi, _mm256_sign_epi32(hi, sv1));

            const __m256i sv2 = _mm256_set1_epi32(a.w2);
            acc2_lo = _mm256_add_epi32(acc2_lo, _mm256_sign_epi32(lo, sv2));
            acc2_hi = _mm256_add_epi32(acc2_hi, _mm256_sign_epi32(hi, sv2));

            const __m256i sv3 = _mm256_set1_epi32(a.w3);
            acc3_lo = _mm256_add_epi32(acc3_lo, _mm256_sign_epi32(lo, sv3));
            acc3_hi = _mm256_add_epi32(acc3_hi, _mm256_sign_epi32(hi, sv3));
        }

        const int oc0 = q * 4;

        for (int i = 0; i < 4; ++i) {
            const __m256i* alo = (i == 0) ? &acc0_lo : (i == 1) ? &acc1_lo :
                                 (i == 2) ? &acc2_lo : &acc3_lo;
            const __m256i* ahi = (i == 0) ? &acc0_hi : (i == 1) ? &acc1_hi :
                                 (i == 2) ? &acc2_hi : &acc3_hi;
            const int oc = oc0 + i;

            const __m256 sc   = _mm256_set1_ps(xq_scale * bn.scale[oc]);
            const __m256 bias = _mm256_set1_ps(bn.bias[oc]);
            __m256 flo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(*alo), sc, bias);
            __m256 fhi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(*ahi), sc, bias);

            if constexpr (HasRelu) {
                flo = _mm256_max_ps(flo, zero_v);
                fhi = _mm256_max_ps(fhi, zero_v);
            }

            float* dst = y_out.data.data() +
                static_cast<size_t>(oc) * (OH * OW) + static_cast<size_t>(oh * OW);
            store14_f32(dst, flo, fhi);
        }
    }
}

void packed_q8_3x3_s1_layer3_oc4_bn_avx2(
    const int8_t* A_panel, const PackedSparseQuadPlan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer3_oc4_row<false>(A_panel, plan, bn, xq_scale, oh, y_out);
}

void packed_q8_3x3_s1_layer3_oc4_bn_relu_avx2(
    const int8_t* A_panel, const PackedSparseQuadPlan& plan,
    const BNFusedParams& bn, float xq_scale, int oh, Tensor& y_out)
{
    compute_layer3_oc4_row<true>(A_panel, plan, bn, xq_scale, oh, y_out);
}
