#pragma once

#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nhwc_block32 {

// Fixed placeholder activation quantization scale (static, uncalibrated),
// matching q2q_engine's ACT_SCALE -- kept as a separate constant here so this
// engine has no header dependency on q2q_engine.
constexpr float ACT_SCALE = 0.05f;

// Horizontal-sums the 8 int32 lanes of an AVX2 vector into one scalar.
static inline int32_t hsum_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// Scales a dequantized FP32 value back to u8 by `v * inv_out_scale`, rounds
// to nearest with ties away from zero (matching std::lround's convention,
// but as a single add+truncate -- no libm call), and clamps to [0, 255].
// Adding a +-0.5 bias before truncating toward zero is exact for the
// magnitudes seen here (well under 2^23, where float addition of 0.5 is
// lossless), so this is bit-identical to std::lround for every value this
// function is ever called with.
static inline uint8_t requantize_u8(float v, float inv_out_scale) {
    float scaled = v * inv_out_scale;
    scaled += (scaled >= 0.0f) ? 0.5f : -0.5f;
    int32_t q = static_cast<int32_t>(scaled);
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    return static_cast<uint8_t>(q);
}

// Vectorized 4-channel OC4 epilogue: takes 4 int32 dot-product accumulators
// `d`, applies the per-channel fused BN affine (d*s + bb), optional ReLU,
// scales by `inv_out_scale`, and requantizes to u8 with round-half-away-
// from-zero + clamp to [0,255] -- bit-identical to calling requantize_u8
// on `d[k]*s[k]+bb[k]` (after ReLU) for each k in [0,4), but as 4-wide SSE
// ops instead of 4 scalar float ops. Writes the 4 output bytes to out_px[0..3].
static inline void epilogue_oc4_avx2(
    const int32_t d[4], const float s[4], const float bb[4],
    bool relu, float inv_out_scale, uint8_t* out_px)
{
    const __m128i di = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d));
    __m128 v = _mm_cvtepi32_ps(di);
    const __m128 sv = _mm_loadu_ps(s);
    const __m128 bv = _mm_loadu_ps(bb);
    v = _mm_fmadd_ps(v, sv, bv);
    if (relu) v = _mm_max_ps(v, _mm_setzero_ps());
    v = _mm_mul_ps(v, _mm_set1_ps(inv_out_scale));

    const __m128 sign_mask = _mm_set1_ps(-0.0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 bias = _mm_or_ps(_mm_and_ps(v, sign_mask), half);
    v = _mm_add_ps(v, bias);

    __m128i qi = _mm_cvttps_epi32(v);
    qi = _mm_max_epi32(qi, _mm_setzero_si128());
    qi = _mm_min_epi32(qi, _mm_set1_epi32(255));

    const __m128i packed16 = _mm_packus_epi32(qi, qi);
    const __m128i packed8 = _mm_packus_epi16(packed16, packed16);
    const uint32_t packed = static_cast<uint32_t>(_mm_cvtsi128_si32(packed8));
    std::memcpy(out_px, &packed, 4);
}

// Wall-clock time in milliseconds (std::chrono::high_resolution_clock), for benchmark timing.
double now_ms();

// Prints min/median/mean/p90 of N timed samples under the given label.
// Note: takes `v` by value -- it sorts its own copy, leaving the caller's intact.
void print_stats(const char* label, std::vector<double> v, int N);

} // namespace nhwc_block32
