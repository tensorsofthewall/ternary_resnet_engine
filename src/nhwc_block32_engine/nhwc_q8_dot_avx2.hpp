#pragma once

#include "nhwc_common.hpp"

#include <immintrin.h>
#include <cstdint>

namespace nhwc_block32 {

// 32-wide uint8 activation . int8 ternary {-1,0,+1} weight dot product,
// without the final horizontal sum: _mm256_maddubs_epi16 multiplies+
// pairwise-adds the 32 (u8 x s8) products into 16 int16 lanes, then
// _mm256_madd_epi16 against an all-ones vector sums adjacent int16 pairs
// into 8 int32 lanes. Safe from int16 overflow: each product is in
// [-255,255], so each pairwise sum is in [-510,510]. Callers that need the
// dot product as a scalar should hsum_epi32 the result; callers accumulating
// many blocks per output pixel should sum these __m256i partials with
// _mm256_add_epi32 and hsum only once at the end (hsum_epi32 is the
// expensive part of this kernel -- see dot_u8_i8_32_avx2).
static inline __m256i dot_u8_i8_32_partial_avx2(const uint8_t* a, const int8_t* w) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    const __m256i prod16 = _mm256_maddubs_epi16(va, vw);
    const __m256i ones16 = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(prod16, ones16);
}

// 32-wide uint8 activation . int8 ternary {-1,0,+1} weight dot product ->
// int32 (dot_u8_i8_32_partial_avx2 followed by a horizontal sum).
static inline int32_t dot_u8_i8_32_avx2(const uint8_t* a, const int8_t* w) {
    return hsum_epi32(dot_u8_i8_32_partial_avx2(a, w));
}

// Computes four 32-wide dot products of the same activation block `a`
// against four different ternary weight blocks (one per output channel),
// amortizing the activation load across all four.
static inline void dot_u8_i8_32x4_avx2(
    const uint8_t* a,
    const int8_t* w0, const int8_t* w1, const int8_t* w2, const int8_t* w3,
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3)
{
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i ones16 = _mm256_set1_epi16(1);

    const __m256i p0 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0)));
    const __m256i p1 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1)));
    const __m256i p2 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2)));
    const __m256i p3 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w3)));

    d0 = hsum_epi32(_mm256_madd_epi16(p0, ones16));
    d1 = hsum_epi32(_mm256_madd_epi16(p1, ones16));
    d2 = hsum_epi32(_mm256_madd_epi16(p2, ones16));
    d3 = hsum_epi32(_mm256_madd_epi16(p3, ones16));
}

} // namespace nhwc_block32
