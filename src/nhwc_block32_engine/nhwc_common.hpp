#pragma once

#include <immintrin.h>

#include <cmath>
#include <cstdint>
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
// to nearest, and clamps to [0, 255].
static inline uint8_t requantize_u8(float v, float inv_out_scale) {
    int32_t q = static_cast<int32_t>(std::lround(v * inv_out_scale));
    q = q < 0 ? 0 : (q > 255 ? 255 : q);
    return static_cast<uint8_t>(q);
}

// Wall-clock time in milliseconds (std::chrono::high_resolution_clock), for benchmark timing.
double now_ms();

// Prints min/median/mean/p90 of N timed samples under the given label.
// Note: takes `v` by value -- it sorts its own copy, leaving the caller's intact.
void print_stats(const char* label, std::vector<double> v, int N);

} // namespace nhwc_block32
