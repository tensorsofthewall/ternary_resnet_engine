#pragma once

#include <immintrin.h>

#include <cstdint>
#include <string>
#include <vector>

namespace q2q {

constexpr int OC_TILE = 4;
constexpr int K_CHUNK = 32;
constexpr int PAD_Q8 = 1;
constexpr float ACT_SCALE = 0.05f;

static inline int32_t hsum_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

double now_ms();

void print_stats(const char* label, std::vector<double> v, int N);

std::vector<float> read_f32_file(const std::string& path, size_t expected_count);

} // namespace q2q
