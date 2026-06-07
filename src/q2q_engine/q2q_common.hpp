#pragma once

#include <immintrin.h>

#include <cstdint>
#include <string>
#include <vector>

namespace q2q {

constexpr int OC_TILE = 4;     // output channels processed per VNNI accumulator group
constexpr int K_CHUNK = 32;    // reduction-axis chunk width (one __m256i of int8 weights/activations)
constexpr int PAD_Q8 = 1;      // halo width for q8 NHWC tensors with 3x3/pad<=1 convs
constexpr float ACT_SCALE = 0.05f;  // fixed placeholder activation quantization scale (static, uncalibrated)

// Horizontal-sums the 8 int32 lanes of a VNNI dpbusd accumulator into one scalar.
static inline int32_t hsum_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// Wall-clock time in milliseconds (std::chrono::high_resolution_clock), for benchmark timing.
double now_ms();

// Prints min/median/mean/p90 of N timed samples under the given label.
// Note: takes `v` by value -- it sorts its own copy, leaving the caller's intact.
void print_stats(const char* label, std::vector<double> v, int N);

// Reads a raw binary file of `expected_count` native-endian floats.
// Throws std::runtime_error if the file can't be opened or the read comes up short.
std::vector<float> read_f32_file(const std::string& path, size_t expected_count);

} // namespace q2q
