#include "block32_residual.hpp"

#include "nhwc_common.hpp"

#include <immintrin.h>
#include <omp.h>

namespace nhwc_block32 {

void block32_residual_add_relu_scalar(
    const q2q::QTensorNHWC& a, const q2q::QTensorNHWC& b,
    float out_scale, int H, int W, int C, q2q::QTensorNHWC& y)
{
    const float inv_out_scale = 1.0f / out_scale;
    const float sa = a.scale, sb = b.scale;
    const uint8_t* abase = a.data.data();
    const uint8_t* bbase = b.data.data();
    uint8_t* ybase = y.data.data();

    for (int h = 0; h < H; ++h) {
        const uint8_t* a_row = abase + ((static_cast<size_t>(h + a.border) * a.W + static_cast<size_t>(a.border)) * C);
        const uint8_t* b_row = bbase + ((static_cast<size_t>(h + b.border) * b.W + static_cast<size_t>(b.border)) * C);
        uint8_t*       y_row = ybase + ((static_cast<size_t>(h + y.border) * y.W + static_cast<size_t>(y.border)) * C);
        for (int w = 0; w < W; ++w) {
            const uint8_t* ap = a_row + static_cast<size_t>(w) * C;
            const uint8_t* bp = b_row + static_cast<size_t>(w) * C;
            uint8_t*       yp = y_row + static_cast<size_t>(w) * C;
            for (int c = 0; c < C; ++c) {
                float sum = static_cast<float>(ap[c]) * sa + static_cast<float>(bp[c]) * sb;
                sum = sum < 0.f ? 0.f : sum;
                yp[c] = requantize_u8(sum, inv_out_scale);
            }
        }
    }
}

void block32_residual_add_relu_avx2(
    const q2q::QTensorNHWC& a, const q2q::QTensorNHWC& b,
    float out_scale, int H, int W, int C, q2q::QTensorNHWC& y)
{
    const float inv_out_scale = 1.0f / out_scale;
    const __m256 sa_v   = _mm256_set1_ps(a.scale);
    const __m256 sb_v   = _mm256_set1_ps(b.scale);
    const __m256 inv_os = _mm256_set1_ps(inv_out_scale);
    const __m256 zero   = _mm256_setzero_ps();
    const __m256i max255 = _mm256_set1_epi32(255);
    const uint8_t* abase = a.data.data();
    const uint8_t* bbase = b.data.data();
    uint8_t* ybase = y.data.data();

    // Parallelize over output rows -- each row reads disjoint, non-
    // overlapping rows of `a`/`b` and writes a disjoint row of `y`, so
    // there are no races.
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const uint8_t* ap = abase + ((static_cast<size_t>(h + a.border) * a.W + static_cast<size_t>(a.border)) * C) + static_cast<size_t>(w) * C;
            const uint8_t* bp = bbase + ((static_cast<size_t>(h + b.border) * b.W + static_cast<size_t>(b.border)) * C) + static_cast<size_t>(w) * C;
            uint8_t*       yp = ybase + ((static_cast<size_t>(h + y.border) * y.W + static_cast<size_t>(y.border)) * C) + static_cast<size_t>(w) * C;
            for (int c = 0; c < C; c += 8) {
                const __m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ap + c));
                const __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp + c));
                const __m256 af = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(a8));
                const __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b8));
                __m256 sum = _mm256_fmadd_ps(af, sa_v, _mm256_mul_ps(bf, sb_v));
                sum = _mm256_max_ps(sum, zero);
                const __m256 scaled = _mm256_mul_ps(sum, inv_os);
                __m256i qi = _mm256_cvtps_epi32(scaled);
                qi = _mm256_max_epi32(qi, _mm256_setzero_si256());
                qi = _mm256_min_epi32(qi, max255);
                const __m128i lo = _mm256_castsi256_si128(qi);
                const __m128i hi = _mm256_extracti128_si256(qi, 1);
                const __m128i packed16 = _mm_packus_epi32(lo, hi);
                const __m128i packed8  = _mm_packus_epi16(packed16, packed16);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(yp + c), packed8);
            }
        }
    }
}

} // namespace nhwc_block32
