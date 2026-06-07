#pragma once

#include "q2q_common.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cstring>

namespace q2q {

// Vectorized BN-scale + bias + (optional) ReLU + requantize-to-u8 for an
// OC_TILE=4 group, replacing 4x scalar requantize_u8/std::lround with one
// SSE pipeline + a single 4-byte store. Validated standalone (~2.08x paired
// win in benchmark_q2q_microkernel_opt; ~4.6x for conv1's larger pixel count
// in benchmark_q2q_conv1_opt) -- shared by both the conv1 kernel and the
// dense apanel conv.
static inline void epilogue_oc4_simd(
    int32_t d0, int32_t d1, int32_t d2, int32_t d3,
    const float s[OC_TILE], const float b[OC_TILE],
    bool relu, float inv_out_scale, uint8_t* out_px)
{
    const __m128i dvec = _mm_setr_epi32(d0, d1, d2, d3);
    const __m128  sv   = _mm_loadu_ps(s);
    const __m128  bv   = _mm_loadu_ps(b);
    __m128 vf = _mm_fmadd_ps(_mm_cvtepi32_ps(dvec), sv, bv);
    if (relu) vf = _mm_max_ps(vf, _mm_setzero_ps());
    vf = _mm_mul_ps(vf, _mm_set1_ps(inv_out_scale));
    __m128i qi = _mm_cvtps_epi32(vf);
    qi = _mm_max_epi32(qi, _mm_setzero_si128());
    qi = _mm_min_epi32(qi, _mm_set1_epi32(255));
    const __m128i packed16 = _mm_packus_epi32(qi, qi);
    const __m128i packed8  = _mm_packus_epi16(packed16, packed16);
    const uint32_t out32 = static_cast<uint32_t>(_mm_cvtsi128_si32(packed8));
    std::memcpy(out_px, &out32, sizeof(out32));
}

} // namespace q2q
