#pragma once

#include "bn_fusion.hpp"
#include "model.hpp"
#include "q2q_common.hpp"
#include "q2q_tensor.hpp"
#include "q2q_weight_panel.hpp"

#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace q2q {

// Computes the OC_TILE=4 dot products for one output pixel (oh,ow): walks the
// KHxKW receptive field and IC_CHUNKS reduction chunks, accumulating four
// VNNI dpbusd_epi32 lanes (one per output channel in the tile) and
// horizontal-summing them into d0..d3. `panel_tile` must already point at
// this oc_tile's slice of the weight panel; `st` gives its (ky,kx,icc)
// strides. `read_offset = x.border - conv.pad_h` maps output coords to padded
// input coords.
static inline void vnni_oc4_dots(
    const uint8_t* base_in, int W_PAD_IN, int C_IN,
    const int8_t* panel_tile, const PanelStrides& st,
    int IC_CHUNKS, int KH, int KW, int oh, int ow, int stride, int read_offset,
    int32_t& d0, int32_t& d1, int32_t& d2, int32_t& d3)
{
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (int ky = 0; ky < KH; ++ky) {
        const int ih = oh * stride + ky + read_offset;
        const int8_t* panel_ky = panel_tile + (size_t)ky * st.st_ky;
        for (int kx = 0; kx < KW; ++kx) {
            const int iw = ow * stride + kx + read_offset;
            const uint8_t* a_row = base_in + ((size_t)ih * W_PAD_IN + iw) * C_IN;
            const int8_t* panel_kx = panel_ky + (size_t)kx * st.st_kx;
            for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                const __m256i a32 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + icc * K_CHUNK));
                const int8_t* wp = panel_kx + (size_t)icc * st.st_icc;
                const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 0*K_CHUNK));
                const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 1*K_CHUNK));
                const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 2*K_CHUNK));
                const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 3*K_CHUNK));
                acc0 = _mm256_dpbusd_epi32(acc0, a32, w0);
                acc1 = _mm256_dpbusd_epi32(acc1, a32, w1);
                acc2 = _mm256_dpbusd_epi32(acc2, a32, w2);
                acc3 = _mm256_dpbusd_epi32(acc3, a32, w3);
            }
        }
    }
    d0 = hsum_epi32(acc0); d1 = hsum_epi32(acc1);
    d2 = hsum_epi32(acc2); d3 = hsum_epi32(acc3);
}

// Like vnni_oc4_dots, but computes two adjacent output pixels (ow, ow+1) at
// once: for each loaded activation chunk, two overlapping a-vectors (offset
// by one pixel, i.e. C_IN bytes) are each dotted against the same four
// weight vectors, filling an 8-accumulator [2][OC_TILE] grid that amortizes
// weight loads across both pixels. Requires stride==1 (the two reads must be
// exactly one pixel apart). Results land in d[0][0..3] / d[1][0..3].
static inline void vnni_oc4_n2_dots(
    const uint8_t* base_in, int W_PAD_IN, int C_IN,
    const int8_t* panel_tile, const PanelStrides& st,
    int IC_CHUNKS, int KH, int KW, int oh, int ow, int read_offset,
    int32_t d[2][OC_TILE])
{
    __m256i acc[2][OC_TILE];
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < OC_TILE; ++j) acc[n][j] = _mm256_setzero_si256();

    for (int ky = 0; ky < KH; ++ky) {
        const int ih = oh + ky + read_offset;
        const int8_t* panel_ky = panel_tile + (size_t)ky * st.st_ky;
        for (int kx = 0; kx < KW; ++kx) {
            const int iw = ow + kx + read_offset;
            const uint8_t* a_row = base_in + ((size_t)ih * W_PAD_IN + iw) * C_IN;
            const int8_t* panel_kx = panel_ky + (size_t)kx * st.st_kx;
            for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                const size_t off = (size_t)icc * K_CHUNK;
                const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_row + off));
                const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_row + C_IN + off));
                const int8_t* wp = panel_kx + (size_t)icc * st.st_icc;
                const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 0*K_CHUNK));
                const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 1*K_CHUNK));
                const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 2*K_CHUNK));
                const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 3*K_CHUNK));
                acc[0][0] = _mm256_dpbusd_epi32(acc[0][0], a0, w0);
                acc[0][1] = _mm256_dpbusd_epi32(acc[0][1], a0, w1);
                acc[0][2] = _mm256_dpbusd_epi32(acc[0][2], a0, w2);
                acc[0][3] = _mm256_dpbusd_epi32(acc[0][3], a0, w3);
                acc[1][0] = _mm256_dpbusd_epi32(acc[1][0], a1, w0);
                acc[1][1] = _mm256_dpbusd_epi32(acc[1][1], a1, w1);
                acc[1][2] = _mm256_dpbusd_epi32(acc[1][2], a1, w2);
                acc[1][3] = _mm256_dpbusd_epi32(acc[1][3], a1, w3);
            }
        }
    }
    for (int n = 0; n < 2; ++n)
        for (int j = 0; j < OC_TILE; ++j) d[n][j] = hsum_epi32(acc[n][j]);
}

// Baseline dense q8 conv + fused BN + (optional) ReLU + requant, q8 in/out.
// Single-threaded, oc_tile-major loop nest calling vnni_oc4_dots per pixel
// and requantizing each lane with the scalar requantize_u8. `panel` is the
// build_oc4_k32_panel-packed weight buffer for this layer.
void q2q_conv_bn_requant(
    const QTensorNHWC& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    bool relu, float out_scale, QTensorNHWC& y, int OH, int OW);

// Tuned dense q8 conv + fused BN + (optional) ReLU + requant, q8 in/out.
// OpenMP-parallel over the flattened (oc_tile, oh) task grid (collapse(2),
// disjoint per-task output regions -- no races); uses the N=2-unrolled
// vnni_oc4_n2_dots + SIMD epilogue_oc4_simd whenever stride==1 and OW is
// even, falling back to the single-pixel vnni_oc4_dots path otherwise.
// Numerically identical to q2q_conv_bn_requant, just faster.
void q2q_conv_bn_requant_apanel(
    const QTensorNHWC& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    bool relu, float out_scale, QTensorNHWC& y, int OH, int OW);

} // namespace q2q
