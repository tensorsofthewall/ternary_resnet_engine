#include "q2q_conv1_vnni.hpp"

#include "q2q_common.hpp"
#include "q2q_epilogue.hpp"

#include <omp.h>

#include <vector>

namespace q2q {

void q2q_conv1_im2col_bn_relu_requant(
    const QTensorNHWC& x, const Conv1Panel& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, KH = conv.kh, KW = conv.kw;
    const int stride = conv.stride_h;
    const int K_FULL = IC * KH * KW;
    const int K_PAD = panel.K_PAD;
    const int IC_CHUNKS = panel.IC_CHUNKS;
    const int OC_TILES = panel.OC_TILES;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const int b = x.border;

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data.data();
    uint8_t*       base_out   = y.data.data();

    // Precompute per-OC scale/bias once -- shared, read-only across threads.
    std::vector<float> s_all((size_t)OC_TILES * OC_TILE), b_all((size_t)OC_TILES * OC_TILE);
    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        const int oc_base = oc_tile * OC_TILE;
        for (int j = 0; j < OC_TILE; ++j) {
            s_all[oc_base + j] = bn.scale[oc_base + j] * dq;
            b_all[oc_base + j] = bn.bias[oc_base + j];
        }
    }

    // Parallelize over output rows (M = OH*OW = 12544 >> 196, so a
    // row-granularity m_tile gives every thread plenty of independent work).
    // Each thread packs its OWN a_panel buffer once (outside the #pragma omp
    // for) and reuses it across all rows it's assigned -- "thread-local A-panel,
    // not shared, no malloc inside the timed loop" per spec: this allocates
    // K_PAD+4 (~164) bytes once per thread per call, not per pixel.
    #pragma omp parallel
    {
        std::vector<uint8_t> a_panel(K_PAD + 4, uint8_t(0));  // +4 slack for the 4-byte gather tail write

        #pragma omp for schedule(static)
        for (int oh = 0; oh < OH; ++oh) {
            const int ih0 = oh * stride - conv.pad_h + b;
            uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C;
            for (int ow = 0; ow < OW; ++ow) {
                const int iw0 = ow * stride - conv.pad_w + b;
                conv1_gather_fast(base_in, x.W, x.C, ih0, iw0, KH, KW, IC, a_panel.data());
                for (int kg = K_FULL; kg < K_PAD; ++kg) a_panel[kg] = 0;

                uint8_t* out_px = out_row + (size_t)ow * y.C;
                for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
                    const int oc_base = oc_tile * OC_TILE;
                    __m256i acc0 = _mm256_setzero_si256();
                    __m256i acc1 = _mm256_setzero_si256();
                    __m256i acc2 = _mm256_setzero_si256();
                    __m256i acc3 = _mm256_setzero_si256();
                    const int8_t* panel_tile = base_panel + (size_t)oc_tile * IC_CHUNKS * OC_TILE * K_CHUNK;
                    for (int icc = 0; icc < IC_CHUNKS; ++icc) {
                        const __m256i a32 = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(a_panel.data() + (size_t)icc * K_CHUNK));
                        const int8_t* wp = panel_tile + (size_t)icc * OC_TILE * K_CHUNK;
                        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 0*K_CHUNK));
                        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 1*K_CHUNK));
                        const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 2*K_CHUNK));
                        const __m256i w3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + 3*K_CHUNK));
                        acc0 = _mm256_dpbusd_epi32(acc0, a32, w0);
                        acc1 = _mm256_dpbusd_epi32(acc1, a32, w1);
                        acc2 = _mm256_dpbusd_epi32(acc2, a32, w2);
                        acc3 = _mm256_dpbusd_epi32(acc3, a32, w3);
                    }
                    const int32_t d0 = hsum_epi32(acc0), d1 = hsum_epi32(acc1);
                    const int32_t d2 = hsum_epi32(acc2), d3 = hsum_epi32(acc3);

                    epilogue_oc4_simd(d0, d1, d2, d3, &s_all[oc_base], &b_all[oc_base],
                                      /*relu=*/true, inv_out_scale, out_px + oc_base);
                }
            }
        }
    }
}

} // namespace q2q
