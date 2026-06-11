#include "im2col_gemm.hpp"

#include "nhwc_common.hpp"

#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <stdexcept>

namespace nhwc_block32 {

PackedBPanel build_packed_b_panel(const Conv3x3S1P1PanelPlan& plan) {
    PackedBPanel out;
    out.out_channels = plan.out_channels;
    const int n_oc_tiles = plan.out_channels / 4;
    const int k_blocks = plan.ic_blocks * 9;
    out.K_pad = k_blocks * 32;
    out.data = AlignedBuffer32<int8_t>(static_cast<size_t>(n_oc_tiles) * static_cast<size_t>(k_blocks) * 4 * 32);

    for (int t = 0; t < n_oc_tiles; ++t) {
        const auto& wt = plan.w[static_cast<size_t>(t)];
        for (int kb = 0; kb < k_blocks; ++kb) {
            for (int lane = 0; lane < 4; ++lane) {
                std::memcpy(const_cast<int8_t*>(out.block(t, kb, lane)), wt[static_cast<size_t>(kb)][static_cast<size_t>(lane)].data(), 32);
            }
        }
    }
    return out;
}

Im2colMatrix build_im2col_3x3s1p1(const q2q::QTensorNHWC& in, int ic_blocks, int OH, int OW) {
    Im2colMatrix out;
    const int k_blocks = ic_blocks * 9;
    out.K_pad = k_blocks * 32;
    out.N = OH * OW;
    out.data = AlignedBuffer32<uint8_t>(static_cast<size_t>(out.N) * static_cast<size_t>(out.K_pad));

    const int C = ic_blocks * BLOCK32;
    const int read_offset = in.border - 1;  // pad == 1

    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            const int n = oh * OW + ow;
            uint8_t* arow = const_cast<uint8_t*>(out.row(n));
            for (int ky = 0; ky < 3; ++ky) {
                const int ih = oh + ky + read_offset;
                for (int kx = 0; kx < 3; ++kx) {
                    const int iw = ow + kx + read_offset;
                    const size_t row_base = (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C);
                    for (int icb = 0; icb < ic_blocks; ++icb) {
                        const int kidx = ky * 3 * ic_blocks + kx * ic_blocks + icb;
                        const uint8_t* src = &in.data[row_base + static_cast<size_t>(icb) * BLOCK32];
                        std::memcpy(arow + static_cast<size_t>(kidx) * 32, src, 32);
                    }
                }
            }
        }
    }
    return out;
}

namespace {

// 32-wide u8 x i8 dot product, both operands already loaded as __m256i,
// without the final horizontal sum (see nhwc_q8_dot_avx2.hpp for the
// memory-operand variants).
inline __m256i dot_u8_i8_32_both_preloaded_avx2(__m256i va, __m256i vw) {
    const __m256i prod16 = _mm256_maddubs_epi16(va, vw);
    const __m256i ones16 = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(prod16, ones16);
}

// GEMM-style OC4 microkernel: oc_tile-major outer loop (weight panel for one
// oc_tile is reused across all N output pixels), N_TILE adjacent im2col rows
// per inner iteration, vector accumulators kept across all K_pad/32 blocks
// and hsum'd once at the end.
template <int N_TILE>
void gemm_oc4_avx2_impl(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    (void)OH;
    if (A.N % N_TILE != 0) {
        throw std::runtime_error("gemm_oc4_avx2_impl: N must be a multiple of N_TILE");
    }
    const int k_blocks = B.K_pad / 32;
    const int n_oc_tiles = B.out_channels / 4;
    const float inv_out_scale = 1.0f / out.scale;

    for (int t = 0; t < n_oc_tiles; ++t) {
        float s[4], bb[4];
        for (int k = 0; k < 4; ++k) {
            s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * in_scale;
            bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
        }

        for (int n0 = 0; n0 < A.N; n0 += N_TILE) {
            __m256i acc[4][N_TILE];
            for (int lane = 0; lane < 4; ++lane) {
                for (int p = 0; p < N_TILE; ++p) acc[lane][p] = _mm256_setzero_si256();
            }

            for (int kb = 0; kb < k_blocks; ++kb) {
                __m256i va[N_TILE];
                for (int p = 0; p < N_TILE; ++p) {
                    va[p] = _mm256_load_si256(reinterpret_cast<const __m256i*>(A.row(n0 + p) + kb * 32));
                }
                for (int lane = 0; lane < 4; ++lane) {
                    const __m256i vw = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.block(t, kb, lane)));
                    for (int p = 0; p < N_TILE; ++p) {
                        acc[lane][p] = _mm256_add_epi32(acc[lane][p], dot_u8_i8_32_both_preloaded_avx2(va[p], vw));
                    }
                }
            }

            for (int p = 0; p < N_TILE; ++p) {
                int32_t d[4];
                for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane][p]);
                const int n = n0 + p;
                const int oh = n / OW;
                const int ow = n % OW;
                uint8_t* out_px = out.data.data()
                    + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow + out.border)) * static_cast<size_t>(out.C))
                    + static_cast<size_t>(t * 4);
                epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
            }
        }
    }
}

// Step 6: panel im2col GEMM. Builds a tile_width x K_pad activation panel for
// each tile of adjacent output columns directly from the q8 NHWC input, then
// runs the OC4 microkernel for every oc_tile against that panel. The main
// loop processes N_TILE-wide tiles; if OW is not a multiple of N_TILE, the
// remaining columns of each row are processed one at a time (tile_width=1)
// using the same panel buffer and accumulator arrays.
template <int N_TILE>
void panel_im2col_gemm_oc4_avx2_impl(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int k_blocks = ic_blocks * 9;
    if (B.K_pad / 32 != k_blocks) {
        throw std::runtime_error("panel_im2col_gemm_oc4_avx2_impl: ic_blocks mismatch with B");
    }
    const int C = ic_blocks * BLOCK32;
    const int read_offset = in.border - 1;  // pad == 1
    const int n_oc_tiles = B.out_channels / 4;
    const float in_scale = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    // Parallelize over output rows -- each row reads an overlapping
    // (read-only, safe) band of `in` and writes a disjoint output row, so
    // there are no races. `panel` is per-thread scratch, allocated once per
    // thread outside the `omp for`.
    #pragma omp parallel
    {
        AlignedBuffer32<uint8_t> panel(static_cast<size_t>(N_TILE) * static_cast<size_t>(B.K_pad));

        auto process_tile = [&](int oh, int ow0, int tile_width) {
            // Build A_panel[tile_width][K_pad] for output columns [ow0, ow0+tile_width).
            for (int ky = 0; ky < 3; ++ky) {
                const int ih = oh + ky + read_offset;
                for (int kx = 0; kx < 3; ++kx) {
                    for (int icb = 0; icb < ic_blocks; ++icb) {
                        const int kidx = ky * 3 * ic_blocks + kx * ic_blocks + icb;
                        for (int p = 0; p < tile_width; ++p) {
                            const int iw = ow0 + p + kx + read_offset;
                            const size_t src_off = (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
                                + static_cast<size_t>(icb) * BLOCK32;
                            uint8_t* dst = panel.data() + (static_cast<size_t>(p) * static_cast<size_t>(B.K_pad)) + static_cast<size_t>(kidx) * 32;
                            std::memcpy(dst, &in.data[src_off], 32);
                        }
                    }
                }
            }

            uint8_t* out_row = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow0 + out.border)) * static_cast<size_t>(out.C));

            for (int t = 0; t < n_oc_tiles; ++t) {
                float s[4], bb[4];
                for (int k = 0; k < 4; ++k) {
                    s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * in_scale;
                    bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
                }

                __m256i acc[4][N_TILE];
                for (int lane = 0; lane < 4; ++lane) {
                    for (int p = 0; p < tile_width; ++p) acc[lane][p] = _mm256_setzero_si256();
                }

                for (int kb = 0; kb < k_blocks; ++kb) {
                    __m256i va[N_TILE];
                    for (int p = 0; p < tile_width; ++p) {
                        va[p] = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel.data() + static_cast<size_t>(p) * static_cast<size_t>(B.K_pad) + kb * 32));
                    }
                    for (int lane = 0; lane < 4; ++lane) {
                        const __m256i vw = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.block(t, kb, lane)));
                        for (int p = 0; p < tile_width; ++p) {
                            acc[lane][p] = _mm256_add_epi32(acc[lane][p], dot_u8_i8_32_both_preloaded_avx2(va[p], vw));
                        }
                    }
                }

                for (int p = 0; p < tile_width; ++p) {
                    int32_t d[4];
                    for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane][p]);
                    uint8_t* out_px = out_row + static_cast<size_t>(p) * static_cast<size_t>(out.C) + static_cast<size_t>(t * 4);
                    epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
                }
            }
        };

        #pragma omp for schedule(static)
        for (int oh = 0; oh < OH; ++oh) {
            int ow0 = 0;
            for (; ow0 + N_TILE <= OW; ow0 += N_TILE) {
                process_tile(oh, ow0, N_TILE);
            }
            for (; ow0 < OW; ++ow0) {
                process_tile(oh, ow0, 1);
            }
        }
    }
}

// Phase 16 Step 3: like panel_im2col_gemm_oc4_avx2_impl but for 3x3/stride2/
// pad1 transition convs. Only the input-coordinate computation differs
// (ih = oh*2 + ky + read_offset, iw = (ow0+p)*2 + kx + read_offset); the
// panel layout, GEMM accumulation, and epilogue are identical.
template <int N_TILE>
void panel_im2col_gemm_oc4_stride2_avx2_impl(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int k_blocks = ic_blocks * 9;
    if (B.K_pad / 32 != k_blocks) {
        throw std::runtime_error("panel_im2col_gemm_oc4_stride2_avx2_impl: ic_blocks mismatch with B");
    }
    const int C = ic_blocks * BLOCK32;
    const int read_offset = in.border - 1;  // pad == 1
    const int n_oc_tiles = B.out_channels / 4;
    const float in_scale = in.scale;
    const float inv_out_scale = 1.0f / out.scale;

    // Parallelize over output rows -- same race-free reasoning as the
    // stride-1 kernel above (overlapping reads from `in`, disjoint writes
    // to `out`, per-thread `panel` scratch).
    #pragma omp parallel
    {
        AlignedBuffer32<uint8_t> panel(static_cast<size_t>(N_TILE) * static_cast<size_t>(B.K_pad));

        auto process_tile = [&](int oh, int ow0, int tile_width) {
            // Build A_panel[tile_width][K_pad] for output columns [ow0, ow0+tile_width).
            for (int ky = 0; ky < 3; ++ky) {
                const int ih = oh * 2 + ky + read_offset;
                for (int kx = 0; kx < 3; ++kx) {
                    for (int icb = 0; icb < ic_blocks; ++icb) {
                        const int kidx = ky * 3 * ic_blocks + kx * ic_blocks + icb;
                        for (int p = 0; p < tile_width; ++p) {
                            const int iw = (ow0 + p) * 2 + kx + read_offset;
                            const size_t src_off = (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)
                                + static_cast<size_t>(icb) * BLOCK32;
                            uint8_t* dst = panel.data() + (static_cast<size_t>(p) * static_cast<size_t>(B.K_pad)) + static_cast<size_t>(kidx) * 32;
                            std::memcpy(dst, &in.data[src_off], 32);
                        }
                    }
                }
            }

            uint8_t* out_row = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(ow0 + out.border)) * static_cast<size_t>(out.C));

            for (int t = 0; t < n_oc_tiles; ++t) {
                float s[4], bb[4];
                for (int k = 0; k < 4; ++k) {
                    s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * in_scale;
                    bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
                }

                __m256i acc[4][N_TILE];
                for (int lane = 0; lane < 4; ++lane) {
                    for (int p = 0; p < tile_width; ++p) acc[lane][p] = _mm256_setzero_si256();
                }

                for (int kb = 0; kb < k_blocks; ++kb) {
                    __m256i va[N_TILE];
                    for (int p = 0; p < tile_width; ++p) {
                        va[p] = _mm256_load_si256(reinterpret_cast<const __m256i*>(panel.data() + static_cast<size_t>(p) * static_cast<size_t>(B.K_pad) + kb * 32));
                    }
                    for (int lane = 0; lane < 4; ++lane) {
                        const __m256i vw = _mm256_load_si256(reinterpret_cast<const __m256i*>(B.block(t, kb, lane)));
                        for (int p = 0; p < tile_width; ++p) {
                            acc[lane][p] = _mm256_add_epi32(acc[lane][p], dot_u8_i8_32_both_preloaded_avx2(va[p], vw));
                        }
                    }
                }

                for (int p = 0; p < tile_width; ++p) {
                    int32_t d[4];
                    for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane][p]);
                    uint8_t* out_px = out_row + static_cast<size_t>(p) * static_cast<size_t>(out.C) + static_cast<size_t>(t * 4);
                    epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
                }
            }
        };

        #pragma omp for schedule(static)
        for (int oh = 0; oh < OH; ++oh) {
            int ow0 = 0;
            for (; ow0 + N_TILE <= OW; ow0 += N_TILE) {
                process_tile(oh, ow0, N_TILE);
            }
            for (; ow0 < OW; ++ow0) {
                process_tile(oh, ow0, 1);
            }
        }
    }
}

} // namespace

void gemm_oc4_n1_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    gemm_oc4_avx2_impl<1>(B, A, bn, in_scale, relu, out, OH, OW);
}

void gemm_oc4_n2_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    gemm_oc4_avx2_impl<2>(B, A, bn, in_scale, relu, out, OH, OW);
}

void gemm_oc4_n4_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    gemm_oc4_avx2_impl<4>(B, A, bn, in_scale, relu, out, OH, OW);
}

void gemm_oc4_n8_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    gemm_oc4_avx2_impl<8>(B, A, bn, in_scale, relu, out, OH, OW);
}

void panel_im2col_gemm_oc4_n1_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    panel_im2col_gemm_oc4_avx2_impl<1>(B, in, ic_blocks, bn, relu, out, OH, OW);
}

void panel_im2col_gemm_oc4_n2_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    panel_im2col_gemm_oc4_avx2_impl<2>(B, in, ic_blocks, bn, relu, out, OH, OW);
}

void panel_im2col_gemm_oc4_n4_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    panel_im2col_gemm_oc4_avx2_impl<4>(B, in, ic_blocks, bn, relu, out, OH, OW);
}

void panel_im2col_gemm_oc4_n8_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    panel_im2col_gemm_oc4_avx2_impl<8>(B, in, ic_blocks, bn, relu, out, OH, OW);
}

void panel_im2col_gemm_oc4_n2_stride2_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    panel_im2col_gemm_oc4_stride2_avx2_impl<2>(B, in, ic_blocks, bn, relu, out, OH, OW);
}

void panel_im2col_gemm_oc4_n4_stride2_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    panel_im2col_gemm_oc4_stride2_avx2_impl<4>(B, in, ic_blocks, bn, relu, out, OH, OW);
}

} // namespace nhwc_block32
