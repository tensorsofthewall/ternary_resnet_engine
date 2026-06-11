#include "block32_layer234_opt.hpp"

#include "nhwc_common.hpp"
#include "nhwc_q8_dot_avx2.hpp"

#include <immintrin.h>
#include <algorithm>
#include <stdexcept>

namespace nhwc_block32 {

Conv3x3S1P1PanelPlan build_conv3x3s1p1_panel_plan(const Block32ConvPlan& plan) {
    if (plan.stride_h != 1 || plan.stride_w != 1) {
        throw std::runtime_error("build_conv3x3s1p1_panel_plan: requires stride 1: " + plan.name);
    }
    return build_conv3x3_panel_plan_strided(plan);
}

Conv3x3S1P1PanelPlan build_conv3x3_panel_plan_strided(const Block32ConvPlan& plan) {
    if (plan.kh != 3 || plan.kw != 3) {
        throw std::runtime_error("build_conv3x3_panel_plan_strided: requires kh==kw==3: " + plan.name);
    }
    if (plan.pad_h != 1 || plan.pad_w != 1) {
        throw std::runtime_error("build_conv3x3_panel_plan_strided: requires pad 1: " + plan.name);
    }
    if (plan.out_channels % 4 != 0) {
        throw std::runtime_error("build_conv3x3_panel_plan_strided: out_channels must be a multiple of 4: " + plan.name);
    }
    require_uniform_block_layout(plan);

    const auto& blocks0 = plan.oc_plans.front().blocks;
    const int K = 9 * plan.ic_blocks;
    if (static_cast<int>(blocks0.size()) != K) {
        throw std::runtime_error("build_conv3x3_panel_plan_strided: expected 9*ic_blocks blocks: " + plan.name);
    }
    for (int kidx = 0; kidx < K; ++kidx) {
        const int ky = kidx / (3 * plan.ic_blocks);
        const int rem = kidx % (3 * plan.ic_blocks);
        const int kx = rem / plan.ic_blocks;
        const int icb = rem % plan.ic_blocks;
        const Block32& b = blocks0[static_cast<size_t>(kidx)];
        if (b.ky != ky || b.kx != kx || b.ic_block != icb) {
            throw std::runtime_error("build_conv3x3_panel_plan_strided: unexpected block order: " + plan.name);
        }
    }

    Conv3x3S1P1PanelPlan out;
    out.out_channels = plan.out_channels;
    out.ic_blocks = plan.ic_blocks;
    const int n_oc_tiles = plan.out_channels / 4;
    out.w.resize(static_cast<size_t>(n_oc_tiles));
    for (int t = 0; t < n_oc_tiles; ++t) {
        out.w[static_cast<size_t>(t)].resize(static_cast<size_t>(K));
        for (int kidx = 0; kidx < K; ++kidx) {
            for (int lane = 0; lane < 4; ++lane) {
                const Block32& b = plan.oc_plans[static_cast<size_t>(t * 4 + lane)].blocks[static_cast<size_t>(kidx)];
                std::copy(b.w, b.w + BLOCK32, out.w[static_cast<size_t>(t)][static_cast<size_t>(kidx)][static_cast<size_t>(lane)].begin());
            }
        }
    }
    return out;
}

Conv1x1PanelPlan build_conv1x1_panel_plan(const Block32ConvPlan& plan) {
    if (plan.kh != 1 || plan.kw != 1) {
        throw std::runtime_error("build_conv1x1_panel_plan: requires kh==kw==1: " + plan.name);
    }
    if (plan.stride_h != 2 || plan.stride_w != 2) {
        throw std::runtime_error("build_conv1x1_panel_plan: requires stride 2: " + plan.name);
    }
    if (plan.pad_h != 0 || plan.pad_w != 0) {
        throw std::runtime_error("build_conv1x1_panel_plan: requires pad 0: " + plan.name);
    }
    if (plan.out_channels % 4 != 0) {
        throw std::runtime_error("build_conv1x1_panel_plan: out_channels must be a multiple of 4: " + plan.name);
    }
    require_uniform_block_layout(plan);

    const auto& blocks0 = plan.oc_plans.front().blocks;
    if (static_cast<int>(blocks0.size()) != plan.ic_blocks) {
        throw std::runtime_error("build_conv1x1_panel_plan: expected ic_blocks blocks: " + plan.name);
    }
    for (int icb = 0; icb < plan.ic_blocks; ++icb) {
        const Block32& b = blocks0[static_cast<size_t>(icb)];
        if (b.ky != 0 || b.kx != 0 || b.ic_block != icb) {
            throw std::runtime_error("build_conv1x1_panel_plan: unexpected block order: " + plan.name);
        }
    }

    Conv1x1PanelPlan out;
    out.out_channels = plan.out_channels;
    out.ic_blocks = plan.ic_blocks;
    const int n_oc_tiles = plan.out_channels / 4;
    out.w.resize(static_cast<size_t>(n_oc_tiles));
    for (int t = 0; t < n_oc_tiles; ++t) {
        out.w[static_cast<size_t>(t)].resize(static_cast<size_t>(plan.ic_blocks));
        for (int icb = 0; icb < plan.ic_blocks; ++icb) {
            for (int lane = 0; lane < 4; ++lane) {
                const Block32& b = plan.oc_plans[static_cast<size_t>(t * 4 + lane)].blocks[static_cast<size_t>(icb)];
                std::copy(b.w, b.w + BLOCK32, out.w[static_cast<size_t>(t)][static_cast<size_t>(icb)][static_cast<size_t>(lane)].begin());
            }
        }
    }
    return out;
}

namespace {

// OC-major, flat weight-panel-addressed 3x3/stride1/pad1 block32 conv.
// IC_BLOCKS is a compile-time constant (4 for layer2's 128-channel convs, 8
// for layer3's 256-channel convs), giving the compiler a fixed K=9*IC_BLOCKS
// reduction trip count to unroll. N_TILE adjacent output columns are
// processed per inner iteration (4*N_TILE accumulators total), each hsummed
// exactly once after the K-loop.
template <int IC_BLOCKS, int N_TILE>
void conv3x3s1p1_oc4_avx2_impl(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    if (plan.ic_blocks != IC_BLOCKS) {
        throw std::runtime_error("conv3x3s1p1_oc4_avx2_impl: ic_blocks mismatch");
    }
    if (OW % N_TILE != 0) {
        throw std::runtime_error("conv3x3s1p1_oc4_avx2_impl: OW must be a multiple of N_TILE");
    }

    constexpr int K = 9 * IC_BLOCKS;
    constexpr int C = IC_BLOCKS * BLOCK32;
    const int read_offset = in.border - 1;  // pad == 1
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;
    const int n_oc_tiles = plan.out_channels / 4;

    for (int t = 0; t < n_oc_tiles; ++t) {
        const auto& wt = plan.w[static_cast<size_t>(t)];
        float s[4], bb[4];
        for (int k = 0; k < 4; ++k) {
            s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * dq;
            bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
        }

        for (int oh = 0; oh < OH; ++oh) {
            const int ih0 = oh + read_offset;
            uint8_t* out_row = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(out.border)) * static_cast<size_t>(out.C))
                + static_cast<size_t>(t * 4);

            for (int ow = 0; ow < OW; ow += N_TILE) {
                __m256i acc[4][N_TILE];
                for (int lane = 0; lane < 4; ++lane) {
                    for (int p = 0; p < N_TILE; ++p) acc[lane][p] = _mm256_setzero_si256();
                }

                const int iw0 = ow + read_offset;
                for (int kidx = 0; kidx < K; ++kidx) {
                    const int ky = kidx / (3 * IC_BLOCKS);
                    const int rem = kidx % (3 * IC_BLOCKS);
                    const int kx = rem / IC_BLOCKS;
                    const int icb = rem % IC_BLOCKS;
                    const int ih = ih0 + ky;
                    const size_t row_base =
                        static_cast<size_t>(ih) * static_cast<size_t>(in.W) * static_cast<size_t>(C)
                        + static_cast<size_t>(icb) * BLOCK32;
                    const auto& wk = wt[static_cast<size_t>(kidx)];

                    for (int p = 0; p < N_TILE; ++p) {
                        const int iw = iw0 + kx + p;
                        const uint8_t* a = &in.data[row_base + static_cast<size_t>(iw) * static_cast<size_t>(C)];
                        for (int lane = 0; lane < 4; ++lane) {
                            acc[lane][p] = _mm256_add_epi32(acc[lane][p], dot_u8_i8_32_partial_avx2(a, wk[static_cast<size_t>(lane)].data()));
                        }
                    }
                }

                for (int p = 0; p < N_TILE; ++p) {
                    int32_t d[4];
                    for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane][p]);
                    uint8_t* out_px = out_row + static_cast<size_t>(ow + p) * static_cast<size_t>(out.C);
                    epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
                }
            }
        }
    }
}

// Same OC-major, flat-weight-panel-addressed 3x3/stride1/pad1 block32 conv as
// conv3x3s1p1_oc4_avx2_impl, but for odd OW: processes floor(OW/2) N2 column
// pairs (8 accumulators, hsummed once each) followed by a single N1 tail
// column (4 accumulators) when OW is odd. Used for layer4's 7x7 convs.
template <int IC_BLOCKS>
void conv3x3s1p1_oc4_n2tail_avx2_impl(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    if (plan.ic_blocks != IC_BLOCKS) {
        throw std::runtime_error("conv3x3s1p1_oc4_n2tail_avx2_impl: ic_blocks mismatch");
    }

    constexpr int K = 9 * IC_BLOCKS;
    constexpr int C = IC_BLOCKS * BLOCK32;
    const int read_offset = in.border - 1;  // pad == 1
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;
    const int n_oc_tiles = plan.out_channels / 4;
    const int OW2 = OW - (OW % 2);  // largest even <= OW

    for (int t = 0; t < n_oc_tiles; ++t) {
        const auto& wt = plan.w[static_cast<size_t>(t)];
        float s[4], bb[4];
        for (int k = 0; k < 4; ++k) {
            s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * dq;
            bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
        }

        for (int oh = 0; oh < OH; ++oh) {
            const int ih0 = oh + read_offset;
            uint8_t* out_row = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(out.border)) * static_cast<size_t>(out.C))
                + static_cast<size_t>(t * 4);

            for (int ow = 0; ow < OW2; ow += 2) {
                __m256i acc[4][2];
                for (int lane = 0; lane < 4; ++lane) {
                    acc[lane][0] = _mm256_setzero_si256();
                    acc[lane][1] = _mm256_setzero_si256();
                }

                const int iw0 = ow + read_offset;
                for (int kidx = 0; kidx < K; ++kidx) {
                    const int ky = kidx / (3 * IC_BLOCKS);
                    const int rem = kidx % (3 * IC_BLOCKS);
                    const int kx = rem / IC_BLOCKS;
                    const int icb = rem % IC_BLOCKS;
                    const int ih = ih0 + ky;
                    const size_t row_base =
                        static_cast<size_t>(ih) * static_cast<size_t>(in.W) * static_cast<size_t>(C)
                        + static_cast<size_t>(icb) * BLOCK32;
                    const auto& wk = wt[static_cast<size_t>(kidx)];

                    for (int p = 0; p < 2; ++p) {
                        const int iw = iw0 + kx + p;
                        const uint8_t* a = &in.data[row_base + static_cast<size_t>(iw) * static_cast<size_t>(C)];
                        for (int lane = 0; lane < 4; ++lane) {
                            acc[lane][p] = _mm256_add_epi32(acc[lane][p], dot_u8_i8_32_partial_avx2(a, wk[static_cast<size_t>(lane)].data()));
                        }
                    }
                }

                for (int p = 0; p < 2; ++p) {
                    int32_t d[4];
                    for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane][p]);
                    uint8_t* out_px = out_row + static_cast<size_t>(ow + p) * static_cast<size_t>(out.C);
                    epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
                }
            }

            // Tail: single output column when OW is odd.
            for (int ow = OW2; ow < OW; ++ow) {
                __m256i acc[4];
                for (int lane = 0; lane < 4; ++lane) acc[lane] = _mm256_setzero_si256();

                const int iw0 = ow + read_offset;
                for (int kidx = 0; kidx < K; ++kidx) {
                    const int ky = kidx / (3 * IC_BLOCKS);
                    const int rem = kidx % (3 * IC_BLOCKS);
                    const int kx = rem / IC_BLOCKS;
                    const int icb = rem % IC_BLOCKS;
                    const int ih = ih0 + ky;
                    const size_t row_base =
                        static_cast<size_t>(ih) * static_cast<size_t>(in.W) * static_cast<size_t>(C)
                        + static_cast<size_t>(icb) * BLOCK32;
                    const auto& wk = wt[static_cast<size_t>(kidx)];
                    const int iw = iw0 + kx;
                    const uint8_t* a = &in.data[row_base + static_cast<size_t>(iw) * static_cast<size_t>(C)];
                    for (int lane = 0; lane < 4; ++lane) {
                        acc[lane] = _mm256_add_epi32(acc[lane], dot_u8_i8_32_partial_avx2(a, wk[static_cast<size_t>(lane)].data()));
                    }
                }

                int32_t d[4];
                for (int lane = 0; lane < 4; ++lane) d[lane] = hsum_epi32(acc[lane]);
                uint8_t* out_px = out_row + static_cast<size_t>(ow) * static_cast<size_t>(out.C);
                epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
            }
        }
    }
}

} // namespace

void conv3x3s1p1_128x128_28x28_oc4_n1_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv3x3s1p1_oc4_avx2_impl<4, 1>(plan, bn, in, relu, out, OH, OW);
}

void conv3x3s1p1_128x128_28x28_oc4_n2_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv3x3s1p1_oc4_avx2_impl<4, 2>(plan, bn, in, relu, out, OH, OW);
}

void conv3x3s1p1_128x128_28x28_oc4_n4_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv3x3s1p1_oc4_avx2_impl<4, 4>(plan, bn, in, relu, out, OH, OW);
}

void conv3x3s1p1_256x256_14x14_oc4_n2_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv3x3s1p1_oc4_avx2_impl<8, 2>(plan, bn, in, relu, out, OH, OW);
}

void conv3x3s1p1_512x512_7x7_oc4_n1_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv3x3s1p1_oc4_avx2_impl<16, 1>(plan, bn, in, relu, out, OH, OW);
}

void conv3x3s1p1_512x512_7x7_oc4_n2_avx2(
    const Conv3x3S1P1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    conv3x3s1p1_oc4_n2tail_avx2_impl<16>(plan, bn, in, relu, out, OH, OW);
}

void conv1x1s2_block32_oc4_avx2(
    const Conv1x1PanelPlan& plan, const BNFusedParams& bn,
    const q2q::QTensorNHWC& in, bool relu, q2q::QTensorNHWC& out, int OH, int OW)
{
    const int IC_BLOCKS = plan.ic_blocks;
    const int C = IC_BLOCKS * BLOCK32;
    const int read_offset = in.border;  // pad == 0
    const float dq = in.scale;
    const float inv_out_scale = 1.0f / out.scale;
    const int n_oc_tiles = plan.out_channels / 4;

    for (int t = 0; t < n_oc_tiles; ++t) {
        const auto& wt = plan.w[static_cast<size_t>(t)];
        float s[4], bb[4];
        for (int k = 0; k < 4; ++k) {
            s[k] = bn.scale[static_cast<size_t>(t * 4 + k)] * dq;
            bb[k] = bn.bias[static_cast<size_t>(t * 4 + k)];
        }

        for (int oh = 0; oh < OH; ++oh) {
            const int ih = oh * 2 + read_offset;
            uint8_t* out_row = out.data.data()
                + ((static_cast<size_t>(oh + out.border) * static_cast<size_t>(out.W) + static_cast<size_t>(out.border)) * static_cast<size_t>(out.C))
                + static_cast<size_t>(t * 4);

            for (int ow = 0; ow < OW; ++ow) {
                const int iw = ow * 2 + read_offset;
                const uint8_t* a_base = &in.data[
                    (static_cast<size_t>(ih) * static_cast<size_t>(in.W) + static_cast<size_t>(iw)) * static_cast<size_t>(C)];

                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                for (int icb = 0; icb < IC_BLOCKS; ++icb) {
                    const uint8_t* a = a_base + static_cast<size_t>(icb) * BLOCK32;
                    const auto& wk = wt[static_cast<size_t>(icb)];
                    acc0 = _mm256_add_epi32(acc0, dot_u8_i8_32_partial_avx2(a, wk[0].data()));
                    acc1 = _mm256_add_epi32(acc1, dot_u8_i8_32_partial_avx2(a, wk[1].data()));
                    acc2 = _mm256_add_epi32(acc2, dot_u8_i8_32_partial_avx2(a, wk[2].data()));
                    acc3 = _mm256_add_epi32(acc3, dot_u8_i8_32_partial_avx2(a, wk[3].data()));
                }

                const int32_t d[4] = {
                    hsum_epi32(acc0), hsum_epi32(acc1), hsum_epi32(acc2), hsum_epi32(acc3),
                };

                uint8_t* out_px = out_row + static_cast<size_t>(ow) * static_cast<size_t>(out.C);
                epilogue_oc4_avx2(d, s, bb, relu, inv_out_scale, out_px);
            }
        }
    }
}

} // namespace nhwc_block32
