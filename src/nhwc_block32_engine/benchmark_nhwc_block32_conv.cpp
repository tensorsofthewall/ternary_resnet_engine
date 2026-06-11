// Microkernel benchmark for the AVX2 NHWC block32 ternary q8 engine.
//
// Implements portable AVX2-only (no AVX-VNNI) dot kernels for one conv layer
// (layer2.1.conv2: 128->128, 3x3, stride 1, pad 1, 28x28) using the
// Block32ConvPlan, with maddubs/madd dot products (nhwc_q8_dot_avx2.hpp)
// plus fused BN + requant (no ReLU, matching this conv's real usage as a
// residual-branch conv2):
//   - scalar reference
//   - OC1: one hsum_epi32 per (output pixel, block)
//   - OC4: four __m256i accumulators per output pixel, one hsum_epi32 per
//     (output pixel, output channel) after all blocks
// A scalar reference implementation over the same plan is used to verify
// exact output match for both AVX2 variants.
//
// Usage: ./benchmark_nhwc_block32_conv <model.bin>

#include "model_loader.hpp"
#include "model.hpp"
#include "nhwc_common.hpp"
#include "nhwc_q8_dot_avx2.hpp"
#include "block32_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace nhwc_block32;

namespace {

// Reference (scalar) and AVX2 conv3x3/stride1/pad1 over a Block32ConvPlan,
// reading from a padded NHWC uint8 `input` (Hp x Wp x C, border=1) and
// writing a logical H x W x OC uint8 `output`. Applies BN scale/bias (no
// ReLU) and requantizes to u8.
void conv3x3_s1_block32_scalar(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const std::vector<uint8_t>& input, int H, int W, int Wp,
    std::vector<uint8_t>& output)
{
    const float inv_out_scale = 1.0f / ACT_SCALE;
    const int OC = plan.out_channels;
    const int C = plan.in_channels;

    for (int oc = 0; oc < OC; ++oc) {
        const auto& blocks = plan.oc_plans[static_cast<size_t>(oc)].blocks;
        const float s = bn.scale[static_cast<size_t>(oc)];
        const float b = bn.bias[static_cast<size_t>(oc)];

        for (int oh = 0; oh < H; ++oh) {
            for (int ow = 0; ow < W; ++ow) {
                int32_t acc = 0;
                for (const Block32& blk : blocks) {
                    const size_t base =
                        (static_cast<size_t>(oh + blk.ky) * Wp + static_cast<size_t>(ow + blk.kx)) * C
                        + static_cast<size_t>(blk.ic_block) * BLOCK32;
                    for (int ci = 0; ci < BLOCK32; ++ci) {
                        acc += static_cast<int32_t>(input[base + static_cast<size_t>(ci)])
                             * static_cast<int32_t>(blk.w[ci]);
                    }
                }
                const float real = static_cast<float>(acc) * ACT_SCALE;
                const float bn_out = real * s + b;
                output[(static_cast<size_t>(oh) * W + static_cast<size_t>(ow)) * OC + static_cast<size_t>(oc)] =
                    requantize_u8(bn_out, inv_out_scale);
            }
        }
    }
}

void conv3x3_s1_block32_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const std::vector<uint8_t>& input, int H, int W, int Wp,
    std::vector<uint8_t>& output)
{
    const float inv_out_scale = 1.0f / ACT_SCALE;
    const int OC = plan.out_channels;
    const int C = plan.in_channels;

    for (int oc = 0; oc < OC; ++oc) {
        const auto& blocks = plan.oc_plans[static_cast<size_t>(oc)].blocks;
        const float s = bn.scale[static_cast<size_t>(oc)];
        const float b = bn.bias[static_cast<size_t>(oc)];

        for (int oh = 0; oh < H; ++oh) {
            for (int ow = 0; ow < W; ++ow) {
                int32_t acc = 0;
                for (const Block32& blk : blocks) {
                    const size_t base =
                        (static_cast<size_t>(oh + blk.ky) * Wp + static_cast<size_t>(ow + blk.kx)) * C
                        + static_cast<size_t>(blk.ic_block) * BLOCK32;
                    acc += dot_u8_i8_32_avx2(&input[base], blk.w);
                }
                const float real = static_cast<float>(acc) * ACT_SCALE;
                const float bn_out = real * s + b;
                output[(static_cast<size_t>(oh) * W + static_cast<size_t>(ow)) * OC + static_cast<size_t>(oc)] =
                    requantize_u8(bn_out, inv_out_scale);
            }
        }
    }
}

// OC4 variant: for each group of 4 output channels, walks the four oc_plans
// in lockstep (one shared (ky,kx,ic_block) sequence -- see
// require_uniform_block_layout) and accumulates each channel's running dot
// product as an __m256i partial sum (dot_u8_i8_32_partial_avx2), loading the
// activation block once and reusing it across all four channels. hsum_epi32
// runs only once per output pixel per channel, after all blocks have been
// accumulated -- not once per block as in the OC1 kernel.
void conv3x3_s1_block32_oc4_avx2(
    const Block32ConvPlan& plan, const BNFusedParams& bn,
    const std::vector<uint8_t>& input, int H, int W, int Wp,
    std::vector<uint8_t>& output)
{
    const float inv_out_scale = 1.0f / ACT_SCALE;
    const int OC = plan.out_channels;
    const int C = plan.in_channels;

    for (int oc0 = 0; oc0 < OC; oc0 += 4) {
        const auto& b0 = plan.oc_plans[static_cast<size_t>(oc0 + 0)].blocks;
        const auto& b1 = plan.oc_plans[static_cast<size_t>(oc0 + 1)].blocks;
        const auto& b2 = plan.oc_plans[static_cast<size_t>(oc0 + 2)].blocks;
        const auto& b3 = plan.oc_plans[static_cast<size_t>(oc0 + 3)].blocks;
        const size_t n = b0.size();

        const float s[4] = {
            bn.scale[static_cast<size_t>(oc0 + 0)], bn.scale[static_cast<size_t>(oc0 + 1)],
            bn.scale[static_cast<size_t>(oc0 + 2)], bn.scale[static_cast<size_t>(oc0 + 3)],
        };
        const float bb[4] = {
            bn.bias[static_cast<size_t>(oc0 + 0)], bn.bias[static_cast<size_t>(oc0 + 1)],
            bn.bias[static_cast<size_t>(oc0 + 2)], bn.bias[static_cast<size_t>(oc0 + 3)],
        };

        for (int oh = 0; oh < H; ++oh) {
            for (int ow = 0; ow < W; ++ow) {
                __m256i acc0 = _mm256_setzero_si256();
                __m256i acc1 = _mm256_setzero_si256();
                __m256i acc2 = _mm256_setzero_si256();
                __m256i acc3 = _mm256_setzero_si256();

                for (size_t i = 0; i < n; ++i) {
                    const Block32& blk = b0[i];
                    const size_t base =
                        (static_cast<size_t>(oh + blk.ky) * Wp + static_cast<size_t>(ow + blk.kx)) * C
                        + static_cast<size_t>(blk.ic_block) * BLOCK32;
                    const uint8_t* a = &input[base];

                    acc0 = _mm256_add_epi32(acc0, dot_u8_i8_32_partial_avx2(a, b0[i].w));
                    acc1 = _mm256_add_epi32(acc1, dot_u8_i8_32_partial_avx2(a, b1[i].w));
                    acc2 = _mm256_add_epi32(acc2, dot_u8_i8_32_partial_avx2(a, b2[i].w));
                    acc3 = _mm256_add_epi32(acc3, dot_u8_i8_32_partial_avx2(a, b3[i].w));
                }

                const int32_t d[4] = {
                    hsum_epi32(acc0), hsum_epi32(acc1), hsum_epi32(acc2), hsum_epi32(acc3),
                };

                uint8_t* out_px =
                    &output[(static_cast<size_t>(oh) * W + static_cast<size_t>(ow)) * OC + static_cast<size_t>(oc0)];
                for (int k = 0; k < 4; ++k) {
                    const float real = static_cast<float>(d[k]) * ACT_SCALE;
                    const float bn_out = real * s[k] + bb[k];
                    out_px[k] = requantize_u8(bn_out, inv_out_scale);
                }
            }
        }
    }
}

// The OC4 kernel above walks four oc_plans in lockstep using only b0[i]'s
// (ky,kx,ic_block) coordinates. This holds as long as every output channel's
// block list has the same length and (ky,kx,ic_block) sequence -- true here
// because build_block32_plan iterates all (ky,kx,ic_block) combos in
// a fixed order and (for this ternary model) never drops one as all-zero
// (see block32_plan.hpp). Throws if that assumption doesn't hold.
void require_uniform_block_layout(const Block32ConvPlan& plan) {
    const auto& blocks0 = plan.oc_plans.front().blocks;
    for (const OCPlan& oc_plan : plan.oc_plans) {
        if (oc_plan.blocks.size() != blocks0.size()) {
            throw std::runtime_error("require_uniform_block_layout: block-count mismatch across output channels");
        }
        for (size_t i = 0; i < blocks0.size(); ++i) {
            const Block32& a = blocks0[i];
            const Block32& b = oc_plan.blocks[i];
            if (a.ky != b.ky || a.kx != b.kx || a.ic_block != b.ic_block) {
                throw std::runtime_error("require_uniform_block_layout: (ky,kx,ic_block) sequence mismatch across output channels");
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    const Model model = load_model(argv[1]);
    const Conv2DParams& conv = model.convs.at("layer2.1.conv2");
    const BNFusedParams& bn = model.fused_bns.at("layer2.1.bn2");

    if (conv.kh != 3 || conv.kw != 3 || conv.stride_h != 1 || conv.stride_w != 1
        || conv.pad_h != 1 || conv.pad_w != 1) {
        throw std::runtime_error("benchmark_nhwc_block32_conv: unexpected layer2.1.conv2 shape");
    }

    const Block32ConvPlan plan = build_block32_plan(conv);
    require_uniform_block_layout(plan);

    const int H = 28, W = 28;
    const int border = 1;
    const int Hp = H + 2 * border, Wp = W + 2 * border;
    const int C = conv.in_channels;
    const int OC = conv.out_channels;

    std::cout << "=== benchmark_nhwc_block32_conv: layer2.1.conv2 (in=" << C
              << " out=" << OC << " 3x3 s=1, " << H << "x" << W << ") ===\n";

    // Random padded NHWC q8 input (border rows/cols left zero).
    std::vector<uint8_t> input(static_cast<size_t>(Hp) * Wp * C, 0);
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    const size_t idx =
                        (static_cast<size_t>(h + border) * Wp + static_cast<size_t>(w + border)) * C
                        + static_cast<size_t>(c);
                    input[idx] = static_cast<uint8_t>(dist(rng));
                }
            }
        }
    }

    std::vector<uint8_t> out_scalar(static_cast<size_t>(H) * W * OC);
    std::vector<uint8_t> out_avx2(static_cast<size_t>(H) * W * OC);
    std::vector<uint8_t> out_oc4(static_cast<size_t>(H) * W * OC);

    // ---- correctness check ----
    conv3x3_s1_block32_scalar(plan, bn, input, H, W, Wp, out_scalar);
    conv3x3_s1_block32_avx2(plan, bn, input, H, W, Wp, out_avx2);
    conv3x3_s1_block32_oc4_avx2(plan, bn, input, H, W, Wp, out_oc4);

    size_t mismatches = 0;
    int max_abs_diff = 0;
    for (size_t i = 0; i < out_scalar.size(); ++i) {
        const int diff = std::abs(static_cast<int>(out_scalar[i]) - static_cast<int>(out_avx2[i]));
        if (diff != 0) ++mismatches;
        max_abs_diff = std::max(max_abs_diff, diff);
    }
    std::cout << "Correctness (OC1 avx2): " << mismatches << " / " << out_scalar.size()
              << " mismatches, max_abs_diff=" << max_abs_diff << "\n";

    size_t mismatches_oc4 = 0;
    int max_abs_diff_oc4 = 0;
    for (size_t i = 0; i < out_scalar.size(); ++i) {
        const int diff = std::abs(static_cast<int>(out_scalar[i]) - static_cast<int>(out_oc4[i]));
        if (diff != 0) ++mismatches_oc4;
        max_abs_diff_oc4 = std::max(max_abs_diff_oc4, diff);
    }
    std::cout << "Correctness (OC4 avx2): " << mismatches_oc4 << " / " << out_scalar.size()
              << " mismatches, max_abs_diff=" << max_abs_diff_oc4 << "\n\n";

    // ---- timing ----
    constexpr int N_WARMUP = 20;
    constexpr int N_TIMED = 100;

    for (int i = 0; i < N_WARMUP; ++i) conv3x3_s1_block32_scalar(plan, bn, input, H, W, Wp, out_scalar);
    std::vector<double> t_scalar(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        const double t0 = now_ms();
        conv3x3_s1_block32_scalar(plan, bn, input, H, W, Wp, out_scalar);
        t_scalar[static_cast<size_t>(i)] = now_ms() - t0;
    }

    for (int i = 0; i < N_WARMUP; ++i) conv3x3_s1_block32_avx2(plan, bn, input, H, W, Wp, out_avx2);
    std::vector<double> t_avx2(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        const double t0 = now_ms();
        conv3x3_s1_block32_avx2(plan, bn, input, H, W, Wp, out_avx2);
        t_avx2[static_cast<size_t>(i)] = now_ms() - t0;
    }

    for (int i = 0; i < N_WARMUP; ++i) conv3x3_s1_block32_oc4_avx2(plan, bn, input, H, W, Wp, out_oc4);
    std::vector<double> t_oc4(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        const double t0 = now_ms();
        conv3x3_s1_block32_oc4_avx2(plan, bn, input, H, W, Wp, out_oc4);
        t_oc4[static_cast<size_t>(i)] = now_ms() - t0;
    }

    std::cout << "Timing (1 thread, " << N_TIMED << " runs):\n";
    print_stats("scalar  ", t_scalar, N_TIMED);
    print_stats("avx2 oc1", t_avx2, N_TIMED);
    print_stats("avx2 oc4", t_oc4, N_TIMED);

    return 0;
}
