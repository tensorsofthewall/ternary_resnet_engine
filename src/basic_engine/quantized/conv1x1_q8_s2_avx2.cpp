#include "conv1x1_q8_s2_avx2.hpp"

#include <algorithm>
#include <immintrin.h>
#include <omp.h>

bool can_use_q8_1x1_s2_avx2(const Conv2DParams& conv) {
    return conv.kh == 1 &&
           conv.kw == 1 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.pad_h == 0 &&
           conv.pad_w == 0 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 2 == 0;
}

namespace {

// Extracts bytes 0,2,4,6,8,10,12,14 from a 16-byte load into positions 0-7.
// Used to read 8 stride-2 INT8 values as a contiguous sequence.
alignas(16) const int8_t STRIDE2_SHUF[16] = {
    0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1
};

// Masks for tail stores: tail_masks[n] writes the first n of 8 FP32 lanes.
alignas(32) const int32_t TAIL_MASKS[9][8] = {
    { 0,  0,  0,  0,  0,  0,  0,  0},
    {-1,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1,  0,  0,  0,  0,  0,  0},
    {-1, -1, -1,  0,  0,  0,  0,  0},
    {-1, -1, -1, -1,  0,  0,  0,  0},
    {-1, -1, -1, -1, -1,  0,  0,  0},
    {-1, -1, -1, -1, -1, -1,  0,  0},
    {-1, -1, -1, -1, -1, -1, -1,  0},
    {-1, -1, -1, -1, -1, -1, -1, -1},
};

} // namespace

void conv1x1_q8_s2_bn_avx2(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    int out_h, int out_w,
    Tensor& y_out
) {
    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();
    const int xq_h = xq.h;  // H (no top/bottom padding)
    const int xq_w = xq.w;  // W + extra_right_pad (safe over-read region)

    const __m128i shuf = _mm_load_si128(reinterpret_cast<const __m128i*>(STRIDE2_SHUF));
    const __m256i zero_i = _mm256_setzero_si256();

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock2PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const __m256 scale0 = _mm256_set1_ps(xq.scale * bn.scale[static_cast<size_t>(oc)]);
        const __m256 scale1 = _mm256_set1_ps(xq.scale * bn.scale[static_cast<size_t>(oc + 1)]);
        const __m256 bias0  = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc)]);
        const __m256 bias1  = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 1)]);

        for (int oh = 0; oh < out_h; ++oh) {
            const int ih = oh * 2;

            for (int ow = 0; ow < out_w; ow += 8) {
                const int iw = ow * 2;
                __m256i acc0 = zero_i, acc1 = zero_i;

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0, w1 = bucket.w1;
                    for (const OCBlock2Action& action : bucket.actions) {
                        const ptrdiff_t ic = static_cast<ptrdiff_t>(action.ic);
                        // action.ky == 0, action.kx == 0 (1×1 kernel)
                        const __m128i raw = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(
                                data + ic * xq_h * xq_w + ih * xq_w + iw
                            )
                        );
                        const __m256i x8 = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(raw, shuf));

                        if (w0 == 1)       acc0 = _mm256_add_epi32(acc0, x8);
                        else if (w0 == -1) acc0 = _mm256_sub_epi32(acc0, x8);
                        if (w1 == 1)       acc1 = _mm256_add_epi32(acc1, x8);
                        else if (w1 == -1) acc1 = _mm256_sub_epi32(acc1, x8);
                    }
                }

                const __m256 y0 = _mm256_add_ps(
                    _mm256_mul_ps(_mm256_cvtepi32_ps(acc0), scale0), bias0);
                const __m256 y1 = _mm256_add_ps(
                    _mm256_mul_ps(_mm256_cvtepi32_ps(acc1), scale1), bias1);

                const int tile_w = std::min(8, out_w - ow);
                if (tile_w == 8) {
                    _mm256_storeu_ps(y_out.ptr(oc,     oh, ow), y0);
                    _mm256_storeu_ps(y_out.ptr(oc + 1, oh, ow), y1);
                } else {
                    const __m256i vmask = _mm256_load_si256(
                        reinterpret_cast<const __m256i*>(TAIL_MASKS[tile_w]));
                    _mm256_maskstore_ps(y_out.ptr(oc,     oh, ow), vmask, y0);
                    _mm256_maskstore_ps(y_out.ptr(oc + 1, oh, ow), vmask, y1);
                }
            }
        }
    }
}
