#include "quantized/conv3x3_q8_pattern_ocblock2_avx2.hpp"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

#include <omp.h>

bool can_use_q8_ocblock2_spatial4_layer2_s2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 && conv.kw == 3 &&
           conv.stride_h == 2 && conv.stride_w == 2 &&
           conv.dilation_h == 1 && conv.dilation_w == 1 &&
           conv.in_channels == 64 && conv.out_channels == 128;
}

bool can_use_q8_ocblock2_spatial4_layer1_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 && conv.kw == 3 &&
           conv.stride_h == 1 && conv.stride_w == 1 &&
           conv.dilation_h == 1 && conv.dilation_w == 1 &&
           conv.in_channels == 64 && conv.out_channels == 64;
}

bool can_use_q8_ocblock2_spatial4_layer2_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 && conv.kw == 3 &&
           conv.stride_h == 1 && conv.stride_w == 1 &&
           conv.dilation_h == 1 && conv.dilation_w == 1 &&
           conv.in_channels == 128 && conv.out_channels == 128;
}

bool can_use_q8_ocblock2_spatial4_layer3_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 && conv.kw == 3 &&
           conv.stride_h == 1 && conv.stride_w == 1 &&
           conv.dilation_h == 1 && conv.dilation_w == 1 &&
           conv.in_channels == 256 && conv.out_channels == 256;
}

bool can_use_q8_ocblock2_spatial4_layer4_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 && conv.kw == 3 &&
           conv.stride_h == 1 && conv.stride_w == 1 &&
           conv.dilation_h == 1 && conv.dilation_w == 1 &&
           conv.in_channels == 512 && conv.out_channels == 512;
}

namespace {

constexpr int PATTERN_NEG_NEG = 0;
constexpr int PATTERN_NEG_ZERO = 1;
constexpr int PATTERN_NEG_POS = 2;
constexpr int PATTERN_ZERO_NEG = 3;
constexpr int PATTERN_ZERO_POS = 5;
constexpr int PATTERN_POS_NEG = 6;
constexpr int PATTERN_POS_ZERO = 7;
constexpr int PATTERN_POS_POS = 8;

inline __m256i load_i8_s1_x8(const int8_t* base) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(base));
    return _mm256_cvtepi8_epi32(bytes);
}

// Load 8 INT8 values at stride-2 (reads 16 bytes, picks every other one)
inline __m256i load_i8_s2_x8(const int8_t* base) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base));
    const __m128i shuf = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 14,12,10,8,6,4,2,0);
    return _mm256_cvtepi8_epi32(_mm_shuffle_epi8(raw, shuf));
}

alignas(32) static const int mask_lookup[9][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},   // 0 lanes
    {-1, 0, 0, 0, 0, 0, 0, 0},  // 1 lane
    {-1,-1, 0, 0, 0, 0, 0, 0},  // 2 lanes
    {-1,-1,-1, 0, 0, 0, 0, 0},  // 3 lanes
    {-1,-1,-1,-1, 0, 0, 0, 0},  // 4 lanes
    {-1,-1,-1,-1,-1, 0, 0, 0},  // 5 lanes
    {-1,-1,-1,-1,-1,-1, 0, 0},  // 6 lanes
    {-1,-1,-1,-1,-1,-1,-1, 0},  // 7 lanes
    {-1,-1,-1,-1,-1,-1,-1,-1},  // 8 lanes (full)
};

inline __m256i make_mask_i32(int valid_lanes) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(mask_lookup[valid_lanes]));
}

// OH_BLOCK=4 inner loop for a single pattern bucket.
// All 8 OC-Block2 accumulators (2 OC × 4 rows) are updated in one pass over the actions.
#define Q8_OCB2_S4_CASE(PATN, X0_OP, X1_OP) \
    case PATN: \
        for (const OCBlock2Action& action : bucket.actions) { \
            const int iw  = ow + static_cast<int>(action.kx); \
            const int ic  = static_cast<int>(action.ic); \
            const int b0  = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw; \
            const int b1  = b0 + xq_w; \
            const int b2  = b1 + xq_w; \
            const int b3  = b2 + xq_w; \
            const __m256i x0 = load_i8_s1_x8(data + b0); \
            const __m256i x1 = load_i8_s1_x8(data + b1); \
            const __m256i x2 = load_i8_s1_x8(data + b2); \
            const __m256i x3 = load_i8_s1_x8(data + b3); \
            a0r0 = _mm256_##X0_OP##_epi32(a0r0, x0); a0r1 = _mm256_##X0_OP##_epi32(a0r1, x1); \
            a0r2 = _mm256_##X0_OP##_epi32(a0r2, x2); a0r3 = _mm256_##X0_OP##_epi32(a0r3, x3); \
            a1r0 = _mm256_##X1_OP##_epi32(a1r0, x0); a1r1 = _mm256_##X1_OP##_epi32(a1r1, x1); \
            a1r2 = _mm256_##X1_OP##_epi32(a1r2, x2); a1r3 = _mm256_##X1_OP##_epi32(a1r3, x3); \
        } \
        break

// ZERO cases only update one accumulator pair
#define Q8_OCB2_S4_CASE_0(PATN, WHICH_OP) \
    case PATN: \
        for (const OCBlock2Action& action : bucket.actions) { \
            const int iw  = ow + static_cast<int>(action.kx); \
            const int ic  = static_cast<int>(action.ic); \
            const int b0  = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw; \
            const int b1  = b0 + xq_w; \
            const int b2  = b1 + xq_w; \
            const int b3  = b2 + xq_w; \
            const __m256i x0 = load_i8_s1_x8(data + b0); \
            const __m256i x1 = load_i8_s1_x8(data + b1); \
            const __m256i x2 = load_i8_s1_x8(data + b2); \
            const __m256i x3 = load_i8_s1_x8(data + b3); \
            a0r0 = _mm256_##WHICH_OP##_epi32(a0r0, x0); a0r1 = _mm256_##WHICH_OP##_epi32(a0r1, x1); \
            a0r2 = _mm256_##WHICH_OP##_epi32(a0r2, x2); a0r3 = _mm256_##WHICH_OP##_epi32(a0r3, x3); \
        } \
        break

#define Q8_OCB2_S4_CASE_1(PATN, WHICH_OP) \
    case PATN: \
        for (const OCBlock2Action& action : bucket.actions) { \
            const int iw  = ow + static_cast<int>(action.kx); \
            const int ic  = static_cast<int>(action.ic); \
            const int b0  = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw; \
            const int b1  = b0 + xq_w; \
            const int b2  = b1 + xq_w; \
            const int b3  = b2 + xq_w; \
            const __m256i x0 = load_i8_s1_x8(data + b0); \
            const __m256i x1 = load_i8_s1_x8(data + b1); \
            const __m256i x2 = load_i8_s1_x8(data + b2); \
            const __m256i x3 = load_i8_s1_x8(data + b3); \
            a1r0 = _mm256_##WHICH_OP##_epi32(a1r0, x0); a1r1 = _mm256_##WHICH_OP##_epi32(a1r1, x1); \
            a1r2 = _mm256_##WHICH_OP##_epi32(a1r2, x2); a1r3 = _mm256_##WHICH_OP##_epi32(a1r3, x3); \
        } \
        break

// Apply BN + optional ReLU, store 4 rows × 1 OC channel (8 output values each).
// For masked stores: valid_lanes < 8.
inline void apply_bn_store(
    __m256i acc_r0, __m256i acc_r1, __m256i acc_r2, __m256i acc_r3,
    float out_scale, float bias,
    bool apply_relu,
    int oc, int oh, int ow, int valid_lanes,
    const __m256 zero_v, const __m256i store_mask,
    Tensor& y_out
) {
    const __m256 sc = _mm256_set1_ps(out_scale);
    const __m256 bi = _mm256_set1_ps(bias);

    auto finish = [&](auto acc, int row) {
        __m256 y = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc), sc), bi);
        if (apply_relu) y = _mm256_max_ps(y, zero_v);
        if (valid_lanes == 8) {
            _mm256_storeu_ps(y_out.ptr(oc, oh + row, ow), y);
        } else {
            _mm256_maskstore_ps(y_out.ptr(oc, oh + row, ow), store_mask, y);
        }
    };
    finish(acc_r0, 0);
    finish(acc_r1, 1);
    finish(acc_r2, 2);
    finish(acc_r3, 3);
}

void conv3x3_q8_ocblock2_spatial4_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    int out_h,
    int out_w,
    Tensor& y_out
) {
    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();
    const int xq_h = xq.h;
    const int xq_w = xq.w;
    const __m256 zero_v = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock2PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float s0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float s1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float b0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float b1 = bn.bias[static_cast<size_t>(oc + 1)];

        // OH_BLOCK=4 main loop
        int oh = 0;
        for (; oh + 3 < out_h; oh += 4) {
            const int ih0 = oh; // stride=1, no offset needed; ky is in the action

            int ow = 0;
            for (; ow + 7 < out_w; ow += 8) {
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
                __m256i a0r2 = _mm256_setzero_si256(), a1r2 = _mm256_setzero_si256();
                __m256i a0r3 = _mm256_setzero_si256(), a1r3 = _mm256_setzero_si256();

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    switch (bucket.pattern_id) {
                        Q8_OCB2_S4_CASE(PATTERN_NEG_NEG,  sub, sub);
                        Q8_OCB2_S4_CASE_0(PATTERN_NEG_ZERO, sub);
                        Q8_OCB2_S4_CASE(PATTERN_NEG_POS,  sub, add);
                        Q8_OCB2_S4_CASE_1(PATTERN_ZERO_NEG, sub);
                        Q8_OCB2_S4_CASE_1(PATTERN_ZERO_POS, add);
                        Q8_OCB2_S4_CASE(PATTERN_POS_NEG,  add, sub);
                        Q8_OCB2_S4_CASE_0(PATTERN_POS_ZERO, add);
                        Q8_OCB2_S4_CASE(PATTERN_POS_POS,  add, add);
                        default: break;
                    }
                }

                const __m256i full_mask = make_mask_i32(8);
                apply_bn_store(a0r0, a0r1, a0r2, a0r3, s0, b0, apply_relu, oc+0, oh, ow, 8, zero_v, full_mask, y_out);
                apply_bn_store(a1r0, a1r1, a1r2, a1r3, s1, b1, apply_relu, oc+1, oh, ow, 8, zero_v, full_mask, y_out);
            }
            // OW tail
            if (ow < out_w) {
                const int valid = out_w - ow;
                const __m256i tail_mask = make_mask_i32(valid);
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
                __m256i a0r2 = _mm256_setzero_si256(), a1r2 = _mm256_setzero_si256();
                __m256i a0r3 = _mm256_setzero_si256(), a1r3 = _mm256_setzero_si256();

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    switch (bucket.pattern_id) {
                        Q8_OCB2_S4_CASE(PATTERN_NEG_NEG,  sub, sub);
                        Q8_OCB2_S4_CASE_0(PATTERN_NEG_ZERO, sub);
                        Q8_OCB2_S4_CASE(PATTERN_NEG_POS,  sub, add);
                        Q8_OCB2_S4_CASE_1(PATTERN_ZERO_NEG, sub);
                        Q8_OCB2_S4_CASE_1(PATTERN_ZERO_POS, add);
                        Q8_OCB2_S4_CASE(PATTERN_POS_NEG,  add, sub);
                        Q8_OCB2_S4_CASE_0(PATTERN_POS_ZERO, add);
                        Q8_OCB2_S4_CASE(PATTERN_POS_POS,  add, add);
                        default: break;
                    }
                }

                apply_bn_store(a0r0, a0r1, a0r2, a0r3, s0, b0, apply_relu, oc+0, oh, ow, valid, zero_v, tail_mask, y_out);
                apply_bn_store(a1r0, a1r1, a1r2, a1r3, s1, b1, apply_relu, oc+1, oh, ow, valid, zero_v, tail_mask, y_out);
            }
        }

        // OH_BLOCK=2 tail for odd remainder rows
        for (; oh + 1 < out_h; oh += 2) {
            const int ih0 = oh;
            int ow = 0;
            for (; ow + 7 < out_w; ow += 8) {
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
                __m256i a0r2 = _mm256_setzero_si256(); // dummy
                __m256i a0r3 = _mm256_setzero_si256(); // dummy
                __m256i a1r2 = _mm256_setzero_si256(); // dummy
                __m256i a1r3 = _mm256_setzero_si256(); // dummy
                (void)a0r2; (void)a0r3; (void)a1r2; (void)a1r3;

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int iw = ow + static_cast<int>(action.kx);
                        const int ic = static_cast<int>(action.ic);
                        const int b0b = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw;
                        const int b1b = b0b + xq_w;
                        const __m256i x0 = load_i8_s1_x8(data + b0b);
                        const __m256i x1 = load_i8_s1_x8(data + b1b);
                        const int8_t w0 = bucket.w0;
                        const int8_t w1 = bucket.w1;
                        if (w0 == 1) { a0r0 = _mm256_add_epi32(a0r0, x0); a0r1 = _mm256_add_epi32(a0r1, x1); }
                        else if (w0 == -1) { a0r0 = _mm256_sub_epi32(a0r0, x0); a0r1 = _mm256_sub_epi32(a0r1, x1); }
                        if (w1 == 1) { a1r0 = _mm256_add_epi32(a1r0, x0); a1r1 = _mm256_add_epi32(a1r1, x1); }
                        else if (w1 == -1) { a1r0 = _mm256_sub_epi32(a1r0, x0); a1r1 = _mm256_sub_epi32(a1r1, x1); }
                    }
                }

                const __m256 sc0 = _mm256_set1_ps(s0); const __m256 bi0 = _mm256_set1_ps(b0);
                const __m256 sc1 = _mm256_set1_ps(s1); const __m256 bi1 = _mm256_set1_ps(b1);
                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0r0), sc0), bi0);
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1r0), sc1), bi1);
                __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0r1), sc0), bi0);
                __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1r1), sc1), bi1);
                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                }
                _mm256_storeu_ps(y_out.ptr(oc+0, oh,   ow), y00);
                _mm256_storeu_ps(y_out.ptr(oc+1, oh,   ow), y01);
                _mm256_storeu_ps(y_out.ptr(oc+0, oh+1, ow), y10);
                _mm256_storeu_ps(y_out.ptr(oc+1, oh+1, ow), y11);
            }
            if (ow < out_w) {
                const int valid = out_w - ow;
                const __m256i mask = make_mask_i32(valid);
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int iw = ow + static_cast<int>(action.kx);
                        const int ic = static_cast<int>(action.ic);
                        const int b0b = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw;
                        const int b1b = b0b + xq_w;
                        const __m256i x0 = load_i8_s1_x8(data + b0b);
                        const __m256i x1 = load_i8_s1_x8(data + b1b);
                        const int8_t w0 = bucket.w0;
                        const int8_t w1 = bucket.w1;
                        if (w0 == 1) { a0r0 = _mm256_add_epi32(a0r0, x0); a0r1 = _mm256_add_epi32(a0r1, x1); }
                        else if (w0 == -1) { a0r0 = _mm256_sub_epi32(a0r0, x0); a0r1 = _mm256_sub_epi32(a0r1, x1); }
                        if (w1 == 1) { a1r0 = _mm256_add_epi32(a1r0, x0); a1r1 = _mm256_add_epi32(a1r1, x1); }
                        else if (w1 == -1) { a1r0 = _mm256_sub_epi32(a1r0, x0); a1r1 = _mm256_sub_epi32(a1r1, x1); }
                    }
                }
                const __m256 sc0 = _mm256_set1_ps(s0); const __m256 bi0 = _mm256_set1_ps(b0);
                const __m256 sc1 = _mm256_set1_ps(s1); const __m256 bi1 = _mm256_set1_ps(b1);
                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0r0), sc0), bi0);
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1r0), sc1), bi1);
                __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0r1), sc0), bi0);
                __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1r1), sc1), bi1);
                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                }
                _mm256_maskstore_ps(y_out.ptr(oc+0, oh,   ow), mask, y00);
                _mm256_maskstore_ps(y_out.ptr(oc+1, oh,   ow), mask, y01);
                _mm256_maskstore_ps(y_out.ptr(oc+0, oh+1, ow), mask, y10);
                _mm256_maskstore_ps(y_out.ptr(oc+1, oh+1, ow), mask, y11);
            }
        }

        // Single-row tail
        for (; oh < out_h; ++oh) {
            const int ih0 = oh;
            int ow = 0;
            for (; ow + 7 < out_w; ow += 8) {
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int iw = ow + static_cast<int>(action.kx);
                        const int ic = static_cast<int>(action.ic);
                        const int b0b = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw;
                        const __m256i x0 = load_i8_s1_x8(data + b0b);
                        const int8_t w0 = bucket.w0;
                        const int8_t w1 = bucket.w1;
                        if (w0 == 1) a0r0 = _mm256_add_epi32(a0r0, x0);
                        else if (w0 == -1) a0r0 = _mm256_sub_epi32(a0r0, x0);
                        if (w1 == 1) a1r0 = _mm256_add_epi32(a1r0, x0);
                        else if (w1 == -1) a1r0 = _mm256_sub_epi32(a1r0, x0);
                    }
                }
                const __m256 sc0 = _mm256_set1_ps(s0); const __m256 bi0 = _mm256_set1_ps(b0);
                const __m256 sc1 = _mm256_set1_ps(s1); const __m256 bi1 = _mm256_set1_ps(b1);
                __m256 y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0r0), sc0), bi0);
                __m256 y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1r0), sc1), bi1);
                if (apply_relu) { y0 = _mm256_max_ps(y0, zero_v); y1 = _mm256_max_ps(y1, zero_v); }
                _mm256_storeu_ps(y_out.ptr(oc+0, oh, ow), y0);
                _mm256_storeu_ps(y_out.ptr(oc+1, oh, ow), y1);
            }
            if (ow < out_w) {
                const int valid = out_w - ow;
                const __m256i mask = make_mask_i32(valid);
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int iw = ow + static_cast<int>(action.kx);
                        const int ic = static_cast<int>(action.ic);
                        const int b0b = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw;
                        const __m256i x0 = load_i8_s1_x8(data + b0b);
                        const int8_t w0 = bucket.w0;
                        const int8_t w1 = bucket.w1;
                        if (w0 == 1) a0r0 = _mm256_add_epi32(a0r0, x0);
                        else if (w0 == -1) a0r0 = _mm256_sub_epi32(a0r0, x0);
                        if (w1 == 1) a1r0 = _mm256_add_epi32(a1r0, x0);
                        else if (w1 == -1) a1r0 = _mm256_sub_epi32(a1r0, x0);
                    }
                }
                const __m256 sc0 = _mm256_set1_ps(s0); const __m256 bi0 = _mm256_set1_ps(b0);
                const __m256 sc1 = _mm256_set1_ps(s1); const __m256 bi1 = _mm256_set1_ps(b1);
                __m256 y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a0r0), sc0), bi0);
                __m256 y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a1r0), sc1), bi1);
                if (apply_relu) { y0 = _mm256_max_ps(y0, zero_v); y1 = _mm256_max_ps(y1, zero_v); }
                _mm256_maskstore_ps(y_out.ptr(oc+0, oh, ow), mask, y0);
                _mm256_maskstore_ps(y_out.ptr(oc+1, oh, ow), mask, y1);
            }
        }
    }
}

#undef Q8_OCB2_S4_CASE
#undef Q8_OCB2_S4_CASE_0
#undef Q8_OCB2_S4_CASE_1

} // namespace

void conv3x3_q8_ocblock2_spatial4_layer1_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_spatial4_impl(xq, conv, plan, bn, false, 56, 56, y_out);
}

void conv3x3_q8_ocblock2_spatial4_layer1_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_spatial4_impl(xq, conv, plan, bn, true, 56, 56, y_out);
}

void conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_spatial4_impl(xq, conv, plan, bn, false, 28, 28, y_out);
}

void conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_spatial4_impl(xq, conv, plan, bn, true, 28, 28, y_out);
}

void conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_spatial4_impl(xq, conv, plan, bn, false, 14, 14, y_out);
}

void conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_spatial4_impl(xq, conv, plan, bn, true, 14, 14, y_out);
}

// Stride-2 OH_BLOCK=4 kernel for layer2.0.conv1 (64→128ch, 56×56→28×28).
// Uses load_i8_s2_x8 (16-byte load, stride-2 extract) for all OW tiles.
// out_h=28 (28/4=7 exact tiles), out_w=28 (3 full + tail at ow=24 valid=4).
// Requires extra_right_pad=8 in padded input (safe 16-byte load at tail).
static void conv3x3_q8_ocblock2_s2_oh4_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    constexpr int OUT_H = 28;
    constexpr int OUT_W = 28;
    y_out.reuse(conv.out_channels, OUT_H, OUT_W);

    const int8_t* const data = xq.data.data();
    const int xq_h = xq.h;
    const int xq_w = xq.w;
    const __m256 zero_v = _mm256_setzero_ps();
    const __m256i tail_mask = make_mask_i32(4);  // ow=24, valid=4

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock2PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float s0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float s1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float b0v = bn.bias[static_cast<size_t>(oc + 0)];
        const float b1v = bn.bias[static_cast<size_t>(oc + 1)];
        const __m256 sc0 = _mm256_set1_ps(s0); const __m256 bi0 = _mm256_set1_ps(b0v);
        const __m256 sc1 = _mm256_set1_ps(s1); const __m256 bi1 = _mm256_set1_ps(b1v);

        for (int oh = 0; oh < OUT_H; oh += 4) {
            const int ih0 = oh * 2;  // stride-2: output row oh → input row 2*oh

            // Full OW tiles (ow=0,8,16)
            for (int ow = 0; ow + 7 < OUT_W; ow += 8) {
                const int iw_base = ow * 2;  // stride-2: output col ow → input col 2*ow
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
                __m256i a0r2 = _mm256_setzero_si256(), a1r2 = _mm256_setzero_si256();
                __m256i a0r3 = _mm256_setzero_si256(), a1r3 = _mm256_setzero_si256();

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int iw  = iw_base + static_cast<int>(action.kx);
                        const int ic  = static_cast<int>(action.ic);
                        const int b0b = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw;
                        const __m256i x0 = load_i8_s2_x8(data + b0b + 0 * 2 * xq_w);
                        const __m256i x1 = load_i8_s2_x8(data + b0b + 1 * 2 * xq_w);
                        const __m256i x2 = load_i8_s2_x8(data + b0b + 2 * 2 * xq_w);
                        const __m256i x3 = load_i8_s2_x8(data + b0b + 3 * 2 * xq_w);
                        const int8_t w0 = bucket.w0;
                        const int8_t w1 = bucket.w1;
                        if (w0 == 1) {
                            a0r0 = _mm256_add_epi32(a0r0, x0); a0r1 = _mm256_add_epi32(a0r1, x1);
                            a0r2 = _mm256_add_epi32(a0r2, x2); a0r3 = _mm256_add_epi32(a0r3, x3);
                        } else if (w0 == -1) {
                            a0r0 = _mm256_sub_epi32(a0r0, x0); a0r1 = _mm256_sub_epi32(a0r1, x1);
                            a0r2 = _mm256_sub_epi32(a0r2, x2); a0r3 = _mm256_sub_epi32(a0r3, x3);
                        }
                        if (w1 == 1) {
                            a1r0 = _mm256_add_epi32(a1r0, x0); a1r1 = _mm256_add_epi32(a1r1, x1);
                            a1r2 = _mm256_add_epi32(a1r2, x2); a1r3 = _mm256_add_epi32(a1r3, x3);
                        } else if (w1 == -1) {
                            a1r0 = _mm256_sub_epi32(a1r0, x0); a1r1 = _mm256_sub_epi32(a1r1, x1);
                            a1r2 = _mm256_sub_epi32(a1r2, x2); a1r3 = _mm256_sub_epi32(a1r3, x3);
                        }
                    }
                }

                auto store8 = [&](auto acc0, auto acc1, int row) {
                    __m256 y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc0), sc0), bi0);
                    __m256 y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc1), sc1), bi1);
                    if (apply_relu) { y0 = _mm256_max_ps(y0, zero_v); y1 = _mm256_max_ps(y1, zero_v); }
                    _mm256_storeu_ps(y_out.ptr(oc+0, oh+row, ow), y0);
                    _mm256_storeu_ps(y_out.ptr(oc+1, oh+row, ow), y1);
                };
                store8(a0r0, a1r0, 0); store8(a0r1, a1r1, 1);
                store8(a0r2, a1r2, 2); store8(a0r3, a1r3, 3);
            }

            // OW tail: ow=24, valid=4. iw_base=48.
            {
                constexpr int ow = 24;
                constexpr int iw_base = 48;
                __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
                __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
                __m256i a0r2 = _mm256_setzero_si256(), a1r2 = _mm256_setzero_si256();
                __m256i a0r3 = _mm256_setzero_si256(), a1r3 = _mm256_setzero_si256();

                for (const OCBlock2PatternBucket& bucket : block.buckets) {
                    for (const OCBlock2Action& action : bucket.actions) {
                        const int iw  = iw_base + static_cast<int>(action.kx);
                        const int ic  = static_cast<int>(action.ic);
                        const int b0b = (ic * xq_h + ih0 + static_cast<int>(action.ky)) * xq_w + iw;
                        const __m256i x0 = load_i8_s2_x8(data + b0b + 0 * 2 * xq_w);
                        const __m256i x1 = load_i8_s2_x8(data + b0b + 1 * 2 * xq_w);
                        const __m256i x2 = load_i8_s2_x8(data + b0b + 2 * 2 * xq_w);
                        const __m256i x3 = load_i8_s2_x8(data + b0b + 3 * 2 * xq_w);
                        const int8_t w0 = bucket.w0;
                        const int8_t w1 = bucket.w1;
                        if (w0 == 1) {
                            a0r0 = _mm256_add_epi32(a0r0, x0); a0r1 = _mm256_add_epi32(a0r1, x1);
                            a0r2 = _mm256_add_epi32(a0r2, x2); a0r3 = _mm256_add_epi32(a0r3, x3);
                        } else if (w0 == -1) {
                            a0r0 = _mm256_sub_epi32(a0r0, x0); a0r1 = _mm256_sub_epi32(a0r1, x1);
                            a0r2 = _mm256_sub_epi32(a0r2, x2); a0r3 = _mm256_sub_epi32(a0r3, x3);
                        }
                        if (w1 == 1) {
                            a1r0 = _mm256_add_epi32(a1r0, x0); a1r1 = _mm256_add_epi32(a1r1, x1);
                            a1r2 = _mm256_add_epi32(a1r2, x2); a1r3 = _mm256_add_epi32(a1r3, x3);
                        } else if (w1 == -1) {
                            a1r0 = _mm256_sub_epi32(a1r0, x0); a1r1 = _mm256_sub_epi32(a1r1, x1);
                            a1r2 = _mm256_sub_epi32(a1r2, x2); a1r3 = _mm256_sub_epi32(a1r3, x3);
                        }
                    }
                }

                auto store4 = [&](auto acc0, auto acc1, int row) {
                    __m256 y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc0), sc0), bi0);
                    __m256 y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc1), sc1), bi1);
                    if (apply_relu) { y0 = _mm256_max_ps(y0, zero_v); y1 = _mm256_max_ps(y1, zero_v); }
                    _mm256_maskstore_ps(y_out.ptr(oc+0, oh+row, ow), tail_mask, y0);
                    _mm256_maskstore_ps(y_out.ptr(oc+1, oh+row, ow), tail_mask, y1);
                };
                store4(a0r0, a1r0, 0); store4(a0r1, a1r1, 1);
                store4(a0r2, a1r2, 2); store4(a0r3, a1r3, 3);
            }
        }
    }
}

void conv3x3_q8_ocblock2_spatial4_layer2_s2_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_s2_oh4_impl(xq, conv, plan, bn, true, y_out);
}

// Layer4-specialized: OH_BLOCK=7 (all rows in one pass), OW=7 (all tail, 1 tile of valid=7).
// 14 INT32 accumulators (7 rows × 2 OC) fit in 16 ymm registers. Single bucket-loop pass.
static void conv3x3_q8_ocblock2_oh7_ow7_impl(
    const QuantTensorI8& xq,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    constexpr int OUT_H = 7;
    constexpr int OUT_W = 7;
    y_out.reuse(static_cast<int>(plan.oc_blocks.size()) * 2, OUT_H, OUT_W);

    const int8_t* const data = xq.data.data();
    const int xq_h = xq.h;
    const int xq_w = xq.w;
    const __m256 zero_v = _mm256_setzero_ps();
    const __m256i tail_mask = make_mask_i32(OUT_W);

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock2PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float s0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float s1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float b0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float b1 = bn.bias[static_cast<size_t>(oc + 1)];

        // Single OW tail tile: ow=0, valid=7. Process all 7 rows at once.
        __m256i a0r0 = _mm256_setzero_si256(), a1r0 = _mm256_setzero_si256();
        __m256i a0r1 = _mm256_setzero_si256(), a1r1 = _mm256_setzero_si256();
        __m256i a0r2 = _mm256_setzero_si256(), a1r2 = _mm256_setzero_si256();
        __m256i a0r3 = _mm256_setzero_si256(), a1r3 = _mm256_setzero_si256();
        __m256i a0r4 = _mm256_setzero_si256(), a1r4 = _mm256_setzero_si256();
        __m256i a0r5 = _mm256_setzero_si256(), a1r5 = _mm256_setzero_si256();
        __m256i a0r6 = _mm256_setzero_si256(), a1r6 = _mm256_setzero_si256();

        for (const OCBlock2PatternBucket& bucket : block.buckets) {
            const int pat = bucket.pattern_id;
            for (const OCBlock2Action& action : bucket.actions) {
                const int iw = static_cast<int>(action.kx);  // ow=0
                const int ic = static_cast<int>(action.ic);
                const int base = (ic * xq_h + static_cast<int>(action.ky)) * xq_w + iw;
                const __m256i x0 = load_i8_s1_x8(data + base + 0 * xq_w);
                const __m256i x1 = load_i8_s1_x8(data + base + 1 * xq_w);
                const __m256i x2 = load_i8_s1_x8(data + base + 2 * xq_w);
                const __m256i x3 = load_i8_s1_x8(data + base + 3 * xq_w);
                const __m256i x4 = load_i8_s1_x8(data + base + 4 * xq_w);
                const __m256i x5 = load_i8_s1_x8(data + base + 5 * xq_w);
                const __m256i x6 = load_i8_s1_x8(data + base + 6 * xq_w);

                switch (pat) {
                    case PATTERN_NEG_NEG:
                        a0r0 = _mm256_sub_epi32(a0r0, x0); a1r0 = _mm256_sub_epi32(a1r0, x0);
                        a0r1 = _mm256_sub_epi32(a0r1, x1); a1r1 = _mm256_sub_epi32(a1r1, x1);
                        a0r2 = _mm256_sub_epi32(a0r2, x2); a1r2 = _mm256_sub_epi32(a1r2, x2);
                        a0r3 = _mm256_sub_epi32(a0r3, x3); a1r3 = _mm256_sub_epi32(a1r3, x3);
                        a0r4 = _mm256_sub_epi32(a0r4, x4); a1r4 = _mm256_sub_epi32(a1r4, x4);
                        a0r5 = _mm256_sub_epi32(a0r5, x5); a1r5 = _mm256_sub_epi32(a1r5, x5);
                        a0r6 = _mm256_sub_epi32(a0r6, x6); a1r6 = _mm256_sub_epi32(a1r6, x6);
                        break;
                    case PATTERN_NEG_ZERO:
                        a0r0 = _mm256_sub_epi32(a0r0, x0); a0r1 = _mm256_sub_epi32(a0r1, x1);
                        a0r2 = _mm256_sub_epi32(a0r2, x2); a0r3 = _mm256_sub_epi32(a0r3, x3);
                        a0r4 = _mm256_sub_epi32(a0r4, x4); a0r5 = _mm256_sub_epi32(a0r5, x5);
                        a0r6 = _mm256_sub_epi32(a0r6, x6);
                        break;
                    case PATTERN_NEG_POS:
                        a0r0 = _mm256_sub_epi32(a0r0, x0); a1r0 = _mm256_add_epi32(a1r0, x0);
                        a0r1 = _mm256_sub_epi32(a0r1, x1); a1r1 = _mm256_add_epi32(a1r1, x1);
                        a0r2 = _mm256_sub_epi32(a0r2, x2); a1r2 = _mm256_add_epi32(a1r2, x2);
                        a0r3 = _mm256_sub_epi32(a0r3, x3); a1r3 = _mm256_add_epi32(a1r3, x3);
                        a0r4 = _mm256_sub_epi32(a0r4, x4); a1r4 = _mm256_add_epi32(a1r4, x4);
                        a0r5 = _mm256_sub_epi32(a0r5, x5); a1r5 = _mm256_add_epi32(a1r5, x5);
                        a0r6 = _mm256_sub_epi32(a0r6, x6); a1r6 = _mm256_add_epi32(a1r6, x6);
                        break;
                    case PATTERN_ZERO_NEG:
                        a1r0 = _mm256_sub_epi32(a1r0, x0); a1r1 = _mm256_sub_epi32(a1r1, x1);
                        a1r2 = _mm256_sub_epi32(a1r2, x2); a1r3 = _mm256_sub_epi32(a1r3, x3);
                        a1r4 = _mm256_sub_epi32(a1r4, x4); a1r5 = _mm256_sub_epi32(a1r5, x5);
                        a1r6 = _mm256_sub_epi32(a1r6, x6);
                        break;
                    case PATTERN_ZERO_POS:
                        a1r0 = _mm256_add_epi32(a1r0, x0); a1r1 = _mm256_add_epi32(a1r1, x1);
                        a1r2 = _mm256_add_epi32(a1r2, x2); a1r3 = _mm256_add_epi32(a1r3, x3);
                        a1r4 = _mm256_add_epi32(a1r4, x4); a1r5 = _mm256_add_epi32(a1r5, x5);
                        a1r6 = _mm256_add_epi32(a1r6, x6);
                        break;
                    case PATTERN_POS_NEG:
                        a0r0 = _mm256_add_epi32(a0r0, x0); a1r0 = _mm256_sub_epi32(a1r0, x0);
                        a0r1 = _mm256_add_epi32(a0r1, x1); a1r1 = _mm256_sub_epi32(a1r1, x1);
                        a0r2 = _mm256_add_epi32(a0r2, x2); a1r2 = _mm256_sub_epi32(a1r2, x2);
                        a0r3 = _mm256_add_epi32(a0r3, x3); a1r3 = _mm256_sub_epi32(a1r3, x3);
                        a0r4 = _mm256_add_epi32(a0r4, x4); a1r4 = _mm256_sub_epi32(a1r4, x4);
                        a0r5 = _mm256_add_epi32(a0r5, x5); a1r5 = _mm256_sub_epi32(a1r5, x5);
                        a0r6 = _mm256_add_epi32(a0r6, x6); a1r6 = _mm256_sub_epi32(a1r6, x6);
                        break;
                    case PATTERN_POS_ZERO:
                        a0r0 = _mm256_add_epi32(a0r0, x0); a0r1 = _mm256_add_epi32(a0r1, x1);
                        a0r2 = _mm256_add_epi32(a0r2, x2); a0r3 = _mm256_add_epi32(a0r3, x3);
                        a0r4 = _mm256_add_epi32(a0r4, x4); a0r5 = _mm256_add_epi32(a0r5, x5);
                        a0r6 = _mm256_add_epi32(a0r6, x6);
                        break;
                    case PATTERN_POS_POS:
                        a0r0 = _mm256_add_epi32(a0r0, x0); a1r0 = _mm256_add_epi32(a1r0, x0);
                        a0r1 = _mm256_add_epi32(a0r1, x1); a1r1 = _mm256_add_epi32(a1r1, x1);
                        a0r2 = _mm256_add_epi32(a0r2, x2); a1r2 = _mm256_add_epi32(a1r2, x2);
                        a0r3 = _mm256_add_epi32(a0r3, x3); a1r3 = _mm256_add_epi32(a1r3, x3);
                        a0r4 = _mm256_add_epi32(a0r4, x4); a1r4 = _mm256_add_epi32(a1r4, x4);
                        a0r5 = _mm256_add_epi32(a0r5, x5); a1r5 = _mm256_add_epi32(a1r5, x5);
                        a0r6 = _mm256_add_epi32(a0r6, x6); a1r6 = _mm256_add_epi32(a1r6, x6);
                        break;
                    default: break;
                }
            }
        }

        // Apply BN, masked store (valid=7)
        const __m256 sc0 = _mm256_set1_ps(s0); const __m256 bi0 = _mm256_set1_ps(b0);
        const __m256 sc1 = _mm256_set1_ps(s1); const __m256 bi1 = _mm256_set1_ps(b1);
        auto store_oc = [&](auto acc0, auto acc1, int row) {
            __m256 y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc0), sc0), bi0);
            __m256 y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc1), sc1), bi1);
            if (apply_relu) { y0 = _mm256_max_ps(y0, zero_v); y1 = _mm256_max_ps(y1, zero_v); }
            _mm256_maskstore_ps(y_out.ptr(oc+0, row, 0), tail_mask, y0);
            _mm256_maskstore_ps(y_out.ptr(oc+1, row, 0), tail_mask, y1);
        };
        store_oc(a0r0, a1r0, 0);
        store_oc(a0r1, a1r1, 1);
        store_oc(a0r2, a1r2, 2);
        store_oc(a0r3, a1r3, 3);
        store_oc(a0r4, a1r4, 4);
        store_oc(a0r5, a1r5, 5);
        store_oc(a0r6, a1r6, 6);
    }
}

void conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_oh7_ow7_impl(xq, plan, bn, false, y_out);
}

void conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out
) {
    conv3x3_q8_ocblock2_oh7_ow7_impl(xq, plan, bn, true, y_out);
}
