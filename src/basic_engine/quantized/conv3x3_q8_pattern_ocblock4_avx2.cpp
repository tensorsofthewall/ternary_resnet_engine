#include "quantized/conv3x3_q8_pattern_ocblock4_avx2.hpp"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>

#include <omp.h>

bool can_use_q8_ocblock4_layer4_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 512 &&
           conv.out_channels == 512 &&
           conv.out_channels % 4 == 0;
}

bool can_use_q8_ocblock4_s2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 128 &&
           conv.out_channels == 256 &&
           conv.out_channels % 4 == 0;
}

bool can_use_q8_ocblock4_layer4_s2_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 256 &&
           conv.out_channels == 512 &&
           conv.out_channels % 4 == 0;
}

bool can_use_q8_ocblock4_layer3_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 256 &&
           conv.out_channels == 256 &&
           conv.out_channels % 4 == 0;
}

bool can_use_q8_ocblock4_layer2_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 128 &&
           conv.out_channels == 128 &&
           conv.out_channels % 4 == 0;
}

bool can_use_q8_ocblock4_layer1_s1_3x3_avx2(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.in_channels == 64 &&
           conv.out_channels == 64 &&
           conv.out_channels % 4 == 0;
}

namespace {

// Reads one quantized int8 activation and widens it to int32 for scalar
// accumulation.
inline int32_t q8_at(
    const QuantTensorI8& xq,
    int c,
    int h,
    int w
) {
    return static_cast<int32_t>(xq(c, h, w));
}

// Builds an AVX2 lane mask with the low `valid_lanes` int32 lanes set to -1
// (all bits) and the rest 0, for masked tail stores when an output row's
// width isn't a multiple of 8.
inline __m256i make_tail_mask_i32(int valid_lanes) {
    alignas(32) int mask_values[8];

    for (int i = 0; i < 8; ++i) {
        mask_values[i] = (i < valid_lanes) ? -1 : 0;
    }

    return _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(mask_values)
    );
}

// Stores up to 8 floats to `base`, writing only the low `valid_lanes` lanes
// -- the masked counterpart to a plain _mm256_storeu_ps for output-row tails
// narrower than 8 pixels.
inline void store_width_masked_8(
    float* base,
    __m256 value,
    int valid_lanes
) {
    const __m256i mask = make_tail_mask_i32(valid_lanes);
    _mm256_maskstore_ps(base, mask, value);
}

// Loads exactly 7 int8 activations (the layer4 single-tile OW=7 case) into
// an 8-lane int32 vector, zero-filling lane 7 via a stack scratch buffer
// (avoids reading the 8th, possibly out-of-bounds, byte directly).
inline __m256i load_i8_width7_to_i32_safe(const int8_t* base) {
    alignas(16) int8_t tmp[8];

    tmp[0] = base[0];
    tmp[1] = base[1];
    tmp[2] = base[2];
    tmp[3] = base[3];
    tmp[4] = base[4];
    tmp[5] = base[5];
    tmp[6] = base[6];
    tmp[7] = 0;

    const __m128i bytes =
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(tmp));

    return _mm256_cvtepi8_epi32(bytes);
}

// Vector stride-1 load: reads 8 consecutive int8 activations and widens them
// to an 8-lane int32 vector with one load + one sign-extend (no shuffling --
// the fast path for stride-1 convs where activations are already contiguous).
inline __m256i load_i8_width8_s1_to_i32_fast(const int8_t* base) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(base));
    return _mm256_cvtepi8_epi32(bytes);
}

// Vector stride-2 load: reads 16 consecutive bytes, extracts every other byte
// (positions 0,2,4,6,8,10,12,14) via SSSE3 shuffle, widens to 8 int32s.
// Requires SSSE3 (_mm_shuffle_epi8). Caller must ensure base[0..15] are in-bounds.
inline __m256i load_i8_stride2_x8_to_i32(const int8_t* base) {
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base));
    // Pick bytes at positions 0,2,4,6,8,10,12,14 into low 8 bytes; -1 → zero upper
    const __m128i shuf = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1, 14,12,10,8,6,4,2,0);
    const __m128i packed = _mm_shuffle_epi8(raw, shuf);
    return _mm256_cvtepi8_epi32(packed);
}

// OH_BLOCK=2 bucket runner: W0..W3 are compile-time constants → branch-free inner loop.
// ow_base: start column of current OW tile (0 for layer4 single-tile; ow for multi-tile).
template <int W0, int W1, int W2, int W3>
inline void run_q8_ocblock4_oh2_bucket(
    const OCBlock4PatternBucket& bucket,
    int oh,
    int ow_base,
    int xq_h,
    int xq_w,
    const int8_t* data,
    __m256i& acc00, __m256i& acc01, __m256i& acc02, __m256i& acc03,
    __m256i& acc10, __m256i& acc11, __m256i& acc12, __m256i& acc13
) {
    for (const OCBlock4PatternAction& action : bucket.actions) {
        const int ih  = oh + static_cast<int>(action.ky);
        const int iw  = ow_base + static_cast<int>(action.kx);
        const int base0 = (static_cast<int>(action.ic) * xq_h + ih) * xq_w + iw;
        const int base1 = base0 + xq_w;
        const __m256i x0 = load_i8_width8_s1_to_i32_fast(data + base0);
        const __m256i x1 = load_i8_width8_s1_to_i32_fast(data + base1);
        if constexpr (W0 ==  1) { acc00 = _mm256_add_epi32(acc00, x0); acc10 = _mm256_add_epi32(acc10, x1); }
        if constexpr (W0 == -1) { acc00 = _mm256_sub_epi32(acc00, x0); acc10 = _mm256_sub_epi32(acc10, x1); }
        if constexpr (W1 ==  1) { acc01 = _mm256_add_epi32(acc01, x0); acc11 = _mm256_add_epi32(acc11, x1); }
        if constexpr (W1 == -1) { acc01 = _mm256_sub_epi32(acc01, x0); acc11 = _mm256_sub_epi32(acc11, x1); }
        if constexpr (W2 ==  1) { acc02 = _mm256_add_epi32(acc02, x0); acc12 = _mm256_add_epi32(acc12, x1); }
        if constexpr (W2 == -1) { acc02 = _mm256_sub_epi32(acc02, x0); acc12 = _mm256_sub_epi32(acc12, x1); }
        if constexpr (W3 ==  1) { acc03 = _mm256_add_epi32(acc03, x0); acc13 = _mm256_add_epi32(acc13, x1); }
        if constexpr (W3 == -1) { acc03 = _mm256_sub_epi32(acc03, x0); acc13 = _mm256_sub_epi32(acc13, x1); }
    }
}

// OH_BLOCK=1 bucket runner (residual row): 4 accumulators.
template <int W0, int W1, int W2, int W3>
inline void run_q8_ocblock4_oh1_bucket(
    const OCBlock4PatternBucket& bucket,
    int oh,
    int ow_base,
    int xq_h,
    int xq_w,
    const int8_t* data,
    __m256i& acc0, __m256i& acc1, __m256i& acc2, __m256i& acc3
) {
    for (const OCBlock4PatternAction& action : bucket.actions) {
        const int ih  = oh + static_cast<int>(action.ky);
        const int iw  = ow_base + static_cast<int>(action.kx);
        const int base = (static_cast<int>(action.ic) * xq_h + ih) * xq_w + iw;
        const __m256i x = load_i8_width8_s1_to_i32_fast(data + base);
        if constexpr (W0 ==  1) acc0 = _mm256_add_epi32(acc0, x);
        if constexpr (W0 == -1) acc0 = _mm256_sub_epi32(acc0, x);
        if constexpr (W1 ==  1) acc1 = _mm256_add_epi32(acc1, x);
        if constexpr (W1 == -1) acc1 = _mm256_sub_epi32(acc1, x);
        if constexpr (W2 ==  1) acc2 = _mm256_add_epi32(acc2, x);
        if constexpr (W2 == -1) acc2 = _mm256_sub_epi32(acc2, x);
        if constexpr (W3 ==  1) acc3 = _mm256_add_epi32(acc3, x);
        if constexpr (W3 == -1) acc3 = _mm256_sub_epi32(acc3, x);
    }
}

#define Q8_OCB4_OH2_DISPATCH(ID, W0, W1, W2, W3) \
    case ID: run_q8_ocblock4_oh2_bucket<W0,W1,W2,W3>( \
        bucket, oh, ow_base, xq_h, xq_w, data, \
        acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13); break

#define Q8_OCB4_OH1_DISPATCH(ID, W0, W1, W2, W3) \
    case ID: run_q8_ocblock4_oh1_bucket<W0,W1,W2,W3>( \
        bucket, oh, ow_base, xq_h, xq_w, data, \
        acc0, acc1, acc2, acc3); break

inline void q8_ocblock4_oh2_dispatch(
    const OCBlock4PatternBucket& bucket,
    int oh, int ow_base, int xq_h, int xq_w, const int8_t* data,
    __m256i& acc00, __m256i& acc01, __m256i& acc02, __m256i& acc03,
    __m256i& acc10, __m256i& acc11, __m256i& acc12, __m256i& acc13
) {
    switch (bucket.pattern_id) {
        Q8_OCB4_OH2_DISPATCH( 0, -1,-1,-1,-1); Q8_OCB4_OH2_DISPATCH( 1, -1,-1,-1, 0);
        Q8_OCB4_OH2_DISPATCH( 2, -1,-1,-1, 1); Q8_OCB4_OH2_DISPATCH( 3, -1,-1, 0,-1);
        Q8_OCB4_OH2_DISPATCH( 4, -1,-1, 0, 0); Q8_OCB4_OH2_DISPATCH( 5, -1,-1, 0, 1);
        Q8_OCB4_OH2_DISPATCH( 6, -1,-1, 1,-1); Q8_OCB4_OH2_DISPATCH( 7, -1,-1, 1, 0);
        Q8_OCB4_OH2_DISPATCH( 8, -1,-1, 1, 1); Q8_OCB4_OH2_DISPATCH( 9, -1, 0,-1,-1);
        Q8_OCB4_OH2_DISPATCH(10, -1, 0,-1, 0); Q8_OCB4_OH2_DISPATCH(11, -1, 0,-1, 1);
        Q8_OCB4_OH2_DISPATCH(12, -1, 0, 0,-1); Q8_OCB4_OH2_DISPATCH(13, -1, 0, 0, 0);
        Q8_OCB4_OH2_DISPATCH(14, -1, 0, 0, 1); Q8_OCB4_OH2_DISPATCH(15, -1, 0, 1,-1);
        Q8_OCB4_OH2_DISPATCH(16, -1, 0, 1, 0); Q8_OCB4_OH2_DISPATCH(17, -1, 0, 1, 1);
        Q8_OCB4_OH2_DISPATCH(18, -1, 1,-1,-1); Q8_OCB4_OH2_DISPATCH(19, -1, 1,-1, 0);
        Q8_OCB4_OH2_DISPATCH(20, -1, 1,-1, 1); Q8_OCB4_OH2_DISPATCH(21, -1, 1, 0,-1);
        Q8_OCB4_OH2_DISPATCH(22, -1, 1, 0, 0); Q8_OCB4_OH2_DISPATCH(23, -1, 1, 0, 1);
        Q8_OCB4_OH2_DISPATCH(24, -1, 1, 1,-1); Q8_OCB4_OH2_DISPATCH(25, -1, 1, 1, 0);
        Q8_OCB4_OH2_DISPATCH(26, -1, 1, 1, 1); Q8_OCB4_OH2_DISPATCH(27,  0,-1,-1,-1);
        Q8_OCB4_OH2_DISPATCH(28,  0,-1,-1, 0); Q8_OCB4_OH2_DISPATCH(29,  0,-1,-1, 1);
        Q8_OCB4_OH2_DISPATCH(30,  0,-1, 0,-1); Q8_OCB4_OH2_DISPATCH(31,  0,-1, 0, 0);
        Q8_OCB4_OH2_DISPATCH(32,  0,-1, 0, 1); Q8_OCB4_OH2_DISPATCH(33,  0,-1, 1,-1);
        Q8_OCB4_OH2_DISPATCH(34,  0,-1, 1, 0); Q8_OCB4_OH2_DISPATCH(35,  0,-1, 1, 1);
        Q8_OCB4_OH2_DISPATCH(36,  0, 0,-1,-1); Q8_OCB4_OH2_DISPATCH(37,  0, 0,-1, 0);
        Q8_OCB4_OH2_DISPATCH(38,  0, 0,-1, 1); Q8_OCB4_OH2_DISPATCH(39,  0, 0, 0,-1);
        case 40: break; // all-zero: should not appear in plan
        Q8_OCB4_OH2_DISPATCH(41,  0, 0, 0, 1); Q8_OCB4_OH2_DISPATCH(42,  0, 0, 1,-1);
        Q8_OCB4_OH2_DISPATCH(43,  0, 0, 1, 0); Q8_OCB4_OH2_DISPATCH(44,  0, 0, 1, 1);
        Q8_OCB4_OH2_DISPATCH(45,  0, 1,-1,-1); Q8_OCB4_OH2_DISPATCH(46,  0, 1,-1, 0);
        Q8_OCB4_OH2_DISPATCH(47,  0, 1,-1, 1); Q8_OCB4_OH2_DISPATCH(48,  0, 1, 0,-1);
        Q8_OCB4_OH2_DISPATCH(49,  0, 1, 0, 0); Q8_OCB4_OH2_DISPATCH(50,  0, 1, 0, 1);
        Q8_OCB4_OH2_DISPATCH(51,  0, 1, 1,-1); Q8_OCB4_OH2_DISPATCH(52,  0, 1, 1, 0);
        Q8_OCB4_OH2_DISPATCH(53,  0, 1, 1, 1); Q8_OCB4_OH2_DISPATCH(54,  1,-1,-1,-1);
        Q8_OCB4_OH2_DISPATCH(55,  1,-1,-1, 0); Q8_OCB4_OH2_DISPATCH(56,  1,-1,-1, 1);
        Q8_OCB4_OH2_DISPATCH(57,  1,-1, 0,-1); Q8_OCB4_OH2_DISPATCH(58,  1,-1, 0, 0);
        Q8_OCB4_OH2_DISPATCH(59,  1,-1, 0, 1); Q8_OCB4_OH2_DISPATCH(60,  1,-1, 1,-1);
        Q8_OCB4_OH2_DISPATCH(61,  1,-1, 1, 0); Q8_OCB4_OH2_DISPATCH(62,  1,-1, 1, 1);
        Q8_OCB4_OH2_DISPATCH(63,  1, 0,-1,-1); Q8_OCB4_OH2_DISPATCH(64,  1, 0,-1, 0);
        Q8_OCB4_OH2_DISPATCH(65,  1, 0,-1, 1); Q8_OCB4_OH2_DISPATCH(66,  1, 0, 0,-1);
        Q8_OCB4_OH2_DISPATCH(67,  1, 0, 0, 0); Q8_OCB4_OH2_DISPATCH(68,  1, 0, 0, 1);
        Q8_OCB4_OH2_DISPATCH(69,  1, 0, 1,-1); Q8_OCB4_OH2_DISPATCH(70,  1, 0, 1, 0);
        Q8_OCB4_OH2_DISPATCH(71,  1, 0, 1, 1); Q8_OCB4_OH2_DISPATCH(72,  1, 1,-1,-1);
        Q8_OCB4_OH2_DISPATCH(73,  1, 1,-1, 0); Q8_OCB4_OH2_DISPATCH(74,  1, 1,-1, 1);
        Q8_OCB4_OH2_DISPATCH(75,  1, 1, 0,-1); Q8_OCB4_OH2_DISPATCH(76,  1, 1, 0, 0);
        Q8_OCB4_OH2_DISPATCH(77,  1, 1, 0, 1); Q8_OCB4_OH2_DISPATCH(78,  1, 1, 1,-1);
        Q8_OCB4_OH2_DISPATCH(79,  1, 1, 1, 0); Q8_OCB4_OH2_DISPATCH(80,  1, 1, 1, 1);
        default: break;
    }
}

inline void q8_ocblock4_oh1_dispatch(
    const OCBlock4PatternBucket& bucket,
    int oh, int ow_base, int xq_h, int xq_w, const int8_t* data,
    __m256i& acc0, __m256i& acc1, __m256i& acc2, __m256i& acc3
) {
    switch (bucket.pattern_id) {
        Q8_OCB4_OH1_DISPATCH( 0, -1,-1,-1,-1); Q8_OCB4_OH1_DISPATCH( 1, -1,-1,-1, 0);
        Q8_OCB4_OH1_DISPATCH( 2, -1,-1,-1, 1); Q8_OCB4_OH1_DISPATCH( 3, -1,-1, 0,-1);
        Q8_OCB4_OH1_DISPATCH( 4, -1,-1, 0, 0); Q8_OCB4_OH1_DISPATCH( 5, -1,-1, 0, 1);
        Q8_OCB4_OH1_DISPATCH( 6, -1,-1, 1,-1); Q8_OCB4_OH1_DISPATCH( 7, -1,-1, 1, 0);
        Q8_OCB4_OH1_DISPATCH( 8, -1,-1, 1, 1); Q8_OCB4_OH1_DISPATCH( 9, -1, 0,-1,-1);
        Q8_OCB4_OH1_DISPATCH(10, -1, 0,-1, 0); Q8_OCB4_OH1_DISPATCH(11, -1, 0,-1, 1);
        Q8_OCB4_OH1_DISPATCH(12, -1, 0, 0,-1); Q8_OCB4_OH1_DISPATCH(13, -1, 0, 0, 0);
        Q8_OCB4_OH1_DISPATCH(14, -1, 0, 0, 1); Q8_OCB4_OH1_DISPATCH(15, -1, 0, 1,-1);
        Q8_OCB4_OH1_DISPATCH(16, -1, 0, 1, 0); Q8_OCB4_OH1_DISPATCH(17, -1, 0, 1, 1);
        Q8_OCB4_OH1_DISPATCH(18, -1, 1,-1,-1); Q8_OCB4_OH1_DISPATCH(19, -1, 1,-1, 0);
        Q8_OCB4_OH1_DISPATCH(20, -1, 1,-1, 1); Q8_OCB4_OH1_DISPATCH(21, -1, 1, 0,-1);
        Q8_OCB4_OH1_DISPATCH(22, -1, 1, 0, 0); Q8_OCB4_OH1_DISPATCH(23, -1, 1, 0, 1);
        Q8_OCB4_OH1_DISPATCH(24, -1, 1, 1,-1); Q8_OCB4_OH1_DISPATCH(25, -1, 1, 1, 0);
        Q8_OCB4_OH1_DISPATCH(26, -1, 1, 1, 1); Q8_OCB4_OH1_DISPATCH(27,  0,-1,-1,-1);
        Q8_OCB4_OH1_DISPATCH(28,  0,-1,-1, 0); Q8_OCB4_OH1_DISPATCH(29,  0,-1,-1, 1);
        Q8_OCB4_OH1_DISPATCH(30,  0,-1, 0,-1); Q8_OCB4_OH1_DISPATCH(31,  0,-1, 0, 0);
        Q8_OCB4_OH1_DISPATCH(32,  0,-1, 0, 1); Q8_OCB4_OH1_DISPATCH(33,  0,-1, 1,-1);
        Q8_OCB4_OH1_DISPATCH(34,  0,-1, 1, 0); Q8_OCB4_OH1_DISPATCH(35,  0,-1, 1, 1);
        Q8_OCB4_OH1_DISPATCH(36,  0, 0,-1,-1); Q8_OCB4_OH1_DISPATCH(37,  0, 0,-1, 0);
        Q8_OCB4_OH1_DISPATCH(38,  0, 0,-1, 1); Q8_OCB4_OH1_DISPATCH(39,  0, 0, 0,-1);
        case 40: break;
        Q8_OCB4_OH1_DISPATCH(41,  0, 0, 0, 1); Q8_OCB4_OH1_DISPATCH(42,  0, 0, 1,-1);
        Q8_OCB4_OH1_DISPATCH(43,  0, 0, 1, 0); Q8_OCB4_OH1_DISPATCH(44,  0, 0, 1, 1);
        Q8_OCB4_OH1_DISPATCH(45,  0, 1,-1,-1); Q8_OCB4_OH1_DISPATCH(46,  0, 1,-1, 0);
        Q8_OCB4_OH1_DISPATCH(47,  0, 1,-1, 1); Q8_OCB4_OH1_DISPATCH(48,  0, 1, 0,-1);
        Q8_OCB4_OH1_DISPATCH(49,  0, 1, 0, 0); Q8_OCB4_OH1_DISPATCH(50,  0, 1, 0, 1);
        Q8_OCB4_OH1_DISPATCH(51,  0, 1, 1,-1); Q8_OCB4_OH1_DISPATCH(52,  0, 1, 1, 0);
        Q8_OCB4_OH1_DISPATCH(53,  0, 1, 1, 1); Q8_OCB4_OH1_DISPATCH(54,  1,-1,-1,-1);
        Q8_OCB4_OH1_DISPATCH(55,  1,-1,-1, 0); Q8_OCB4_OH1_DISPATCH(56,  1,-1,-1, 1);
        Q8_OCB4_OH1_DISPATCH(57,  1,-1, 0,-1); Q8_OCB4_OH1_DISPATCH(58,  1,-1, 0, 0);
        Q8_OCB4_OH1_DISPATCH(59,  1,-1, 0, 1); Q8_OCB4_OH1_DISPATCH(60,  1,-1, 1,-1);
        Q8_OCB4_OH1_DISPATCH(61,  1,-1, 1, 0); Q8_OCB4_OH1_DISPATCH(62,  1,-1, 1, 1);
        Q8_OCB4_OH1_DISPATCH(63,  1, 0,-1,-1); Q8_OCB4_OH1_DISPATCH(64,  1, 0,-1, 0);
        Q8_OCB4_OH1_DISPATCH(65,  1, 0,-1, 1); Q8_OCB4_OH1_DISPATCH(66,  1, 0, 0,-1);
        Q8_OCB4_OH1_DISPATCH(67,  1, 0, 0, 0); Q8_OCB4_OH1_DISPATCH(68,  1, 0, 0, 1);
        Q8_OCB4_OH1_DISPATCH(69,  1, 0, 1,-1); Q8_OCB4_OH1_DISPATCH(70,  1, 0, 1, 0);
        Q8_OCB4_OH1_DISPATCH(71,  1, 0, 1, 1); Q8_OCB4_OH1_DISPATCH(72,  1, 1,-1,-1);
        Q8_OCB4_OH1_DISPATCH(73,  1, 1,-1, 0); Q8_OCB4_OH1_DISPATCH(74,  1, 1,-1, 1);
        Q8_OCB4_OH1_DISPATCH(75,  1, 1, 0,-1); Q8_OCB4_OH1_DISPATCH(76,  1, 1, 0, 0);
        Q8_OCB4_OH1_DISPATCH(77,  1, 1, 0, 1); Q8_OCB4_OH1_DISPATCH(78,  1, 1, 1,-1);
        Q8_OCB4_OH1_DISPATCH(79,  1, 1, 1, 0); Q8_OCB4_OH1_DISPATCH(80,  1, 1, 1, 1);
        default: break;
    }
}

// Scalar ternary accumulation: acc += x if w==+1, acc -= x if w==-1, no-op if
// w==0 -- the branch-on-value fallback used where bucket dispatch isn't worth
// the setup (e.g. masked tail lanes).
inline void apply_weight_i32(
    int32_t& acc,
    int32_t x,
    int8_t w
) {
    if (w == 1) {
        acc += x;
    } else if (w == -1) {
        acc -= x;
    }
}

// ---------------------------------------------------------------------
// _impl kernels: each parallelizes over plan.oc_blocks (OpenMP, one
// PatternOCBlock4Plan -- 4 output channels -- per task), and for every
// output-row tile dispatches each bucket's actions through
// q8_ocblock4_oh{1,2}_dispatch (branch-free add/sub on the int32
// activation-sum accumulators per the bucket's compile-time-constant ternary
// pattern), then applies the fused per-channel BN scale/bias, optional ReLU
// clamp, and a masked vector store for any output-row tail narrower than 8
// pixels. They differ only in their fixed input/output extents and
// row/column tiling (each documented inline below); throwing if called on a
// conv that doesn't match their can_use_* predicate's shape.
// ---------------------------------------------------------------------

// Targets layer4's 7x7 stride-1 output (OH_BLOCK=2 with a 1-row remainder;
// single OW=7 tile, masked to valid_lanes=7).
void conv3x3_q8_ocblock4_layer4_s1_bn_avx2_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    if (!can_use_q8_ocblock4_layer4_s1_3x3_avx2(conv)) {
        throw std::runtime_error(
            "q8 OC-Block4 layer4 stride-1 kernel called on unsupported conv"
        );
    }

    // xq: 512×9×10 (pad=1 symmetric + extra_right=1 → w=10 for safe 8-byte loads)
    constexpr int out_h = 7;
    constexpr int out_w = 7;
    constexpr int valid_lanes = 7;

    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float out_scale0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float out_scale1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float out_scale2 = xq.scale * bn.scale[static_cast<size_t>(oc + 2)];
        const float out_scale3 = xq.scale * bn.scale[static_cast<size_t>(oc + 3)];

        const float bias0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float bias1 = bn.bias[static_cast<size_t>(oc + 1)];
        const float bias2 = bn.bias[static_cast<size_t>(oc + 2)];
        const float bias3 = bn.bias[static_cast<size_t>(oc + 3)];

        const __m256 zero_v = _mm256_setzero_ps();

        // OH_BLOCK=2: process pairs of output rows to improve load ILP
        for (int oh = 0; oh + 1 < out_h; oh += 2) {
            __m256i acc00 = _mm256_setzero_si256();
            __m256i acc01 = _mm256_setzero_si256();
            __m256i acc02 = _mm256_setzero_si256();
            __m256i acc03 = _mm256_setzero_si256();
            __m256i acc10 = _mm256_setzero_si256();
            __m256i acc11 = _mm256_setzero_si256();
            __m256i acc12 = _mm256_setzero_si256();
            __m256i acc13 = _mm256_setzero_si256();

            for (const OCBlock4PatternBucket& bucket : block.buckets) {
                q8_ocblock4_oh2_dispatch(bucket, oh, 0, xq.h, xq.w, data,
                    acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13);
            }

            __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc00), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
            __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc01), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
            __m256 y02 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc02), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
            __m256 y03 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc03), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));
            __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc10), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
            __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc11), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
            __m256 y12 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc12), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
            __m256 y13 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc13), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));

            if (apply_relu) {
                y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                y02 = _mm256_max_ps(y02, zero_v); y03 = _mm256_max_ps(y03, zero_v);
                y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                y12 = _mm256_max_ps(y12, zero_v); y13 = _mm256_max_ps(y13, zero_v);
            }

            store_width_masked_8(y_out.ptr(oc + 0, oh,     0), y00, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 1, oh,     0), y01, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 2, oh,     0), y02, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 3, oh,     0), y03, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 0, oh + 1, 0), y10, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 1, oh + 1, 0), y11, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 2, oh + 1, 0), y12, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 3, oh + 1, 0), y13, valid_lanes);
        }

        // Residual single row when out_h is odd (out_h=7 → process row 6 here)
        if (out_h % 2 == 1) {
            constexpr int oh = out_h - 1;

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            for (const OCBlock4PatternBucket& bucket : block.buckets) {
                q8_ocblock4_oh1_dispatch(bucket, oh, 0, xq.h, xq.w, data,
                    acc0, acc1, acc2, acc3);
            }

            __m256 y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc0), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
            __m256 y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc1), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
            __m256 y2 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc2), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
            __m256 y3 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc3), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));

            if (apply_relu) {
                y0 = _mm256_max_ps(y0, zero_v); y1 = _mm256_max_ps(y1, zero_v);
                y2 = _mm256_max_ps(y2, zero_v); y3 = _mm256_max_ps(y3, zero_v);
            }

            store_width_masked_8(y_out.ptr(oc + 0, oh, 0), y0, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 1, oh, 0), y1, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 2, oh, 0), y2, valid_lanes);
            store_width_masked_8(y_out.ptr(oc + 3, oh, 0), y3, valid_lanes);
        }
    }

}

// Shared stride-2 downsample kernel for both supported shapes (caller passes
// the actual `out_h`/`out_w`): OH_BLOCK=2 row pairs, output columns split
// into an 8-wide main tile plus a narrower tail tile, with stride-2
// activation gather via load_i8_stride2_x8_to_i32 (SSSE3 byte shuffle).
void conv3x3_q8_ocblock4_s2_bn_avx2_out_hw_impl(
    const QuantTensorI8& xq,
    int out_h,
    int out_w,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    // xq: 128×30×34 (pad=1 symmetric + extra_right=4 → w=34 for safe 16-byte vector loads)
    // out_h=14 (always even), out_w=14 (2 tiles: ow=0 full, ow=8 tail)
    // OH_BLOCK=2 + vector stride-2 load via _mm_shuffle_epi8

    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();
    const __m256 zero_v = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float out_scale0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float out_scale1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float out_scale2 = xq.scale * bn.scale[static_cast<size_t>(oc + 2)];
        const float out_scale3 = xq.scale * bn.scale[static_cast<size_t>(oc + 3)];

        const float bias0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float bias1 = bn.bias[static_cast<size_t>(oc + 1)];
        const float bias2 = bn.bias[static_cast<size_t>(oc + 2)];
        const float bias3 = bn.bias[static_cast<size_t>(oc + 3)];

        // OH_BLOCK=2: process pairs; handles odd out_h with a residual single-row pass
        int oh = 0;
        for (; oh + 1 < out_h; oh += 2) {
            const int input_row0 = oh * conv.stride_h;
            const int input_row1 = input_row0 + conv.stride_h;

            for (int ow = 0; ow < out_w; ow += 8) {
                const int valid_lanes = std::min(8, out_w - ow);
                const int input_col_base = ow * conv.stride_w;

                __m256i acc00 = _mm256_setzero_si256();
                __m256i acc01 = _mm256_setzero_si256();
                __m256i acc02 = _mm256_setzero_si256();
                __m256i acc03 = _mm256_setzero_si256();
                __m256i acc10 = _mm256_setzero_si256();
                __m256i acc11 = _mm256_setzero_si256();
                __m256i acc12 = _mm256_setzero_si256();
                __m256i acc13 = _mm256_setzero_si256();

                for (const OCBlock4PatternBucket& bucket : block.buckets) {
                    const int8_t w0 = bucket.w0;
                    const int8_t w1 = bucket.w1;
                    const int8_t w2 = bucket.w2;
                    const int8_t w3 = bucket.w3;

                    for (const OCBlock4PatternAction& action : bucket.actions) {
                        const int ih0 = input_row0 + static_cast<int>(action.ky);
                        const int ih1 = input_row1 + static_cast<int>(action.ky);
                        const int iw  = input_col_base + static_cast<int>(action.kx);
                        const int ic_stride = static_cast<int>(action.ic) * xq.h;
                        const int base0 = (ic_stride + ih0) * xq.w + iw;
                        const int base1 = (ic_stride + ih1) * xq.w + iw;

                        const __m256i xvec0 = load_i8_stride2_x8_to_i32(data + base0);
                        const __m256i xvec1 = load_i8_stride2_x8_to_i32(data + base1);

                        if (w0 == 1) {
                            acc00 = _mm256_add_epi32(acc00, xvec0);
                            acc10 = _mm256_add_epi32(acc10, xvec1);
                        } else if (w0 == -1) {
                            acc00 = _mm256_sub_epi32(acc00, xvec0);
                            acc10 = _mm256_sub_epi32(acc10, xvec1);
                        }
                        if (w1 == 1) {
                            acc01 = _mm256_add_epi32(acc01, xvec0);
                            acc11 = _mm256_add_epi32(acc11, xvec1);
                        } else if (w1 == -1) {
                            acc01 = _mm256_sub_epi32(acc01, xvec0);
                            acc11 = _mm256_sub_epi32(acc11, xvec1);
                        }
                        if (w2 == 1) {
                            acc02 = _mm256_add_epi32(acc02, xvec0);
                            acc12 = _mm256_add_epi32(acc12, xvec1);
                        } else if (w2 == -1) {
                            acc02 = _mm256_sub_epi32(acc02, xvec0);
                            acc12 = _mm256_sub_epi32(acc12, xvec1);
                        }
                        if (w3 == 1) {
                            acc03 = _mm256_add_epi32(acc03, xvec0);
                            acc13 = _mm256_add_epi32(acc13, xvec1);
                        } else if (w3 == -1) {
                            acc03 = _mm256_sub_epi32(acc03, xvec0);
                            acc13 = _mm256_sub_epi32(acc13, xvec1);
                        }
                    }
                }

                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc00), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc01), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y02 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc02), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y03 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc03), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));
                __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc10), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc11), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y12 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc12), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y13 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc13), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));

                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y02 = _mm256_max_ps(y02, zero_v); y03 = _mm256_max_ps(y03, zero_v);
                    y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                    y12 = _mm256_max_ps(y12, zero_v); y13 = _mm256_max_ps(y13, zero_v);
                }

                store_width_masked_8(y_out.ptr(oc + 0, oh,     ow), y00, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh,     ow), y01, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh,     ow), y02, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh,     ow), y03, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 0, oh + 1, ow), y10, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh + 1, ow), y11, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh + 1, ow), y12, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh + 1, ow), y13, valid_lanes);
            }
        }
        // Residual single row when out_h is odd
        if (oh < out_h) {
            const int input_row0 = oh * conv.stride_h;
            for (int ow = 0; ow < out_w; ow += 8) {
                const int valid_lanes = std::min(8, out_w - ow);
                const int input_col_base = ow * conv.stride_w;

                __m256i acc00 = _mm256_setzero_si256();
                __m256i acc01 = _mm256_setzero_si256();
                __m256i acc02 = _mm256_setzero_si256();
                __m256i acc03 = _mm256_setzero_si256();

                for (const OCBlock4PatternBucket& bucket : block.buckets) {
                    for (const OCBlock4PatternAction& action : bucket.actions) {
                        const int ih0 = input_row0 + static_cast<int>(action.ky);
                        const int iw  = input_col_base + static_cast<int>(action.kx);
                        const int base0 = (static_cast<int>(action.ic) * xq.h + ih0) * xq.w + iw;
                        const __m256i xvec0 = load_i8_stride2_x8_to_i32(data + base0);
                        if (bucket.w0 == 1)       acc00 = _mm256_add_epi32(acc00, xvec0);
                        else if (bucket.w0 == -1) acc00 = _mm256_sub_epi32(acc00, xvec0);
                        if (bucket.w1 == 1)       acc01 = _mm256_add_epi32(acc01, xvec0);
                        else if (bucket.w1 == -1) acc01 = _mm256_sub_epi32(acc01, xvec0);
                        if (bucket.w2 == 1)       acc02 = _mm256_add_epi32(acc02, xvec0);
                        else if (bucket.w2 == -1) acc02 = _mm256_sub_epi32(acc02, xvec0);
                        if (bucket.w3 == 1)       acc03 = _mm256_add_epi32(acc03, xvec0);
                        else if (bucket.w3 == -1) acc03 = _mm256_sub_epi32(acc03, xvec0);
                    }
                }

                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc00), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc01), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y02 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc02), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y03 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc03), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));
                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y02 = _mm256_max_ps(y02, zero_v); y03 = _mm256_max_ps(y03, zero_v);
                }
                store_width_masked_8(y_out.ptr(oc + 0, oh, ow), y00, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh, ow), y01, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh, ow), y02, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh, ow), y03, valid_lanes);
            }
        }
    }

}

// Targets layer2's 28x28 stride-1 output: OH_BLOCK=2, OW=28 split into four
// 8-wide tiles with the last one's tail masked to valid_lanes=4.
void conv3x3_q8_ocblock4_layer2_s1_bn_avx2_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    // xq: 128×30×34 (pad=1 symmetric + extra_right=4 for safe 8-byte loads at tail tile)
    // out_h=28 (always even), out_w=28 (4 tiles: ow=0,8,16,24; tail at ow=24, valid=4)
    // OH_BLOCK=2: halves row-loop iterations, improves load ILP
    constexpr int out_h = 28;
    constexpr int out_w = 28;

    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float out_scale0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float out_scale1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float out_scale2 = xq.scale * bn.scale[static_cast<size_t>(oc + 2)];
        const float out_scale3 = xq.scale * bn.scale[static_cast<size_t>(oc + 3)];

        const float bias0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float bias1 = bn.bias[static_cast<size_t>(oc + 1)];
        const float bias2 = bn.bias[static_cast<size_t>(oc + 2)];
        const float bias3 = bn.bias[static_cast<size_t>(oc + 3)];

        const __m256 zero_v = _mm256_setzero_ps();

        // OH_BLOCK=2: out_h=28 is always even
        for (int oh = 0; oh < out_h; oh += 2) {
            for (int ow = 0; ow < out_w; ow += 8) {
                const int valid_lanes = std::min(8, out_w - ow);

                __m256i acc00 = _mm256_setzero_si256();
                __m256i acc01 = _mm256_setzero_si256();
                __m256i acc02 = _mm256_setzero_si256();
                __m256i acc03 = _mm256_setzero_si256();
                __m256i acc10 = _mm256_setzero_si256();
                __m256i acc11 = _mm256_setzero_si256();
                __m256i acc12 = _mm256_setzero_si256();
                __m256i acc13 = _mm256_setzero_si256();

                for (const OCBlock4PatternBucket& bucket : block.buckets) {
                    q8_ocblock4_oh2_dispatch(bucket, oh, ow, xq.h, xq.w, data,
                        acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13);
                }

                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc00), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc01), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y02 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc02), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y03 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc03), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));
                __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc10), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc11), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y12 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc12), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y13 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc13), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));

                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y02 = _mm256_max_ps(y02, zero_v); y03 = _mm256_max_ps(y03, zero_v);
                    y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                    y12 = _mm256_max_ps(y12, zero_v); y13 = _mm256_max_ps(y13, zero_v);
                }

                store_width_masked_8(y_out.ptr(oc + 0, oh,     ow), y00, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh,     ow), y01, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh,     ow), y02, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh,     ow), y03, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 0, oh + 1, ow), y10, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh + 1, ow), y11, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh + 1, ow), y12, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh + 1, ow), y13, valid_lanes);
            }
        }
    }
}

// Targets layer3's 14x14 stride-1 output: OH_BLOCK=2, OW=14 split into an
// 8-wide main tile and a 6-lane tail tile.
void conv3x3_q8_ocblock4_layer3_s1_bn_avx2_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    // xq: 256×16×18 (pad=1 symmetric + extra_right=2 for safe 8-byte loads)
    // out_h=14 (always even), out_w=14 (2 tiles: ow=0 full, ow=8 tail/6 lanes)
    constexpr int out_h = 14;
    constexpr int out_w = 14;

    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float out_scale0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float out_scale1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float out_scale2 = xq.scale * bn.scale[static_cast<size_t>(oc + 2)];
        const float out_scale3 = xq.scale * bn.scale[static_cast<size_t>(oc + 3)];

        const float bias0 = bn.bias[static_cast<size_t>(oc + 0)];
        const float bias1 = bn.bias[static_cast<size_t>(oc + 1)];
        const float bias2 = bn.bias[static_cast<size_t>(oc + 2)];
        const float bias3 = bn.bias[static_cast<size_t>(oc + 3)];

        const __m256 zero_v = _mm256_setzero_ps();

        // OH_BLOCK=2: process pairs of output rows; out_h=14 is always even
        for (int oh = 0; oh < out_h; oh += 2) {
            for (int ow = 0; ow < out_w; ow += 8) {
                const int valid_lanes = std::min(8, out_w - ow);

                __m256i acc00 = _mm256_setzero_si256();
                __m256i acc01 = _mm256_setzero_si256();
                __m256i acc02 = _mm256_setzero_si256();
                __m256i acc03 = _mm256_setzero_si256();
                __m256i acc10 = _mm256_setzero_si256();
                __m256i acc11 = _mm256_setzero_si256();
                __m256i acc12 = _mm256_setzero_si256();
                __m256i acc13 = _mm256_setzero_si256();

                for (const OCBlock4PatternBucket& bucket : block.buckets) {
                    q8_ocblock4_oh2_dispatch(bucket, oh, ow, xq.h, xq.w, data,
                        acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13);
                }

                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc00), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc01), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y02 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc02), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y03 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc03), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));
                __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc10), _mm256_set1_ps(out_scale0)), _mm256_set1_ps(bias0));
                __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc11), _mm256_set1_ps(out_scale1)), _mm256_set1_ps(bias1));
                __m256 y12 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc12), _mm256_set1_ps(out_scale2)), _mm256_set1_ps(bias2));
                __m256 y13 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc13), _mm256_set1_ps(out_scale3)), _mm256_set1_ps(bias3));

                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y02 = _mm256_max_ps(y02, zero_v); y03 = _mm256_max_ps(y03, zero_v);
                    y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                    y12 = _mm256_max_ps(y12, zero_v); y13 = _mm256_max_ps(y13, zero_v);
                }

                store_width_masked_8(y_out.ptr(oc + 0, oh,     ow), y00, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh,     ow), y01, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh,     ow), y02, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh,     ow), y03, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 0, oh + 1, ow), y10, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 1, oh + 1, ow), y11, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 2, oh + 1, ow), y12, valid_lanes);
                store_width_masked_8(y_out.ptr(oc + 3, oh + 1, ow), y13, valid_lanes);
            }
        }
    }
}

// Targets layer1's 56x56 stride-1 output: OH_BLOCK=2 (28 row-pairs), OW=56
// as 7 full 8-wide tiles with no tail (the input's pad layout guarantees the
// last tile's 8-byte load stays in-bounds without extra padding).
void conv3x3_q8_ocblock4_layer1_s1_bn_avx2_impl(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    bool apply_relu,
    Tensor& y_out
) {
    // xq: 64×58×58 (symmetric pad=1, no extra_right_pad: last tile ow=48 reads cols 50..57 < 58)
    // out_h=56, out_w=56: 7 full tiles of 8 columns, no tail; 28 OH-pairs
    constexpr int out_h = 56;
    constexpr int out_w = 56;

    y_out.reuse(conv.out_channels, out_h, out_w);

    const int8_t* const data = xq.data.data();
    const int xq_h = xq.h;
    const int xq_w = xq.w;

    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < static_cast<int>(plan.oc_blocks.size()); ++block_idx) {
        const OCBlock4PatternPlan& block = plan.oc_blocks[block_idx];
        const int oc = block.oc_start;

        const float out_scale0 = xq.scale * bn.scale[static_cast<size_t>(oc + 0)];
        const float out_scale1 = xq.scale * bn.scale[static_cast<size_t>(oc + 1)];
        const float out_scale2 = xq.scale * bn.scale[static_cast<size_t>(oc + 2)];
        const float out_scale3 = xq.scale * bn.scale[static_cast<size_t>(oc + 3)];

        const __m256 s0_v = _mm256_set1_ps(out_scale0);
        const __m256 s1_v = _mm256_set1_ps(out_scale1);
        const __m256 s2_v = _mm256_set1_ps(out_scale2);
        const __m256 s3_v = _mm256_set1_ps(out_scale3);
        const __m256 b0_v = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 0)]);
        const __m256 b1_v = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 1)]);
        const __m256 b2_v = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 2)]);
        const __m256 b3_v = _mm256_set1_ps(bn.bias[static_cast<size_t>(oc + 3)]);
        const __m256 zero_v = _mm256_setzero_ps();

        // out_h=56, out_w=56 → 28 OH pairs, 7 OW tiles all full (56/8=7 exact)
        for (int oh = 0; oh < out_h; oh += 2) {
            for (int ow = 0; ow < out_w; ow += 8) {
                __m256i acc00 = _mm256_setzero_si256();
                __m256i acc01 = _mm256_setzero_si256();
                __m256i acc02 = _mm256_setzero_si256();
                __m256i acc03 = _mm256_setzero_si256();
                __m256i acc10 = _mm256_setzero_si256();
                __m256i acc11 = _mm256_setzero_si256();
                __m256i acc12 = _mm256_setzero_si256();
                __m256i acc13 = _mm256_setzero_si256();

                for (const OCBlock4PatternBucket& bucket : block.buckets) {
                    q8_ocblock4_oh2_dispatch(bucket, oh, ow, xq_h, xq_w, data,
                        acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13);
                }

                __m256 y00 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc00), s0_v), b0_v);
                __m256 y01 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc01), s1_v), b1_v);
                __m256 y02 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc02), s2_v), b2_v);
                __m256 y03 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc03), s3_v), b3_v);
                __m256 y10 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc10), s0_v), b0_v);
                __m256 y11 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc11), s1_v), b1_v);
                __m256 y12 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc12), s2_v), b2_v);
                __m256 y13 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(acc13), s3_v), b3_v);

                if (apply_relu) {
                    y00 = _mm256_max_ps(y00, zero_v); y01 = _mm256_max_ps(y01, zero_v);
                    y02 = _mm256_max_ps(y02, zero_v); y03 = _mm256_max_ps(y03, zero_v);
                    y10 = _mm256_max_ps(y10, zero_v); y11 = _mm256_max_ps(y11, zero_v);
                    y12 = _mm256_max_ps(y12, zero_v); y13 = _mm256_max_ps(y13, zero_v);
                }

                _mm256_storeu_ps(y_out.ptr(oc + 0, oh,     ow), y00);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh,     ow), y01);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh,     ow), y02);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh,     ow), y03);
                _mm256_storeu_ps(y_out.ptr(oc + 0, oh + 1, ow), y10);
                _mm256_storeu_ps(y_out.ptr(oc + 1, oh + 1, ow), y11);
                _mm256_storeu_ps(y_out.ptr(oc + 2, oh + 1, ow), y12);
                _mm256_storeu_ps(y_out.ptr(oc + 3, oh + 1, ow), y13);
            }
        }
    }
}

} // namespace

void conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer4_s1_bn_avx2_impl(xq, conv, plan, bn, false, y_out);
}

void conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer4_s1_bn_avx2_impl(xq, conv, plan, bn, true, y_out);
}

void conv3x3_q8_ocblock4_s2_bn_avx2_out_hw(
    const QuantTensorI8& xq, int out_h, int out_w,
    const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_s2_bn_avx2_out_hw_impl(xq, out_h, out_w, conv, plan, bn, false, y_out);
}

void conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
    const QuantTensorI8& xq, int out_h, int out_w,
    const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_s2_bn_avx2_out_hw_impl(xq, out_h, out_w, conv, plan, bn, true, y_out);
}

void conv3x3_q8_ocblock4_layer3_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer3_s1_bn_avx2_impl(xq, conv, plan, bn, false, y_out);
}

void conv3x3_q8_ocblock4_layer3_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer3_s1_bn_avx2_impl(xq, conv, plan, bn, true, y_out);
}

void conv3x3_q8_ocblock4_layer2_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer2_s1_bn_avx2_impl(xq, conv, plan, bn, false, y_out);
}

void conv3x3_q8_ocblock4_layer2_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer2_s1_bn_avx2_impl(xq, conv, plan, bn, true, y_out);
}

void conv3x3_q8_ocblock4_layer1_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer1_s1_bn_avx2_impl(xq, conv, plan, bn, false, y_out);
}

void conv3x3_q8_ocblock4_layer1_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace&, Tensor& y_out
) {
    conv3x3_q8_ocblock4_layer1_s1_bn_avx2_impl(xq, conv, plan, bn, true, y_out);
}