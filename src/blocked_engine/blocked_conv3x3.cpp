#include "blocked_conv3x3.hpp"
#include "q8_runtime.hpp"

#include <algorithm>
#include <cstdint>

namespace {

// Shared scalar implementation for any KH/KW/stride/pad combination.
static void blocked_conv_ref_impl(
    const BlockedTensorI8& xq,
    const BlockedConvWeights& bw,
    const BNFusedParams& bn,
    bool relu,
    int pad,
    int stride,
    BlockedTensorFP32& output)
{
    const int KH = bw.kh;
    const int KW = bw.kw;
    const int OH = (xq.H + 2 * pad - KH) / stride + 1;
    const int OW = (xq.W + 2 * pad - KW) / stride + 1;

    output.reuse(bw.oc_blocks, OH, OW);

    // Returns 0 for out-of-bounds — avoids materialising a padded copy.
    auto get_a = [&](int cb, int h, int w, int ci) -> int8_t {
        if (h < 0 || h >= xq.H || w < 0 || w >= xq.W) return 0;
        return xq.pixel(cb, h, w)[ci];
    };

    for (int oc_cb = 0; oc_cb < bw.oc_blocks; ++oc_cb) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int32_t acc[16] = {};

                for (int ic_cb = 0; ic_cb < bw.ic_blocks; ++ic_cb) {
                    for (int ky = 0; ky < KH; ++ky) {
                        const int ih = oh * stride + ky - pad;
                        for (int kx = 0; kx < KW; ++kx) {
                            const int iw = ow * stride + kx - pad;
                            const int8_t* tile = bw.block(oc_cb, ic_cb, ky, kx);
                            for (int ci_in = 0; ci_in < 16; ++ci_in) {
                                const int32_t a =
                                    static_cast<int32_t>(get_a(ic_cb, ih, iw, ci_in));
                                if (a == 0) continue;
                                for (int ci_out = 0; ci_out < 16; ++ci_out)
                                    acc[ci_out] += static_cast<int32_t>(
                                        tile[ci_in * 16 + ci_out]) * a;
                            }
                        }
                    }
                }

                float* out_pix = output.pixel(oc_cb, oh, ow);
                for (int ci_out = 0; ci_out < 16; ++ci_out) {
                    const int oc  = oc_cb * 16 + ci_out;
                    float val = static_cast<float>(acc[ci_out]) * xq.scale
                                * bn.scale[oc] + bn.bias[oc];
                    if (relu && val < 0.0f) val = 0.0f;
                    out_pix[ci_out] = val;
                }
            }
        }
    }
}

} // namespace

void blocked_conv3x3_s1_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_ref_impl(xq, w, bn, relu, /*pad=*/1, /*stride=*/1, output);
}

void blocked_conv3x3_s2_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_ref_impl(xq, w, bn, relu, /*pad=*/1, /*stride=*/2, output);
}

void blocked_conv1x1_s2_ref(
    const BlockedTensorFP32& input,
    const BlockedConvWeights& w,
    const BNFusedParams& bn,
    bool relu,
    BlockedTensorFP32& output)
{
    BlockedTensorI8 xq;
    quantize_blocked_fp32_to_i8(input, xq);
    blocked_conv_ref_impl(xq, w, bn, relu, /*pad=*/0, /*stride=*/2, output);
}
