#include "blocked_model.hpp"

#include <stdexcept>

BlockedConvWeights build_blocked_conv_weights(const Conv2DParams& conv) {
    const int OC = conv.out_channels;
    const int IC = conv.in_channels;
    const int KH = conv.kh;
    const int KW = conv.kw;

    if (OC % 16 != 0)
        throw std::runtime_error("blocked engine: OC must be divisible by 16 for: " + conv.name);
    if (IC % 16 != 0)
        throw std::runtime_error("blocked engine: IC must be divisible by 16 for: " + conv.name);

    BlockedConvWeights bw;
    bw.oc_blocks = OC / 16;
    bw.ic_blocks = IC / 16;
    bw.kh = KH;
    bw.kw = KW;
    bw.data.assign(
        static_cast<size_t>(bw.oc_blocks * bw.ic_blocks * KH * KW) * 256, 0);

    // Source layout: conv.weight[oc * IC * KH * KW + ic * KH * KW + ky * KW + kx]
    // Dest layout:   block(oc_cb, ic_cb, ky, kx)[ci_in * 16 + ci_out]
    for (int oc = 0; oc < OC; ++oc) {
        const int oc_cb  = oc / 16;
        const int ci_out = oc % 16;
        for (int ic = 0; ic < IC; ++ic) {
            const int ic_cb = ic / 16;
            const int ci_in = ic % 16;
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    const size_t src_idx = static_cast<size_t>(
                        ((oc * IC + ic) * KH + ky) * KW + kx);
                    bw.block(oc_cb, ic_cb, ky, kx)[ci_in * 16 + ci_out] =
                        conv.weight[src_idx];
                }
            }
        }
    }

    return bw;
}
