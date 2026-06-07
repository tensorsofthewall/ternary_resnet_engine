#include "blocked_conv1.hpp"

#include <stdexcept>

void blocked_conv1_7x7_s2(
    const Tensor& input,
    const Conv2DParams& conv,
    const BNFusedParams& bn,
    BlockedTensorFP32& output)
{
    const int OC = conv.out_channels;
    const int IC = conv.in_channels;
    const int KH = conv.kh;
    const int KW = conv.kw;
    const int SH = conv.stride_h;
    const int SW = conv.stride_w;
    const int PH = conv.pad_h;
    const int PW = conv.pad_w;
    const int IH = input.h;
    const int IW = input.w;
    const int OH = (IH + 2 * PH - KH) / SH + 1;
    const int OW = (IW + 2 * PW - KW) / SW + 1;

    if (OC % 16 != 0)
        throw std::runtime_error("blocked_conv1: OC must be divisible by 16");

    output.reuse(OC / 16, OH, OW);

    // Weights stored as int8 {-1, 0, +1} in OICHW order.
    // Cast to float for direct FP32 multiply with the input.
    for (int oc = 0; oc < OC; ++oc) {
        const int oc_cb  = oc / 16;
        const int ci_out = oc % 16;

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float acc = 0.0f;

                for (int ic = 0; ic < IC; ++ic) {
                    for (int ky = 0; ky < KH; ++ky) {
                        const int ih = oh * SH + ky - PH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kx = 0; kx < KW; ++kx) {
                            const int iw = ow * SW + kx - PW;
                            if (iw < 0 || iw >= IW) continue;
                            const size_t widx = static_cast<size_t>(
                                ((oc * IC + ic) * KH + ky) * KW + kx);
                            acc += static_cast<float>(conv.weight[widx])
                                   * input(ic, ih, iw);
                        }
                    }
                }

                // BN + ReLU (conv1 is always followed by ReLU in ResNet18).
                float val = acc * bn.scale[oc] + bn.bias[oc];
                if (val < 0.0f) val = 0.0f;
                output.pixel(oc_cb, oh, ow)[ci_out] = val;
            }
        }
    }
}
