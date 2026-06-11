#include "block32_plan.hpp"

#include <stdexcept>

namespace nhwc_block32 {

namespace {

size_t conv_weight_index(const Conv2DParams& conv, int oc, int ic, int ky, int kx) {
    return static_cast<size_t>(((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx);
}

} // namespace

bool can_use_block32_plan(const Conv2DParams& conv) {
    return conv.in_channels % BLOCK32 == 0;
}

Block32ConvPlan build_block32_plan(const Conv2DParams& conv) {
    if (!can_use_block32_plan(conv)) {
        throw std::runtime_error(
            "build_block32_plan: in_channels not a multiple of " +
            std::to_string(BLOCK32) + " for conv: " + conv.name);
    }

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );
    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error("Conv weight size mismatch while building Block32ConvPlan: " + conv.name);
    }

    Block32ConvPlan plan;
    plan.name = conv.name;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;
    plan.ic_blocks = conv.in_channels / BLOCK32;

    plan.oc_plans.resize(static_cast<size_t>(conv.out_channels));

    for (int oc = 0; oc < conv.out_channels; ++oc) {
        OCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

        for (int ky = 0; ky < conv.kh; ++ky) {
            for (int kx = 0; kx < conv.kw; ++kx) {
                for (int ic_block = 0; ic_block < plan.ic_blocks; ++ic_block) {
                    Block32 block;
                    bool any_nonzero = false;

                    for (int ci = 0; ci < BLOCK32; ++ci) {
                        const int ic = ic_block * BLOCK32 + ci;
                        const int8_t w = conv.weight[conv_weight_index(conv, oc, ic, ky, kx)];

                        if (w < -1 || w > 1) {
                            throw std::runtime_error(
                                "Non-ternary weight value while building Block32ConvPlan: " + conv.name);
                        }

                        block.w[ci] = w;
                        if (w != 0) any_nonzero = true;
                    }

                    if (!any_nonzero) continue;

                    block.ky = static_cast<uint8_t>(ky);
                    block.kx = static_cast<uint8_t>(kx);
                    block.ic_block = static_cast<uint16_t>(ic_block);
                    oc_plan.blocks.push_back(block);
                }
            }
        }
    }

    return plan;
}

} // namespace nhwc_block32
