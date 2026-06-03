#include "conv_plan.hpp"

#include <stdexcept>

namespace {

size_t conv_weight_index(
    const Conv2DParams& conv,
    int oc,
    int ic,
    int ky,
    int kx
) {
    return static_cast<size_t>(((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx);
}

} // namespace

ConvPlan build_conv_plan(const Conv2DParams& conv) {
    ConvPlan plan;
    plan.name = conv.name;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;

    plan.oc_plans.resize(static_cast<size_t>(conv.out_channels));

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );

    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error("Conv weight size mismatch while building ConvPlan: " + conv.name);
    }

    for (int oc = 0; oc < conv.out_channels; ++oc) {
        ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

        for (int ic = 0; ic < conv.in_channels; ++ic) {
            for (int ky = 0; ky < conv.kh; ++ky) {
                for (int kx = 0; kx < conv.kw; ++kx) {
                    const int8_t w = conv.weight[conv_weight_index(conv, oc, ic, ky, kx)];

                    if (w == 1) {
                        oc_plan.pos.push_back(
                            WeightIndex{
                                static_cast<uint16_t>(ic),
                                static_cast<uint8_t>(ky),
                                static_cast<uint8_t>(kx),
                            }
                        );
                    } else if (w == -1) {
                        oc_plan.neg.push_back(
                            WeightIndex{
                                static_cast<uint16_t>(ic),
                                static_cast<uint8_t>(ky),
                                static_cast<uint8_t>(kx),
                            }
                        );
                    } else if (w != 0) {
                        throw std::runtime_error("Non-ternary weight found in ConvPlan: " + conv.name);
                    }
                }
            }
        }
    }

    return plan;
}

void build_conv_plans(Model& model) {
    model.conv_plans.clear();

    for (const auto& [name, conv] : model.convs) {
        model.conv_plans.emplace(name, build_conv_plan(conv));
    }
}