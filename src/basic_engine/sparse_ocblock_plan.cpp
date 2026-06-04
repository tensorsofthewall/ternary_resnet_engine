#include "sparse_ocblock_plan.hpp"


#include <stdexcept>

namespace {
    size_t weight_idx(const Conv2DParams& conv, int oc, int ic, int ky, int kx) {
        return static_cast<size_t>(((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx);
    }

    bool supports_ocblock4_plan(const Conv2DParams& conv) {
        // For simplicity, we only support 3x3s1 convs with output channels divisible by 4 for OCBlock4Plan
        return conv.kh == 3 && conv.kw == 3 && conv.stride_h == 1 && conv.stride_w == 1 && conv.dilation_h == 1 && conv.dilation_w == 1 && conv.out_channels % 4 == 0;
    }
}

SparseOCBlock4ConvPlan build_sparse_ocblock4_conv_plan(const Conv2DParams& conv) {
    if (!supports_ocblock4_plan(conv)) {
        throw std::runtime_error("Sparse OCBlock4 plan requested for unsupported conv layer: " + conv.name);
    }

    SparseOCBlock4ConvPlan plan;
    plan.name = conv.name;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );

    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error("Weight size mismatch while building sparse OC-block4 plan: " + conv.name);
    }

    plan.oc_blocks.reserve(static_cast<size_t>(conv.out_channels / 4));

    for (int oc = 0; oc < conv.out_channels; oc += 4) {
        OCBlock4Plan block;
        block.oc_start = oc;

        for (int ic = 0; ic < conv.in_channels; ++ic) {
            for (int ky = 0; ky < conv.kh; ++ky) {
                for (int kx = 0; kx < conv.kw; ++kx) {
                    const int8_t w0 = conv.weight[weight_idx(conv, oc, ic, ky, kx)];
                    const int8_t w1 = conv.weight[weight_idx(conv, oc + 1, ic, ky, kx)];
                    const int8_t w2 = conv.weight[weight_idx(conv, oc + 2, ic, ky, kx)];
                    const int8_t w3 = conv.weight[weight_idx(conv, oc + 3, ic, ky, kx)];

                    if (w0 == 0 && w1 == 0 && w2 == 0 && w3 == 0) {
                        continue; // skip zero-weight positions
                    }

                    if ((w0 < -1 || w0 > 1) || (w1 < -1 || w1 > 1) || (w2 < -1 || w2 > 1) || (w3 < -1 || w3 > 1)) {
                        throw std::runtime_error("Non-ternary weight found while building sparse OC-block4 plan: " + conv.name);
                    }

                    block.actions.push_back(
                        OCBlock4Action{
                            static_cast<uint16_t>(ic),
                            static_cast<uint8_t>(ky),
                            static_cast<uint8_t>(kx),
                            w0,
                            w1,
                            w2,
                            w3
                        }
                    );
                }
            }
        }
        plan.oc_blocks.push_back(std::move(block));
    }
    return plan;
}

void build_sparse_ocblock4_conv_plans(Model& model) {
    model.sparse_ocblock4_plans.clear();

    for (const auto& [name, conv] : model.convs) {
        if (!supports_ocblock4_plan(conv)) {
            continue;
        }

        model.sparse_ocblock4_plans.emplace(
            name,
            build_sparse_ocblock4_conv_plan(conv)
        );
    }
}