#include "conv1_ocblock4_plan.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace {

size_t weight_index(
    const Conv2DParams& conv,
    int oc,
    int ic,
    int ky,
    int kx
) {
    return static_cast<size_t>(
        ((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx
    );
}

bool supports_conv1_ocblock4_plan(const Conv2DParams& conv) {
    return conv.name == "conv1" &&
           conv.in_channels == 3 &&
           conv.out_channels == 64 &&
           conv.kh == 7 &&
           conv.kw == 7 &&
           conv.stride_h == 2 &&
           conv.stride_w == 2 &&
           conv.pad_h == 3 &&
           conv.pad_w == 3 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 4 == 0;
}

int ternary_digit(int8_t w) {
    return static_cast<int>(w) + 1; // -1,0,+1 -> 0,1,2
}

int pattern_index(int8_t w0, int8_t w1, int8_t w2, int8_t w3) {
    return (((ternary_digit(w0) * 3 + ternary_digit(w1)) * 3 +
             ternary_digit(w2)) * 3 + ternary_digit(w3));
}

void pattern_from_index(
    int idx,
    int8_t& w0,
    int8_t& w1,
    int8_t& w2,
    int8_t& w3
) {
    const int d3 = idx % 3;
    idx /= 3;
    const int d2 = idx % 3;
    idx /= 3;
    const int d1 = idx % 3;
    idx /= 3;
    const int d0 = idx % 3;

    w0 = static_cast<int8_t>(d0 - 1);
    w1 = static_cast<int8_t>(d1 - 1);
    w2 = static_cast<int8_t>(d2 - 1);
    w3 = static_cast<int8_t>(d3 - 1);
}

} // namespace

Conv1OCBlock4ConvPlan build_conv1_ocblock4_plan(const Conv2DParams& conv) {
    if (!supports_conv1_ocblock4_plan(conv)) {
        throw std::runtime_error(
            "Conv1 OC-Block4 plan requested for unsupported conv: " + conv.name
        );
    }

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );

    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error(
            "Weight size mismatch while building conv1 OC-Block4 plan: " + conv.name
        );
    }

    Conv1OCBlock4ConvPlan plan;
    plan.name = conv.name;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;

    plan.oc_blocks.reserve(static_cast<size_t>(conv.out_channels / 4));

    for (int oc = 0; oc < conv.out_channels; oc += 4) {
        std::array<std::vector<Conv1OCBlock4Action>, 81> bucket_actions;

        for (int ic = 0; ic < conv.in_channels; ++ic) {
            for (int ky = 0; ky < conv.kh; ++ky) {
                for (int kx = 0; kx < conv.kw; ++kx) {
                    const int8_t w0 = conv.weight[weight_index(conv, oc + 0, ic, ky, kx)];
                    const int8_t w1 = conv.weight[weight_index(conv, oc + 1, ic, ky, kx)];
                    const int8_t w2 = conv.weight[weight_index(conv, oc + 2, ic, ky, kx)];
                    const int8_t w3 = conv.weight[weight_index(conv, oc + 3, ic, ky, kx)];

                    if ((w0 < -1 || w0 > 1) ||
                        (w1 < -1 || w1 > 1) ||
                        (w2 < -1 || w2 > 1) ||
                        (w3 < -1 || w3 > 1)) {
                        throw std::runtime_error(
                            "Non-ternary weight in conv1 OC-Block4 plan: " + conv.name
                        );
                    }

                    if (w0 == 0 && w1 == 0 && w2 == 0 && w3 == 0) {
                        continue;
                    }

                    const int idx = pattern_index(w0, w1, w2, w3);

                    bucket_actions[static_cast<size_t>(idx)].push_back(
                        Conv1OCBlock4Action{
                            static_cast<uint8_t>(ic),
                            static_cast<uint8_t>(ky),
                            static_cast<uint8_t>(kx)
                        }
                    );
                }
            }
        }

        Conv1OCBlock4Plan block;
        block.oc_start = oc;

        for (int idx = 0; idx < 81; ++idx) {
            auto& actions = bucket_actions[static_cast<size_t>(idx)];

            if (actions.empty()) {
                continue;
            }

            int8_t w0 = 0;
            int8_t w1 = 0;
            int8_t w2 = 0;
            int8_t w3 = 0;

            pattern_from_index(idx, w0, w1, w2, w3);

            if (w0 == 0 && w1 == 0 && w2 == 0 && w3 == 0) {
                continue;
            }

            Conv1OCBlock4PatternBucket bucket;
            bucket.pattern_id = idx;
            bucket.w0 = w0;
            bucket.w1 = w1;
            bucket.w2 = w2;
            bucket.w3 = w3;
            bucket.actions = std::move(actions);

            block.buckets.push_back(std::move(bucket));
        }

        std::sort(
            block.buckets.begin(),
            block.buckets.end(),
            [](const Conv1OCBlock4PatternBucket& a,
               const Conv1OCBlock4PatternBucket& b) {
                return a.actions.size() > b.actions.size();
            }
        );

        plan.oc_blocks.push_back(std::move(block));
    }

    return plan;
}

void build_conv1_ocblock4_plans(Model& model) {
    model.conv1_ocblock4_plans.clear();

    for (const auto& [name, conv] : model.convs) {
        if (!supports_conv1_ocblock4_plan(conv)) {
            continue;
        }

        model.conv1_ocblock4_plans.emplace(
            name,
            build_conv1_ocblock4_plan(conv)
        );
    }
}