#include "pattern_ocblock2_plan.hpp"

#include <array>
#include <stdexcept>
#include <utility>
#include <algorithm>

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

bool supports_pattern_ocblock2_plan(const Conv2DParams& conv) {
    return conv.kh == 3 &&
           conv.kw == 3 &&
           conv.stride_h == 1 &&
           conv.stride_w == 1 &&
           conv.dilation_h == 1 &&
           conv.dilation_w == 1 &&
           conv.out_channels % 2 == 0;
}

int pattern_index(int8_t w0, int8_t w1) {
    // Map -1,0,+1 -> 0,1,2
    const int a = static_cast<int>(w0) + 1;
    const int b = static_cast<int>(w1) + 1;
    return a * 3 + b;
}

std::pair<int8_t, int8_t> pattern_from_index(int idx) {
    const int a = idx / 3;
    const int b = idx % 3;

    return {
        static_cast<int8_t>(a - 1),
        static_cast<int8_t>(b - 1)
    };
}

} // namespace

PatternOCBlock2ConvPlan build_pattern_ocblock2_plan(const Conv2DParams& conv) {
    if (!supports_pattern_ocblock2_plan(conv)) {
        throw std::runtime_error(
            "Pattern OC-block2 plan requested for unsupported conv: " + conv.name
        );
    }

    const size_t expected_weight_count = static_cast<size_t>(
        conv.out_channels * conv.in_channels * conv.kh * conv.kw
    );

    if (conv.weight.size() != expected_weight_count) {
        throw std::runtime_error(
            "Weight size mismatch while building pattern OC-block2 plan: " + conv.name
        );
    }

    PatternOCBlock2ConvPlan plan;
    plan.name = conv.name;
    plan.in_channels = conv.in_channels;
    plan.out_channels = conv.out_channels;
    plan.kh = conv.kh;
    plan.kw = conv.kw;
    plan.stride_h = conv.stride_h;
    plan.stride_w = conv.stride_w;
    plan.pad_h = conv.pad_h;
    plan.pad_w = conv.pad_w;

    plan.oc_blocks.reserve(static_cast<size_t>(conv.out_channels / 2));

    for (int oc = 0; oc < conv.out_channels; oc += 2) {
        std::array<std::vector<OCBlock2Action>, 9> bucket_actions;

        for (int ic = 0; ic < conv.in_channels; ++ic) {
            for (int ky = 0; ky < conv.kh; ++ky) {
                for (int kx = 0; kx < conv.kw; ++kx) {
                    const int8_t w0 = conv.weight[
                        weight_index(conv, oc + 0, ic, ky, kx)
                    ];
                    const int8_t w1 = conv.weight[
                        weight_index(conv, oc + 1, ic, ky, kx)
                    ];

                    if ((w0 < -1 || w0 > 1) || (w1 < -1 || w1 > 1)) {
                        throw std::runtime_error(
                            "Non-ternary weight in pattern OC-block2 plan: " + conv.name
                        );
                    }

                    if (w0 == 0 && w1 == 0) {
                        continue;
                    }

                    const int idx = pattern_index(w0, w1);

                    bucket_actions[static_cast<size_t>(idx)].push_back(
                        OCBlock2Action{
                            static_cast<uint16_t>(ic),
                            static_cast<uint8_t>(ky),
                            static_cast<uint8_t>(kx)
                        }
                    );
                }
            }
        }

        OCBlock2PatternPlan block;
        block.oc_start = oc;

        for (int idx = 0; idx < 9; ++idx) {
            auto& actions = bucket_actions[static_cast<size_t>(idx)];

            if (actions.empty()) {
                continue;
            }

            const auto [w0, w1] = pattern_from_index(idx);

            if (w0 == 0 && w1 == 0) {
                continue;
            }

            OCBlock2PatternBucket bucket;
            bucket.pattern_id = idx;
            bucket.w0 = w0;
            bucket.w1 = w1;
            bucket.actions = std::move(actions);

            block.buckets.push_back(std::move(bucket));
        }
        // Sort buckets by size for more efficient dispatching in the kernel
        std::sort(
            block.buckets.begin(),
            block.buckets.end(),
            [](const OCBlock2PatternBucket& a, const OCBlock2PatternBucket& b) {
                return a.actions.size() > b.actions.size();
            }
        );
        plan.oc_blocks.push_back(std::move(block));
    }

    return plan;
}

void build_pattern_ocblock2_plans(Model& model) {
    model.pattern_ocblock2_plans.clear();

    for (const auto& [name, conv] : model.convs) {
        if (!supports_pattern_ocblock2_plan(conv)) {
            continue;
        }

        model.pattern_ocblock2_plans.emplace(
            name,
            build_pattern_ocblock2_plan(conv)
        );
    }
}