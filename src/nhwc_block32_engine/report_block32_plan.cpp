// Sanity tool for the AVX2 NHWC block32 ternary q8 engine: builds a
// Block32ConvPlan for every ResNet18 conv whose in_channels is a
// multiple of 32 (i.e. every conv except conv1) and reports, per layer, how
// many (ky,kx,ic_block) reduction blocks are kept vs. dropped as all-zero.
//
// Usage: ./report_block32_plan <model.bin>

#include "model_loader.hpp"
#include "model.hpp"
#include "block32_plan.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using nhwc_block32::build_block32_plan;
using nhwc_block32::can_use_block32_plan;
using nhwc_block32::Block32ConvPlan;

namespace {

const std::vector<std::string> kConvOrder = {
    "conv1",
    "layer1.0.conv1", "layer1.0.conv2",
    "layer1.1.conv1", "layer1.1.conv2",
    "layer2.0.conv1", "layer2.0.conv2", "layer2.0.downsample.0",
    "layer2.1.conv1", "layer2.1.conv2",
    "layer3.0.conv1", "layer3.0.conv2", "layer3.0.downsample.0",
    "layer3.1.conv1", "layer3.1.conv2",
    "layer4.0.conv1", "layer4.0.conv2", "layer4.0.downsample.0",
    "layer4.1.conv1", "layer4.1.conv2",
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    const Model model = load_model(argv[1]);

    long total_kept_overall = 0;
    long total_possible_overall = 0;

    for (const std::string& name : kConvOrder) {
        const Conv2DParams& conv = model.convs.at(name);

        if (!can_use_block32_plan(conv)) {
            std::printf("%-24s in=%-4d out=%-4d %dx%d  -- skipped (in_channels not a multiple of 32)\n",
                         name.c_str(), conv.in_channels, conv.out_channels, conv.kh, conv.kw);
            continue;
        }

        const Block32ConvPlan plan = build_block32_plan(conv);

        const long total_possible =
            static_cast<long>(plan.out_channels) * plan.ic_blocks * plan.kh * plan.kw;

        long total_kept = 0;
        for (const auto& oc_plan : plan.oc_plans) {
            total_kept += static_cast<long>(oc_plan.blocks.size());
        }

        const double sparsity_pct =
            100.0 * (1.0 - static_cast<double>(total_kept) / static_cast<double>(total_possible));
        const double blocks_per_oc =
            static_cast<double>(total_kept) / static_cast<double>(plan.out_channels);

        std::printf("%-24s in=%-4d out=%-4d %dx%d s=%d  ic_blocks=%-3d  "
                    "blocks kept=%-6ld / %-6ld (%.1f%% sparse)  avg/oc=%.2f\n",
                    name.c_str(), plan.in_channels, plan.out_channels, plan.kh, plan.kw,
                    plan.stride_h, plan.ic_blocks, total_kept, total_possible, sparsity_pct,
                    blocks_per_oc);

        total_kept_overall += total_kept;
        total_possible_overall += total_possible;
    }

    const double overall_sparsity_pct =
        100.0 * (1.0 - static_cast<double>(total_kept_overall) / static_cast<double>(total_possible_overall));

    std::printf("\nOverall (excluding conv1): blocks kept=%ld / %ld (%.1f%% sparse)\n",
                total_kept_overall, total_possible_overall, overall_sparsity_pct);

    return 0;
}
