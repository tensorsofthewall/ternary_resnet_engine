// Blocked engine: NCHWc16 activations + mullo_epi32 ternary conv prototype.
// Phase 2: q8 runtime + reference scalar conv3x3 smoke test.
// Forward pass added in Phase 4.

#include "blocked_model.hpp"
#include "blocked_conv3x3.hpp"
#include "model_loader.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin>\n";
        return 1;
    }

    try {
        const Model model = load_model(argv[1]);
        std::cout << "Model loaded: " << model.convs.size() << " conv layers\n";

        // Build blocked weights for layer2.1.conv2.
        const auto& conv = model.convs.at("layer2.1.conv2");
        const BlockedConvWeights bw = build_blocked_conv_weights(conv);
        const BNFusedParams& bn     = model.fused_bns.at("layer2.1.bn2");

        std::cout << "layer2.1.conv2: "
                  << bw.oc_blocks << " oc_blocks × "
                  << bw.ic_blocks << " ic_blocks × "
                  << bw.kh << "×" << bw.kw << "\n";

        // Phase 2 smoke test: run reference conv3x3_s1 on a zero input.
        // IC=128, OH=OW=28 → input CB=8, H=28, W=28.
        BlockedTensorFP32 dummy_input(8, 28, 28);
        BlockedTensorFP32 output;
        blocked_conv3x3_s1_ref(dummy_input, bw, bn, /*relu=*/true, output);

        std::cout << "blocked_conv3x3_s1_ref OK: output "
                  << output.CB << "×" << output.H << "×" << output.W << "×16\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
