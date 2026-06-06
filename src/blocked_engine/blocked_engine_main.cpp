// Blocked engine: NCHWc16 activations + mullo_epi32 ternary conv prototype.
// Phase 1 stub: verifies types compile and model loads.
// Forward pass and AVX2 kernel added in Phases 2–5.

#include "blocked_model.hpp"
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

        // Phase 1 compile test: build blocked weights for layer2.1.conv2.
        const auto& conv = model.convs.at("layer2.1.conv2");
        const BlockedConvWeights bw = build_blocked_conv_weights(conv);
        std::cout << "layer2.1.conv2: "
                  << bw.oc_blocks << " oc_blocks × "
                  << bw.ic_blocks << " ic_blocks × "
                  << bw.kh << "×" << bw.kw << " = "
                  << bw.data.size() << " bytes\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
