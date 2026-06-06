// Blocked engine: NCHWc16 activations + mullo_epi32 ternary conv prototype.
// Phase 3: supporting ops + conv1 smoke test.
// Forward pass added in Phase 4.

#include "blocked_model.hpp"
#include "blocked_conv3x3.hpp"
#include "blocked_conv1.hpp"
#include "blocked_ops.hpp"
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

        // Phase 2 check: reference scalar conv3x3_s1 on zero input.
        {
            const BlockedConvWeights bw = build_blocked_conv_weights(
                model.convs.at("layer2.1.conv2"));
            const BNFusedParams& bn = model.fused_bns.at("layer2.1.bn2");
            BlockedTensorFP32 dummy(8, 28, 28), out;
            blocked_conv3x3_s1_ref(dummy, bw, bn, true, out);
            std::cout << "conv3x3_s1_ref OK: " << out.CB << "×" << out.H << "×" << out.W << "×16\n";
        }

        // Phase 3 check: conv1 on zero input.
        {
            const Conv2DParams& c1 = model.convs.at("conv1");
            const BNFusedParams& b1 = model.fused_bns.at("bn1");
            Tensor dummy_input(3, 224, 224);
            BlockedTensorFP32 out;
            blocked_conv1_7x7_s2(dummy_input, c1, b1, out);
            std::cout << "conv1_7x7_s2 OK: " << out.CB << "×" << out.H << "×" << out.W << "×16\n";
        }

        // Phase 3 check: maxpool + avgpool shape.
        {
            BlockedTensorFP32 pool_in(4, 112, 112);   // 64×112×112
            BlockedTensorFP32 pool_out;
            blocked_maxpool_3x3_s2(pool_in, pool_out);
            std::cout << "maxpool OK: " << pool_out.CB << "×" << pool_out.H << "×" << pool_out.W << "×16\n";

            BlockedTensorFP32 avg_out;
            blocked_avgpool_global(pool_out, avg_out);
            std::cout << "avgpool OK: " << avg_out.CB << "×" << avg_out.H << "×" << avg_out.W << "×16\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
