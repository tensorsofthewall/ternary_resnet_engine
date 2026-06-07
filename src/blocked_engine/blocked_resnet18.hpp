#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include "blocked_tensor.hpp"
#include "blocked_model.hpp"
#include "model.hpp"
#include "tensor.hpp"

// All blocked weights for the full ResNet18 (conv1 excluded — handled separately).
struct BlockedModel {
    std::unordered_map<std::string, BlockedConvWeights> blocked_convs;
    const Model* model = nullptr;  // lifetime must exceed BlockedModel
};

// Builds blocked (16x16-tiled) weight copies for every conv in `m` whose
// in/out channel counts are multiples of 16 (i.e. every conv except conv1,
// which keeps its dense Conv2DParams form and is handled by
// blocked_conv1_7x7_s2). `bm.model` aliases `m` -- `m` must outlive `bm`.
BlockedModel build_blocked_model(const Model& m);

// Pre-allocated activation buffers for one forward pass.
// All tensors are zero-sized on construction and sized by the first forward call.
struct BlockedEngineWorkspace {
    BlockedTensorFP32 conv1_out;   // 64×112×112

    BlockedTensorFP32 pool_out;    // 64×56×56

    BlockedTensorFP32 l1_0_c1, l1_0_c2;           // 64×56×56
    BlockedTensorFP32 l1_1_c1, l1_1_c2;           // 64×56×56

    BlockedTensorFP32 l2_0_c1, l2_0_c2, l2_0_ds;  // 128×28×28
    BlockedTensorFP32 l2_1_c1, l2_1_c2;            // 128×28×28

    BlockedTensorFP32 l3_0_c1, l3_0_c2, l3_0_ds;  // 256×14×14
    BlockedTensorFP32 l3_1_c1, l3_1_c2;            // 256×14×14

    BlockedTensorFP32 l4_0_c1, l4_0_c2, l4_0_ds;  // 512×7×7
    BlockedTensorFP32 l4_1_c1, l4_1_c2;            // 512×7×7

    BlockedTensorFP32 pool_final;  // 512×1×1
};

// Run one full ResNet18 forward pass with the reference scalar kernels.
// Returns softmax probabilities (1000 values).
// timings is filled with wall-clock ms per named section (may be left empty by the caller).
std::vector<float> forward_blocked_resnet18(
    const BlockedModel& bm,
    const Tensor& input,
    BlockedEngineWorkspace& ws,
    std::unordered_map<std::string, double>& timings);
