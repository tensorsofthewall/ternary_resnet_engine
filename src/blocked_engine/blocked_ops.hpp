#pragma once

#include <vector>
#include "blocked_tensor.hpp"
#include "model.hpp"

// MaxPool 3×3, stride 2, pad 1. Used after conv1: 64×112×112 → 64×56×56.
void blocked_maxpool_3x3_s2(const BlockedTensorFP32& input, BlockedTensorFP32& output);

// Global average pool: (CB, H, W) → (CB, 1, 1). Averages over all H×W spatial positions.
void blocked_avgpool_global(const BlockedTensorFP32& input, BlockedTensorFP32& output);

// Fully connected layer. Input must be (CB, 1, 1) from avgpool (C = CB*16 flat values).
void blocked_fc(const BlockedTensorFP32& input, const LinearParams& fc,
                std::vector<float>& logits);

// In-place elementwise residual add ± ReLU. x and skip must have identical shape.
void blocked_residual_add_relu(BlockedTensorFP32& x, const BlockedTensorFP32& skip);
void blocked_residual_add(BlockedTensorFP32& x, const BlockedTensorFP32& skip);

// In-place softmax over a flat logit vector.
void softmax_inplace(std::vector<float>& x);
