#pragma once

#include <vector>
#include "model.hpp"
#include "tensor.hpp"
#include "timing.hpp"

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input);

Tensor forward_resnet18_scalar_timed(
    const Model& model,
    const Tensor& input,
    std::vector<TimingEntry>& timings
);