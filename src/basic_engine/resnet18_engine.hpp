#pragma once

#include <vector>
#include "model.hpp"
#include "tensor.hpp"
#include "timing.hpp"

#include "workspace.hpp"

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input, EngineWorkspace& workspace);

Tensor forward_resnet18_scalar_timed(
    const Model& model,
    const Tensor& input,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
);