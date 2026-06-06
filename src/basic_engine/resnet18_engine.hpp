#pragma once

#include <string>
#include <vector>
#include "model.hpp"
#include "tensor.hpp"
#include "timing.hpp"

#include "workspace.hpp"

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input, EngineWorkspace& workspace);

Tensor forward_resnet18_workspace_timed(
    const Model& model,
    const Tensor& input,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
);

Tensor forward_resnet18_scalar_timed(
    const Model& model,
    const Tensor& input,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
);

// Dump control for standalone layer benchmarks.
// Enable before a single-threaded diagnostic forward pass; disable immediately after.
// Not thread-safe.
void set_tensor_dump_layers(const std::vector<std::string>& layer_names);
void clear_tensor_dump_layers();