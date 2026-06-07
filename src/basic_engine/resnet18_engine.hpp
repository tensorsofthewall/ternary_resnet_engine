#pragma once

#include <string>
#include <vector>
#include "model.hpp"
#include "tensor.hpp"
#include "timing.hpp"

#include "workspace.hpp"

// Full ResNet18 forward pass (conv1+bn+relu -> maxpool -> 8 basic blocks ->
// avgpool -> fc), reusing `workspace`'s preallocated buffers throughout so no
// per-layer Tensor allocation occurs in the hot path. Returns logits
// (workspace.buf_fc -- valid until the next call reuses it).
Tensor forward_resnet18_scalar(const Model& model, const Tensor& input, EngineWorkspace& workspace);

// Returns softmax probabilities over the 1000-class output.
Tensor forward_resnet18_softmax(const Model& model, const Tensor& input, EngineWorkspace& workspace);

// Same forward pass as forward_resnet18_scalar, but records a per-stage
// TimingEntry (conv1+bn1+relu, maxpool, each basic block, avgpool, fc) into
// `timings` -- the workspace-backed path used for reported benchmark numbers.
Tensor forward_resnet18_workspace_timed(
    const Model& model,
    const Tensor& input,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
);

// Legacy debug-only timing path: re-implements the forward pass with
// return-by-value Tensor temporaries (via basic_block_timed/timed_op) instead
// of the workspace buffers, so per-stage timings include allocation overhead.
// Not used for reported benchmark numbers -- use
// forward_resnet18_workspace_timed for profiling the optimized engine.
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