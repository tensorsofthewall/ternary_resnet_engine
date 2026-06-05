#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"

Tensor forward_layer4_q8_experimental(
    Model& model,
    const Tensor& x,
    EngineWorkspace& workspace
);