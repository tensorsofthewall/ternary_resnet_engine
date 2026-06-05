#pragma once

#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

struct EngineWorkspace {
    Tensor pad_buffer;

    QuantTensorI8 q8_padded_input;
    QuantTensorI8 q8_padded_input_s2;

    // Pre-allocated activation buffers for forward_resnet18_scalar.
    // Reused across inference calls via Tensor::reuse() — no malloc/zero-init after first run.
    Tensor buf_x;         // current input "x" in the forward pass
    Tensor buf_a;         // conv1 output within a block
    Tensor buf_b;         // conv2 output / new x candidate
    Tensor buf_identity;  // identity branch for downsampling blocks
};