#pragma once

#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

struct EngineWorkspace {
    Tensor pad_buffer;

    QuantTensorI8 q8_padded_input;
};