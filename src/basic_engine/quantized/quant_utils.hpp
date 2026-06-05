#pragma once

#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

float choose_symmetric_i8_scale(const Tensor& x);

QuantTensorI8 quantize_tensor_symmetric_i8(const Tensor& x);

Tensor dequantize_tensor_i8(const QuantTensorI8& xq);

QuantTensorI8 pad_quant_tensor_i8(
    const QuantTensorI8& x,
    int pad_h,
    int pad_w
);