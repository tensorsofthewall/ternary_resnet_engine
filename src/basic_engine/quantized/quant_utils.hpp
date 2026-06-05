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

QuantTensorI8& quantize_pad_tensor_symmetric_i8_into(
    const Tensor& x,
    int pad_h,
    int pad_w,
    QuantTensorI8& out
);

QuantTensorI8 pad_quant_tensor_i8_asymmetric(
    const QuantTensorI8& x,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right
);

QuantTensorI8& quantize_pad_tensor_symmetric_i8_asymmetric_into(
    const Tensor& x,
    int pad_top,
    int pad_bottom,
    int pad_left,
    int pad_right,
    QuantTensorI8& out
);