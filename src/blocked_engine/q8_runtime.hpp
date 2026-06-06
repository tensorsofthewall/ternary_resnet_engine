#pragma once

#include "blocked_tensor.hpp"

// Symmetric per-tensor quantization: scale = max_abs / 127.
void quantize_blocked_fp32_to_i8(const BlockedTensorFP32& src, BlockedTensorI8& dst);
void dequantize_blocked_i8_to_fp32(const BlockedTensorI8& src, BlockedTensorFP32& dst);
