#pragma once

#include "blocked_tensor.hpp"
#include "model.hpp"
#include "bn_fusion.hpp"
#include "tensor.hpp"

// Special-case conv1: IC=3 (not divisible by 16), OC=64, KH=KW=7, stride=2, pad=3.
// Accepts raw NCHW FP32 input (e.g. 3×224×224) and writes NCHWc16 FP32 output.
// Padding, stride, and kernel size are read from conv.
// Always applies BN + ReLU.
void blocked_conv1_7x7_s2(
    const Tensor& input,
    const Conv2DParams& conv,
    const BNFusedParams& bn,
    BlockedTensorFP32& output);
