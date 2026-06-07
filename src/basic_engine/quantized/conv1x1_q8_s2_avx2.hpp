#pragma once
#include "model.hpp"
#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

// Matches the 1x1/stride-2/no-pad downsample-projection convs (the residual
// shortcut convs at the start of layer2/3/4) with an even output-channel count.
bool can_use_q8_1x1_s2_avx2(const Conv2DParams& conv);

// AVX2 q8 1x1/stride-2 conv + fused BN, dispatching on each spatial
// position's 2-output-channel ternary sign pattern via a
// PatternOCBlock2ConvPlan (see model.hpp): gathers 8 stride-2 activations at
// once via a fixed byte-shuffle (STRIDE2_SHUF), accumulates per-pattern
// add/sub in int32, then applies the per-channel BN scale/bias and a masked
// vector store for output-row tails narrower than 8 pixels (TAIL_MASKS). No
// ReLU: it's dispatched (from conv_bn) only for the downsample/shortcut
// branch, whose output feeds the residual add -- ReLU is applied after that
// add, not here. OpenMP-parallel over plan.oc_blocks.
void conv1x1_q8_s2_bn_avx2(const QuantTensorI8& xq, const Conv2DParams& conv,
                             const PatternOCBlock2ConvPlan& plan,
                             const BNFusedParams& bn, int out_h, int out_w,
                             Tensor& y_out);
