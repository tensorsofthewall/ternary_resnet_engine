#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"
#include "quantized/quant_tensor.hpp"

// Shape-specialized AVX2 q8 (quantized int8 activation) ternary 3x3 convs,
// dispatching on each spatial position's 4-output-channel ternary sign
// pattern via a PatternOCBlock4ConvPlan (see model.hpp) -- fused with BN and
// an optional ReLU. Each `can_use_*` predicate matches exactly one ResNet18
// layer's (in_channels, out_channels, stride) shape, and each kernel is named
// after the layer it targets; callers probe the predicates to pick the right
// specialized kernel for a given Conv2DParams. The `_bn_*` / `_bn_relu_*`
// pairs share one `_impl` (in the .cpp) that takes a `relu` flag -- the two
// entry points just fix that flag so dispatch sites don't need a branch.

// Matches layer4's 3x3/stride-1 convs (in=out=512).
bool can_use_q8_ocblock4_layer4_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

void conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

// Matches the layer3 downsample conv's shape (3x3/stride-2, in=128, out=256).
bool can_use_q8_ocblock4_s2_3x3_avx2(const Conv2DParams& conv);
// Matches the layer4 downsample conv's shape (3x3/stride-2, in=256, out=512).
bool can_use_q8_ocblock4_layer4_s2_3x3_avx2(const Conv2DParams& conv);

// Generic stride-2 kernel pair shared by both downsample shapes above --
// `out_h`/`out_w` are passed explicitly (rather than derived internally)
// since the caller already knows the block's output extents.
void conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
    const QuantTensorI8& xq, int out_h, int out_w,
    const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out);

void conv3x3_q8_ocblock4_s2_bn_avx2_out_hw(
    const QuantTensorI8& xq, int out_h, int out_w,
    const Conv2DParams& conv, const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out);

// Matches layer3's 3x3/stride-1 convs (in=out=256).
bool can_use_q8_ocblock4_layer3_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock4_layer3_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

void conv3x3_q8_ocblock4_layer3_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

// Matches layer2's 3x3/stride-1 convs (in=out=128).
bool can_use_q8_ocblock4_layer2_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock4_layer2_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

void conv3x3_q8_ocblock4_layer2_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

// Matches layer1's 3x3/stride-1 convs (in=out=64).
bool can_use_q8_ocblock4_layer1_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock4_layer1_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);

void conv3x3_q8_ocblock4_layer1_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan, const BNFusedParams& bn,
    EngineWorkspace& workspace, Tensor& y_out);
