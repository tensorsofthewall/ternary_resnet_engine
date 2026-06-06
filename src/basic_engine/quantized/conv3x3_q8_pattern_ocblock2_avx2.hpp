#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"
#include "quantized/quant_tensor.hpp"

// Layer2.0.conv1 (64→128ch, stride-2, 56×56→28×28): OC-Block2 + OH_BLOCK=4 + INT8 inputs.
// extra_right_pad=14 needed for stride-2 8-lane load at last tile (ow=24→iw=48+kx, 16-byte read).
bool can_use_q8_ocblock2_spatial4_layer2_s2_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock2_spatial4_layer2_s2_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

// Layer1 (64ch, 56×56, stride-1): OC-Block2 + OH_BLOCK=4 + INT8 inputs.
// xq must be pre-quantized and padded (pad=1, no extra right pad needed).
bool can_use_q8_ocblock2_spatial4_layer1_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock2_spatial4_layer1_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

void conv3x3_q8_ocblock2_spatial4_layer1_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

// Layer2 (128ch, 28×28, stride-1): OC-Block2 + OH_BLOCK=4 + INT8 inputs.
// xq must be pre-quantized and padded (pad=1 + extra_right=4 for safe 8-byte loads at tail).
bool can_use_q8_ocblock2_spatial4_layer2_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

void conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

// Layer3 (256ch, 14×14, stride-1): OC-Block2 + OH_BLOCK=4 + INT8 inputs.
// xq must be pre-quantized and padded (pad=1 + extra_right=4 for safe loads at tail).
bool can_use_q8_ocblock2_spatial4_layer3_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

void conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

// Layer4 (512ch, 7×7, stride-1): OC-Block2 + OH_BLOCK=4 + INT8 inputs.
// 512×9×9 = 40KB INT8 padded input fits in L1 cache.
// xq must be pre-quantized and padded (pad=1 + extra_right=4 for safe 8-byte loads).
bool can_use_q8_ocblock2_spatial4_layer4_s1_3x3_avx2(const Conv2DParams& conv);

void conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);

void conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_relu_avx2(
    const QuantTensorI8& xq, const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan, const BNFusedParams& bn,
    Tensor& y_out);
