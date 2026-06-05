#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"
#include "quantized/quant_tensor.hpp"

bool can_use_q8_ocblock4_layer4_s1_3x3_avx2(const Conv2DParams& conv);

Tensor conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);

Tensor conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
    const QuantTensorI8& xq,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);

bool can_use_q8_ocblock4_s2_3x3_avx2(const Conv2DParams& conv);

Tensor conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
    const QuantTensorI8& xq,
    int out_h,
    int out_w,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);

Tensor conv3x3_q8_ocblock4_s2_bn_avx2_out_hw(
    const QuantTensorI8& xq,
    int out_h,
    int out_w,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);