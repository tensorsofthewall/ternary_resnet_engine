#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"

bool can_use_pattern_ocblock4_layer4_3x3_avx2(const Conv2DParams& conv);

Tensor conv3x3_pattern_ocblock4_layer4_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);

Tensor conv3x3_pattern_ocblock4_layer4_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock4ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);