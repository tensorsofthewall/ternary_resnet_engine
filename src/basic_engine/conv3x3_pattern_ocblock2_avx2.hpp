#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"

bool can_use_pattern_ocblock2_3x3_avx2(const Conv2DParams& conv);

Tensor conv3x3s1_pattern_ocblock2_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);

Tensor conv3x3s1_pattern_ocblock2_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);