#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"

// Channel block pattern convolution with 3x3 kernel, optimized with AVX2
bool can_use_pattern_ocblock2_3x3_avx2(const Conv2DParams& conv);

// Channel + Spatial block pattern convolution with 3x3 kernel, optimized with AVX2
bool can_use_pattern_ocblock2_spatial2_3x3_avx2(const Conv2DParams& conv);

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

// Spatial2 variant of the above two functions, which use a different data access pattern that is more efficient for certain layers
Tensor conv3x3_pattern_ocblock2_spatial2_bn_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);

Tensor conv3x3_pattern_ocblock2_spatial2_bn_relu_avx2(
    const Tensor& x,
    const Conv2DParams& conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams& bn,
    EngineWorkspace& workspace
);