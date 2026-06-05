#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"

bool can_use_conv1_ocblock4_7x7s2_avx2(const Conv2DParams& conv);

void conv1_ocblock4_7x7s2_bn_relu_avx2(
    const Tensor& x, const Conv2DParams& conv, const Conv1OCBlock4ConvPlan& plan,
    const BNFusedParams& bn, EngineWorkspace& workspace, Tensor& y_out);