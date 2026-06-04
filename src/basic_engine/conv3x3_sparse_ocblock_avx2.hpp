#pragma once

#include "model.hpp"
#include "tensor.hpp"
#include "workspace.hpp"

bool can_use_sparse_ocblock4_3x3s1_avx2(const Conv2DParams& conv);

Tensor conv3x3s1_sparse_ocblock4_bn_avx2(const Tensor& x, const Conv2DParams& conv, const SparseOCBlock4ConvPlan& plan, const BNFusedParams& bn, EngineWorkspace& workspace);
Tensor conv3x3s1_sparse_ocblock4_bn_relu_avx2(const Tensor& x, const Conv2DParams& conv, const SparseOCBlock4ConvPlan& plan, const BNFusedParams& bn, EngineWorkspace& workspace);