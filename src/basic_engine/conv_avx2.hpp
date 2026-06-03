#pragma once

#include "model.hpp"
#include "tensor.hpp"

bool can_use_avx2_stride1(const Conv2DParams& conv);

Tensor conv2d_ternary_sparse_padded_bn_relu_avx2_stride1(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn);
Tensor conv2d_ternary_sparse_padded_bn_avx2_stride1(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn);