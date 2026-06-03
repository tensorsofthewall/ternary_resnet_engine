#pragma once

#include "model.hpp"
#include "tensor.hpp"

Tensor conv2d_ternary_scalar(const Tensor& x, const Conv2DParams& conv);
Tensor batchnorm2d(const Tensor& x, const BatchNormParams& bn);
Tensor relu(const Tensor& x);
Tensor maxpool2d_3x3_s2_p1(const Tensor& x);
Tensor adaptive_avgpool_1x1(const Tensor& x);
Tensor linear(const Tensor& x, const LinearParams& fc);
Tensor add_tensors(const Tensor& a, const Tensor& b);


//Basic sparse scalar conv op
Tensor conv2d_ternary_scalar_sparse(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan);

// No-padding sparse conv op
Tensor conv2d_ternary_sparse_nopad_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan);

// Padded sparse conv op
Tensor conv2d_ternary_sparse_padded_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan);

// Padding sparse tensors 
Tensor pad_tensor(const Tensor& x, int pad_h, int pad_w);

// Fused Padded Conv + BatchNorm and optional ReLU op
Tensor conv2d_ternary_sparse_padded_bn_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn);
Tensor conv2d_ternary_sparse_padded_bn_relu_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn);