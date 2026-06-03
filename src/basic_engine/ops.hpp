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