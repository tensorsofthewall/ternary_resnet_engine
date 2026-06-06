#pragma once
#include "model.hpp"
#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

bool can_use_q8_1x1_s2_avx2(const Conv2DParams& conv);

void conv1x1_q8_s2_bn_avx2(const QuantTensorI8& xq, const Conv2DParams& conv,
                             const PatternOCBlock2ConvPlan& plan,
                             const BNFusedParams& bn, int out_h, int out_w,
                             Tensor& y_out);
