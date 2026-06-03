#pragma once

#include "model.hpp"
#include "tensor.hpp"

// Fused BatchNorm Helper
BNFusedParams make_bn_fused_params(const BatchNormParams& bn);

void build_fused_bns(Model& model);