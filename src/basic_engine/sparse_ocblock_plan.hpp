#pragma once

#include "model.hpp"

SparseOCBlock4ConvPlan build_sparse_ocblock4_conv_plan(const Conv2DParams& conv);
void build_sparse_ocblock4_conv_plans(Model& model);