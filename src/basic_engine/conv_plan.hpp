#pragma once

#include "model.hpp"

ConvPlan build_conv_plan(const Conv2DParams& conv);
void build_conv_plans(Model& model);