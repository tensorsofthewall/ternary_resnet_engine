#pragma once

#include "model.hpp"

PatternOCBlock2ConvPlan build_pattern_ocblock2_plan(const Conv2DParams& conv);

void build_pattern_ocblock2_plans(Model& model);