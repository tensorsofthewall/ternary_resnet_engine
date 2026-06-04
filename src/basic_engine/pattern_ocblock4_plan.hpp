#pragma once

#include "model.hpp"

PatternOCBlock4ConvPlan build_pattern_ocblock4_plan(const Conv2DParams& conv);

void build_pattern_ocblock4_plans(Model& model);