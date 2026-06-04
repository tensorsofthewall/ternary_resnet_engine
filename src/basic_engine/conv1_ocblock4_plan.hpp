#pragma once

#include "model.hpp"

Conv1OCBlock4ConvPlan build_conv1_ocblock4_plan(const Conv2DParams& conv);

void build_conv1_ocblock4_plans(Model& model);