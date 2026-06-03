#pragma once

#include "model.hpp"
#include "tensor.hpp"

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input);