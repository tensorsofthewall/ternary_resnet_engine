#pragma once

#include "model.hpp"
#include "tensor.hpp"

// Reference dense ternary conv: naive 6-nested-loop scalar accumulation
// straight from Conv2DParams::weight (each tap multiplied by its -1/0/+1
// weight), with implicit zero-padding. The numerical baseline every other
// conv variant is checked against.
Tensor conv2d_ternary_scalar(const Tensor& x, const Conv2DParams& conv);

// Reference scalar batch-norm: y = gamma*(x-mean)/sqrt(var+eps) + beta,
// applied per channel from the raw (unfused) BatchNormParams.
Tensor batchnorm2d(const Tensor& x, const BatchNormParams& bn);

// Elementwise max(x, 0), returned as a new tensor.
Tensor relu(const Tensor& x);

// 3x3/stride=2/pad=1 max-pool (the ResNet18 stem pooling layer).
Tensor maxpool2d_3x3_s2_p1(const Tensor& x);

// Global average pool: collapses each channel's HxW plane to a single value,
// producing a [C, 1, 1] tensor.
Tensor adaptive_avgpool_1x1(const Tensor& x);

// Dense fully-connected layer: y = x @ weight^T (+ bias), from LinearParams.
Tensor linear(const Tensor& x, const LinearParams& fc);

// Elementwise a + b, returned as a new tensor (shapes must match).
Tensor add_tensors(const Tensor& a, const Tensor& b);


// Sparse ternary conv using a ConvPlan: per output channel, sums activations
// at the "+1" positions and subtracts those at the "-1" positions (no
// multiplies). Implicit zero-padding, same accumulation order/output shape
// as conv2d_ternary_scalar -- numerically equivalent, just avoids
// multiplying by zero/one.
Tensor conv2d_ternary_scalar_sparse(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan);

// Sparse ternary conv over an already-unpadded input (out_h/out_w computed
// with no implicit padding -- caller must pre-pad via pad_tensor if the
// layer has pad_h/pad_w > 0). Faster than conv2d_ternary_scalar_sparse
// because the inner loops need no in-bounds checks.
Tensor conv2d_ternary_sparse_nopad_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan);

// Sparse ternary conv that materializes a zero-padded copy of `x` (via
// pad_tensor) up front and then runs the bounds-check-free
// conv2d_ternary_sparse_nopad_scalar path -- or calls it directly when the
// layer has no padding.
Tensor conv2d_ternary_sparse_padded_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan);

// Returns a copy of `x` with `pad_h`/`pad_w` zero-padding added on every side
// of the spatial dims.
Tensor pad_tensor(const Tensor& x, int pad_h, int pad_w);

// Fused padded-sparse-conv + BN(+ReLU) ops: run conv2d_ternary_sparse_padded_scalar's
// accumulation, then apply the already-fused BNFusedParams affine transform
// (and clamp to >=0 in the _relu variant) directly into `y_out`, in one pass
// -- avoiding a separate materialized conv-output tensor.
void conv2d_ternary_sparse_padded_bn_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, Tensor& y_out);
void conv2d_ternary_sparse_padded_bn_relu_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, Tensor& y_out);

// Same as pad_tensor, but writes into a caller-provided, pre-sized `y`
// instead of allocating a new Tensor each call.
void pad_tensor_into(const Tensor& x, int pad_h, int pad_w, Tensor& y);

// In-place variants of relu/add_tensors: mutate `x`/`acc` directly rather
// than allocating a new output tensor.
void relu_inplace(Tensor& x);
void add_inplace(Tensor& acc, const Tensor& b);

// Write-into-existing-tensor variants of maxpool2d_3x3_s2_p1 /
// adaptive_avgpool_1x1 / linear: identical numerics, but reuse the
// caller-provided `y_out` to avoid per-call allocation in the hot path.
void maxpool2d_3x3_s2_p1_into(const Tensor& x, Tensor& y_out);
void adaptive_avgpool_1x1_into(const Tensor& x, Tensor& y_out);
void linear_into(const Tensor& x, const LinearParams& fc, Tensor& y_out);

// Numerically-stable softmax over the whole tensor (subtracts the global max
// before exponentiating, then normalizes by the sum), written into `y_out`.
void softmax_into(const Tensor& x, Tensor& y_out);