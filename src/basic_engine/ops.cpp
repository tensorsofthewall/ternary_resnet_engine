#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Function to reduce padding allocation overhead
void pad_tensor_into(const Tensor& x, int pad_h, int pad_w, Tensor& y) {
    const int padded_h = x.h + 2 * pad_h;
    const int padded_w = x.w + 2 * pad_w;
    if (y.c != x.c || y.h != padded_h || y.w != padded_w) {
        y = Tensor(x.c, padded_h, padded_w);
    } else {
        std::fill(y.data.begin(), y.data.end(), 0.0f);
    }
    for (int c = 0; c < x.c; ++c) {
        for (int h = 0; h < x.h; ++h) {
            const float* src = x.ptr(c,h,0);
            float* dst = y.ptr(c, h + pad_h, pad_w);
            std::copy(src, src + x.w, dst);
        }
    }
}

namespace {

size_t conv_weight_index(
    const Conv2DParams& conv,
    int oc,
    int ic,
    int ky,
    int kx
) {
    return static_cast<size_t>(((oc * conv.in_channels + ic) * conv.kh + ky) * conv.kw + kx);
}

} // namespace

Tensor conv2d_ternary_scalar(const Tensor& x, const Conv2DParams& conv) {
    const int out_h = (x.h + 2 * conv.pad_h - conv.kh) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - conv.kw) / conv.stride_w + 1;

    Tensor y(conv.out_channels, out_h, out_w);

    for (int oc = 0; oc < conv.out_channels; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;

                for (int ic = 0; ic < conv.in_channels; ++ic) {
                    for (int ky = 0; ky < conv.kh; ++ky) {
                        for (int kx = 0; kx < conv.kw; ++kx) {
                            const int ih = oh * conv.stride_h + ky - conv.pad_h;
                            const int iw = ow * conv.stride_w + kx - conv.pad_w;

                            if (ih < 0 || ih >= x.h || iw < 0 || iw >= x.w) {
                                continue;
                            }

                            const int8_t w = conv.weight[
                                conv_weight_index(conv, oc, ic, ky, kx)
                            ];

                            if (w == 1) {
                                acc += x(ic, ih, iw);
                            } else if (w == -1) {
                                acc -= x(ic, ih, iw);
                            }
                        }
                    }
                }

                if (conv.has_bias) {
                    acc += conv.bias[static_cast<size_t>(oc)];
                }

                y(oc, oh, ow) = acc;
            }
        }
    }

    return y;
}

Tensor batchnorm2d(const Tensor& x, const BatchNormParams& bn) {
    Tensor y(x.c, x.h, x.w);

    for (int c = 0; c < x.c; ++c) {
        const float scale = bn.gamma[static_cast<size_t>(c)] /
            std::sqrt(bn.running_var[static_cast<size_t>(c)] + bn.eps);

        const float bias = bn.beta[static_cast<size_t>(c)] -
            bn.running_mean[static_cast<size_t>(c)] * scale;

        for (int h = 0; h < x.h; ++h) {
            for (int w = 0; w < x.w; ++w) {
                y(c, h, w) = x(c, h, w) * scale + bias;
            }
        }
    }

    return y;
}

Tensor relu(const Tensor& x) {
    Tensor y(x.c, x.h, x.w);

    for (size_t i = 0; i < x.data.size(); ++i) {
        y.data[i] = std::max(0.0f, x.data[i]);
    }

    return y;
}

void relu_inplace(Tensor& x) {
    for (float& v : x.data) v = std::max(0.0f, v);
}

void add_inplace(Tensor& acc, const Tensor& b) {
    for (size_t i = 0; i < acc.data.size(); ++i) acc.data[i] += b.data[i];
}

void maxpool2d_3x3_s2_p1_into(const Tensor& x, Tensor& y_out) {
    const int kh = 3, kw = 3, stride = 2, pad = 1;
    const int out_h = (x.h + 2 * pad - kh) / stride + 1;
    const int out_w = (x.w + 2 * pad - kw) / stride + 1;
    y_out.reuse(x.c, out_h, out_w);

    for (int c = 0; c < x.c; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        const int ih = oh * stride + ky - pad;
                        const int iw = ow * stride + kx - pad;
                        if (ih < 0 || ih >= x.h || iw < 0 || iw >= x.w) continue;
                        max_val = std::max(max_val, x(c, ih, iw));
                    }
                }
                y_out(c, oh, ow) = max_val;
            }
        }
    }
}

Tensor add_tensors(const Tensor& a, const Tensor& b) {
    if (a.c != b.c || a.h != b.h || a.w != b.w) {
        throw std::runtime_error("Tensor shape mismatch in add_tensors");
    }

    Tensor y(a.c, a.h, a.w);

    for (size_t i = 0; i < a.data.size(); ++i) {
        y.data[i] = a.data[i] + b.data[i];
    }

    return y;
}

Tensor maxpool2d_3x3_s2_p1(const Tensor& x) {
    const int kh = 3;
    const int kw = 3;
    const int stride = 2;
    const int pad = 1;

    const int out_h = (x.h + 2 * pad - kh) / stride + 1;
    const int out_w = (x.w + 2 * pad - kw) / stride + 1;

    Tensor y(x.c, out_h, out_w);

    for (int c = 0; c < x.c; ++c) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float max_val = -std::numeric_limits<float>::infinity();

                for (int ky = 0; ky < kh; ++ky) {
                    for (int kx = 0; kx < kw; ++kx) {
                        const int ih = oh * stride + ky - pad;
                        const int iw = ow * stride + kx - pad;

                        if (ih < 0 || ih >= x.h || iw < 0 || iw >= x.w) {
                            continue;
                        }

                        max_val = std::max(max_val, x(c, ih, iw));
                    }
                }

                y(c, oh, ow) = max_val;
            }
        }
    }

    return y;
}

Tensor adaptive_avgpool_1x1(const Tensor& x) {
    Tensor y(x.c, 1, 1);

    const float denom = static_cast<float>(x.h * x.w);

    for (int c = 0; c < x.c; ++c) {
        float sum = 0.0f;

        for (int h = 0; h < x.h; ++h) {
            for (int w = 0; w < x.w; ++w) {
                sum += x(c, h, w);
            }
        }

        y(c, 0, 0) = sum / denom;
    }

    return y;
}

Tensor linear(const Tensor& x, const LinearParams& fc) {
    if (x.c * x.h * x.w != fc.in_features) {
        throw std::runtime_error("Linear input size mismatch");
    }

    Tensor y(fc.out_features, 1, 1);

    for (int o = 0; o < fc.out_features; ++o) {
        float acc = 0.0f;

        for (int i = 0; i < fc.in_features; ++i) {
            acc += fc.weight[static_cast<size_t>(o * fc.in_features + i)] *
                   x.data[static_cast<size_t>(i)];
        }

        if (fc.has_bias) {
            acc += fc.bias[static_cast<size_t>(o)];
        }

        y(o, 0, 0) = acc;
    }

    return y;
}

void adaptive_avgpool_1x1_into(const Tensor& x, Tensor& y_out) {
    y_out.reuse(x.c, 1, 1);

    const float denom = static_cast<float>(x.h * x.w);

    for (int c = 0; c < x.c; ++c) {
        float sum = 0.0f;
        for (int h = 0; h < x.h; ++h) {
            for (int w = 0; w < x.w; ++w) {
                sum += x(c, h, w);
            }
        }
        y_out(c, 0, 0) = sum / denom;
    }
}

void linear_into(const Tensor& x, const LinearParams& fc, Tensor& y_out) {
    if (x.c * x.h * x.w != fc.in_features) {
        throw std::runtime_error("Linear input size mismatch");
    }

    y_out.reuse(fc.out_features, 1, 1);

    for (int o = 0; o < fc.out_features; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < fc.in_features; ++i) {
            acc += fc.weight[static_cast<size_t>(o * fc.in_features + i)] *
                   x.data[static_cast<size_t>(i)];
        }
        if (fc.has_bias) {
            acc += fc.bias[static_cast<size_t>(o)];
        }
        y_out(o, 0, 0) = acc;
    }
}

void softmax_into(const Tensor& x, Tensor& y_out) {
    const int n = x.c * x.h * x.w;
    y_out.reuse(x.c, x.h, x.w);

    const float* src = x.data.data();
    float* dst = y_out.data.data();

    float max_val = *std::max_element(src, src + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        dst[i] = std::exp(src[i] - max_val);
        sum += dst[i];
    }
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        dst[i] *= inv_sum;
    }
}

// Padding sparse tensors
Tensor pad_tensor(const Tensor& x, int pad_h, int pad_w) {
    Tensor y(x.c, x.h + 2 * pad_h, x.w + 2 * pad_w);

    for (int c = 0; c < x.c; ++c) {
        for (int h = 0; h < x.h; ++h) {
            const float* src = x.ptr(c,h,0);
            float* dst = y.ptr(c,h+pad_h, pad_w);
            std::copy(src, src + x.w, dst);
        }
    }

    return y;
}

// Sparse scalar conv op
Tensor conv2d_ternary_scalar_sparse(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan) {
    const int out_h = (x.h + 2 * conv.pad_h - conv.kh) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - conv.kw) / conv.stride_w + 1;

    Tensor y(conv.out_channels, out_h, out_w);

    for (int oc = 0; oc < conv.out_channels; ++oc) {
        const ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;

                for (const WeightIndex& wi : oc_plan.pos) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky) - conv.pad_h;
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx) - conv.pad_w;

                    if (ih >= 0 && ih < x.h && iw >= 0 && iw < x.w) {
                        acc += x(static_cast<int>(wi.ic), ih, iw);
                    }
                }

                for (const WeightIndex& wi : oc_plan.neg) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky) - conv.pad_h;
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx) - conv.pad_w;

                    if (ih >= 0 && ih < x.h && iw >= 0 && iw < x.w) {
                        acc -= x(static_cast<int>(wi.ic), ih, iw);
                    }
                }

                if (conv.has_bias) {
                    acc += conv.bias[static_cast<size_t>(oc)];
                }

                y(oc, oh, ow) = acc;
            }
        }
    }

    return y;   
}

// No-padding sparse conv op
Tensor conv2d_ternary_sparse_nopad_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan) {
    const int out_h = (x.h-conv.kh)/conv.stride_h+1;
    const int out_w = (x.w-conv.kw)/conv.stride_w+1;

    Tensor y(conv.out_channels, out_h, out_w);

    for (int oc=0; oc < conv.out_channels; ++oc) {
        const ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

        for (int oh=0; oh < out_h; ++oh) {
            for (int ow=0; ow < out_w; ++ow) {
                float acc = 0.0f;

                for (const WeightIndex& wi : oc_plan.pos) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);

                    acc += x(static_cast<int>(wi.ic), ih, iw);
                }

                for (const WeightIndex& wi : oc_plan.neg) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);

                    acc -= x(static_cast<int>(wi.ic), ih, iw);
                }

                if (conv.has_bias) {
                    acc += conv.bias[static_cast<size_t>(oc)];
                }

                y(oc, oh, ow) = acc;
            }
        }
    }

    return y;
}

// Padded sparse conv op
Tensor conv2d_ternary_sparse_padded_scalar(
    const Tensor& x,
    const Conv2DParams& conv,
    const ConvPlan& plan
) {
    if (conv.pad_h == 0 && conv.pad_w == 0) {
        return conv2d_ternary_sparse_nopad_scalar(x, conv, plan);
    }

    const Tensor xpad = pad_tensor(x, conv.pad_h, conv.pad_w);

    const int out_h = (x.h + 2 * conv.pad_h - conv.kh) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - conv.kw) / conv.stride_w + 1;

    Tensor y(conv.out_channels, out_h, out_w);

    for (int oc = 0; oc < conv.out_channels; ++oc) {
        const ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];

        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;

                for (const WeightIndex& wi : oc_plan.pos) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);

                    acc += xpad(static_cast<int>(wi.ic), ih, iw);
                }

                for (const WeightIndex& wi : oc_plan.neg) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);

                    acc -= xpad(static_cast<int>(wi.ic), ih, iw);
                }

                if (conv.has_bias) {
                    acc += conv.bias[static_cast<size_t>(oc)];
                }

                y(oc, oh, ow) = acc;
            }
        }
    }

    return y;
}

// Fused Padded Sparse Conv2D+BatchNorm implementation
// Helper for fused op with ReLU flag
void conv2d_ternary_sparse_padded_bn_impl(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, bool apply_relu, Tensor& y_out) {
    Tensor xpad;
    const Tensor* input_ptr = &x;

    if (conv.pad_h > 0 || conv.pad_w > 0) {
        xpad = pad_tensor(x, conv.pad_h, conv.pad_w);
        input_ptr = &xpad;
    }

    const Tensor& xin  = *input_ptr;
    const int out_h = (x.h + 2 * conv.pad_h - conv.kh) / conv.stride_h + 1;
    const int out_w = (x.w + 2 * conv.pad_w - conv.kw) / conv.stride_w + 1;

    y_out.reuse(conv.out_channels, out_h, out_w);

    for (int oc=0; oc < conv.out_channels; ++oc) {
        const ConvOCPlan& oc_plan = plan.oc_plans[static_cast<size_t>(oc)];
        const float bn_scale = bn.scale[static_cast<size_t>(oc)];
        const float bn_bias = bn.bias[static_cast<size_t>(oc)];
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                float acc = 0.0f;
                for (const WeightIndex& wi : oc_plan.pos) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);

                    acc += xin(static_cast<int>(wi.ic), ih, iw);
                }

                for (const WeightIndex& wi : oc_plan.neg) {
                    const int ih = oh * conv.stride_h + static_cast<int>(wi.ky);
                    const int iw = ow * conv.stride_w + static_cast<int>(wi.kx);

                    acc -= xin(static_cast<int>(wi.ic), ih, iw);
                }

                if (conv.has_bias) {
                    acc += conv.bias[static_cast<size_t>(oc)];
                }

                acc = acc*bn_scale + bn_bias;

                if (apply_relu && acc < 0.0f) {
                    acc = 0.0f;
                }
                y_out(oc, oh, ow) = acc;
            }
        }
    }
}

void conv2d_ternary_sparse_padded_bn_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, Tensor& y_out) {
    conv2d_ternary_sparse_padded_bn_impl(x, conv, plan, bn, false, y_out);
}

void conv2d_ternary_sparse_padded_bn_relu_scalar(const Tensor& x, const Conv2DParams& conv, const ConvPlan& plan, const BNFusedParams& bn, Tensor& y_out) {
    conv2d_ternary_sparse_padded_bn_impl(x, conv, plan, bn, true, y_out);
}
