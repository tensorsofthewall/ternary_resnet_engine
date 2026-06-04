#include "resnet18_engine.hpp"

#include "ops.hpp"
#include "conv_avx2.hpp"
#include "timing.hpp"

namespace {

// // Naive scalar conv + bn + relu block
// Tensor conv_bn_relu(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     Tensor y = conv2d_ternary_scalar(x, m.convs.at(conv_name));
//     y = batchnorm2d(y, m.bns.at(bn_name));
//     y = relu(y);
//     return y;
// }

// Tensor conv_bn(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     Tensor y = conv2d_ternary_scalar(x, m.convs.at(conv_name));
//     y = batchnorm2d(y, m.bns.at(bn_name));
//     return y;
// }

// // Sparse conv versions
// Tensor conv_bn_relu(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     Tensor y = conv2d_ternary_scalar_sparse(x, m.convs.at(conv_name), m.conv_plans.at(conv_name));
//     y = batchnorm2d(y, m.bns.at(bn_name));
//     y = relu(y);
//     return y;
// }

// Tensor conv_bn(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     Tensor y = conv2d_ternary_scalar_sparse(x, m.convs.at(conv_name), m.conv_plans.at(conv_name));
//     y = batchnorm2d(y, m.bns.at(bn_name));
//     return y;
// }

// // Sparse Padded versions
// Tensor conv_bn_relu(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     Tensor y = conv2d_ternary_sparse_padded_scalar(x, m.convs.at(conv_name), m.conv_plans.at(conv_name));
//     y = batchnorm2d(y, m.bns.at(bn_name));
//     y = relu(y);
//     return y;
// }

// Tensor conv_bn(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     Tensor y = conv2d_ternary_sparse_padded_scalar(x, m.convs.at(conv_name), m.conv_plans.at(conv_name));
//     y = batchnorm2d(y, m.bns.at(bn_name));
//     return y;
// }

// // Fused Sparse Padded versions
// Tensor conv_bn_relu(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     const BNFusedParams bn_fused = m.fused_bns.at(bn_name);
//     return conv2d_ternary_sparse_padded_bn_relu_scalar(x, m.convs.at(conv_name), m.conv_plans.at(conv_name), bn_fused);
// }

// Tensor conv_bn(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
//     const BNFusedParams bn_fused = m.fused_bns.at(bn_name);
//     return conv2d_ternary_sparse_padded_bn_scalar(x, m.convs.at(conv_name), m.conv_plans.at(conv_name), bn_fused);
// }

// AVX2-Optimized Fused Sparse Padded versions
Tensor conv_bn_relu(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name) {
    const Conv2DParams& conv = m.convs.at(conv_name);
    const ConvPlan& plan = m.conv_plans.at(conv_name);
    const BNFusedParams bn_fused = m.fused_bns.at(bn_name);
    if (can_use_avx2_width_kernel(conv)) {
        return conv2d_ternary_sparse_padded_bn_relu_avx2_width(x, conv, plan, bn_fused);
    }
    return conv2d_ternary_sparse_padded_bn_relu_scalar(x, conv, plan, bn_fused);
}

Tensor conv_bn(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name) {
    const Conv2DParams& conv = m.convs.at(conv_name);
    const ConvPlan& plan = m.conv_plans.at(conv_name);
    const BNFusedParams bn_fused = m.fused_bns.at(bn_name);
    if (can_use_avx2_width_kernel(conv)) {
        return conv2d_ternary_sparse_padded_bn_avx2_width(x, conv, plan, bn_fused);
    }
    return conv2d_ternary_sparse_padded_bn_scalar(x, conv, plan, bn_fused);
}

Tensor basic_block(
    const Model& m,
    const Tensor& x,
    const char* prefix,
    bool has_downsample
) {
    const std::string p(prefix);

    Tensor identity = x;

    Tensor y = conv_bn_relu(
        m,
        x,
        (p + ".conv1").c_str(),
        (p + ".bn1").c_str()
    );

    y = conv_bn(
        m,
        y,
        (p + ".conv2").c_str(),
        (p + ".bn2").c_str()
    );

    if (has_downsample) {
        identity = conv_bn(
            m,
            x,
            (p + ".downsample.0").c_str(),
            (p + ".downsample.1").c_str()
        );
    }

    y = add_tensors(y, identity);
    y = relu(y);

    return y;
}

// Timed versions of blocks for profiling
Tensor conv_bn_relu_timed(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name, std::vector<TimingEntry>& timings) {
    return timed_op(timings, conv_name + "+" + bn_name + "+relu", [&]() {
        return conv_bn_relu(m, x, conv_name, bn_name);
    });
}
Tensor conv_bn_timed(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name, std::vector<TimingEntry>& timings) {
    return timed_op(timings, conv_name + "+" + bn_name, [&]() {
        return conv_bn(m, x, conv_name, bn_name);
    });
}
Tensor add_relu_timed(const Tensor& a, const Tensor& b, const std::string& name, std::vector<TimingEntry>& timings) {
    return timed_op(timings, name + "+add+relu", [&]() {
        Tensor y = add_tensors(a, b);
        y = relu(y);
        return y;
    });
}
Tensor basic_block_timed(const Model& m, const Tensor& x, const std::string& prefix, bool has_downsample, std::vector<TimingEntry>& timings) {
    Tensor identity = x;

    Tensor y = conv_bn_relu_timed(m,x,(prefix + ".conv1").c_str(),(prefix + ".bn1").c_str(),timings);

    y = conv_bn_timed(m,y,(prefix + ".conv2").c_str(),(prefix + ".bn2").c_str(),timings);

    if (has_downsample) {
        identity = conv_bn_timed(m,x,(prefix + ".downsample.0").c_str(),(prefix + ".downsample.1").c_str(),timings);
    }

    y = add_relu_timed(y, identity, prefix, timings);

    return y;
}
} // namespace

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input) {
    Tensor x = conv_bn_relu(model, input, "conv1", "bn1");
    x = maxpool2d_3x3_s2_p1(x);

    x = basic_block(model, x, "layer1.0", false);
    x = basic_block(model, x, "layer1.1", false);

    x = basic_block(model, x, "layer2.0", true);
    x = basic_block(model, x, "layer2.1", false);

    x = basic_block(model, x, "layer3.0", true);
    x = basic_block(model, x, "layer3.1", false);

    x = basic_block(model, x, "layer4.0", true);
    x = basic_block(model, x, "layer4.1", false);

    x = adaptive_avgpool_1x1(x);
    x = linear(x, model.linears.at("fc"));

    return x;
}

Tensor forward_resnet18_scalar_timed(
    const Model& model,
    const Tensor& input,
    std::vector<TimingEntry>& timings
) {
    Tensor x = conv_bn_relu_timed(
        model,
        input,
        "conv1",
        "bn1",
        timings
    );

    x = timed_op(timings, "maxpool", [&]() {
        return maxpool2d_3x3_s2_p1(x);
    });

    x = basic_block_timed(model, x, "layer1.0", false, timings);
    x = basic_block_timed(model, x, "layer1.1", false, timings);

    x = basic_block_timed(model, x, "layer2.0", true, timings);
    x = basic_block_timed(model, x, "layer2.1", false, timings);

    x = basic_block_timed(model, x, "layer3.0", true, timings);
    x = basic_block_timed(model, x, "layer3.1", false, timings);

    x = basic_block_timed(model, x, "layer4.0", true, timings);
    x = basic_block_timed(model, x, "layer4.1", false, timings);

    x = timed_op(timings, "avgpool", [&]() {
        return adaptive_avgpool_1x1(x);
    });

    x = timed_op(timings, "fc", [&]() {
        return linear(x, model.linears.at("fc"));
    });

    return x;
}