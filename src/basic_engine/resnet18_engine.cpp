#include "resnet18_engine.hpp"

#include "ops.hpp"

namespace {

Tensor conv_bn_relu(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
    Tensor y = conv2d_ternary_scalar(x, m.convs.at(conv_name));
    y = batchnorm2d(y, m.bns.at(bn_name));
    y = relu(y);
    return y;
}

Tensor conv_bn(const Model& m, const Tensor& x, const char* conv_name, const char* bn_name) {
    Tensor y = conv2d_ternary_scalar(x, m.convs.at(conv_name));
    y = batchnorm2d(y, m.bns.at(bn_name));
    return y;
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