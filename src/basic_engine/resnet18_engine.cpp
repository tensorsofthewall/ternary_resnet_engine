#include "resnet18_engine.hpp"

#include "ops.hpp"
#include "conv_avx2.hpp"
#include "timing.hpp"
#include "conv3x3_sparse_ocblock_avx2.hpp"
#include "conv3x3_pattern_ocblock2_avx2.hpp"
#include "conv3x3_pattern_ocblock4_avx2.hpp"
#include "conv1_ocblock4_avx2.hpp"
#include "quantized/quant_utils.hpp"
#include "quantized/conv3x3_q8_pattern_ocblock4_avx2.hpp"
#include "quantized/conv3x3_q8_pattern_ocblock2_avx2.hpp"
#include "quantized/conv1x1_q8_s2_avx2.hpp"

#include <stdexcept>
#include <unordered_set>

// Helper for intermediate capture for layer4.0.conv2
#include <fstream>
void write_tensor_f32(const std::string& path, const Tensor& x) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open tensor dump file: " + path);
    }

    f.write(
        reinterpret_cast<const char*>(x.data.data()),
        static_cast<std::streamsize>(x.data.size() * sizeof(float))
    );

    if (!f) {
        throw std::runtime_error("Failed to write tensor dump file: " + path);
    }
}
// Runtime dump control — set via set_tensor_dump_layers / clear_tensor_dump_layers.
// Not thread-safe; only call from the benchmark binary with a single forward pass.
static bool g_tensor_dump_enabled = false;
static std::unordered_set<std::string> g_tensor_dump_layers;

bool should_dump_intermediate_tensor(const std::string& name) {
    return g_tensor_dump_enabled && g_tensor_dump_layers.count(name) > 0;
}
std::string sanitize_tensor_name(std::string name) {
    for (char& c : name) {
        if (c == '.') {
            c = '_';
        }
    }

    return name;
}

struct BasicBlockSpec {
    const char* name;
    const char* conv1;
    const char* bn1;
    const char* conv2;
    const char* bn2;
    const char* downsample_conv;
    const char* downsample_bn;
    bool has_downsample;
};

static constexpr BasicBlockSpec kLayer1_0{ "layer1.0", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2", nullptr, nullptr, false };
static constexpr BasicBlockSpec kLayer1_1{ "layer1.1", "layer1.1.conv1", "layer1.1.bn1", "layer1.1.conv2", "layer1.1.bn2", nullptr, nullptr, false };
static constexpr BasicBlockSpec kLayer2_0{ "layer2.0", "layer2.0.conv1", "layer2.0.bn1", "layer2.0.conv2", "layer2.0.bn2", "layer2.0.downsample.0", "layer2.0.downsample.1", true };
static constexpr BasicBlockSpec kLayer2_1{ "layer2.1", "layer2.1.conv1", "layer2.1.bn1", "layer2.1.conv2", "layer2.1.bn2", nullptr, nullptr, false };
static constexpr BasicBlockSpec kLayer3_0{ "layer3.0", "layer3.0.conv1", "layer3.0.bn1", "layer3.0.conv2", "layer3.0.bn2", "layer3.0.downsample.0", "layer3.0.downsample.1", true };
static constexpr BasicBlockSpec kLayer3_1{ "layer3.1", "layer3.1.conv1", "layer3.1.bn1", "layer3.1.conv2", "layer3.1.bn2", nullptr, nullptr, false };
static constexpr BasicBlockSpec kLayer4_0{ "layer4.0", "layer4.0.conv1", "layer4.0.bn1", "layer4.0.conv2", "layer4.0.bn2", "layer4.0.downsample.0", "layer4.0.downsample.1", true };
static constexpr BasicBlockSpec kLayer4_1{ "layer4.1", "layer4.1.conv1", "layer4.1.bn1", "layer4.1.conv2", "layer4.1.bn2", nullptr, nullptr, false };

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
void conv_bn_relu(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name, EngineWorkspace& workspace, Tensor& y_out) {
    const Conv2DParams& conv = m.convs.at(conv_name);
    const ConvPlan& plan = m.conv_plans.at(conv_name);
    const BNFusedParams& bn_fused = m.fused_bns.at(bn_name);
    constexpr bool enable_ocblock_kernel = false;
    constexpr bool enable_pattern_ocblock2_kernel = true;
    constexpr bool enable_pattern_ocblock2_spatial4_kernel = true;
    constexpr bool enable_pattern_ocblock2_spatial2_kernel = true;
    constexpr bool enable_pattern_ocblock4_layer4_kernel = true;
    constexpr bool enable_pattern_ocblock4_general_s1_kernel = false;
    constexpr bool enable_conv1_ocblock4_kernel = true;
    constexpr bool enable_q8_layer4_stride1_kernel = true;
    constexpr bool enable_q8_layer4_stride2_kernel = true;
    constexpr bool enable_q8_layer3_s2_kernel = true;
    constexpr bool enable_q8_layer3_s1_kernel = true;
    constexpr bool enable_q8_layer2_s1_kernel = false;
    constexpr bool enable_q8_layer1_s1_kernel = false;
    constexpr bool enable_q8_ocblock2_layer1_s1_kernel = true;
    constexpr bool enable_q8_ocblock2_layer2_s1_kernel = true;
    constexpr bool enable_q8_ocblock2_layer2_s2_kernel = true;
    constexpr bool enable_q8_ocblock2_layer3_s1_kernel = true;
    constexpr bool enable_q8_ocblock2_layer4_s1_kernel = true;
    if (enable_q8_ocblock2_layer4_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer4_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_q8_layer4_stride1_kernel &&
        can_use_q8_ocblock4_layer4_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int l4_extra_right_pad = 1;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + l4_extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_layer4_stride2_kernel &&
        can_use_q8_ocblock4_layer4_s2_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int extra_right_pad = 2;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input_s2);
        conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(xq, 7, 7, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_layer3_s2_kernel && conv_name == "layer3.0.conv1" &&
        can_use_q8_ocblock4_s2_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input_s2);
        conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(xq, 14, 14, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer3_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer3_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_q8_layer3_s1_kernel &&
        can_use_q8_ocblock4_layer3_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int extra_right_pad = 2;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer3_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_layer2_s1_kernel &&
        can_use_q8_ocblock4_layer2_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer2_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_layer1_s1_kernel &&
        can_use_q8_ocblock4_layer1_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer1_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer2_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer2_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer1_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer1_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer1_s1_bn_relu_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_conv1_ocblock4_kernel && can_use_conv1_ocblock4_7x7s2_avx2(conv)) {
        const auto plan_it = m.conv1_ocblock4_plans.find(conv_name);
        if (plan_it == m.conv1_ocblock4_plans.end())
            throw std::runtime_error("Missing conv1 OC-Block4 plan for conv: " + conv_name);
        conv1_ocblock4_7x7s2_bn_relu_avx2(x, conv, plan_it->second, bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock4_layer4_kernel && can_use_pattern_ocblock4_layer4_s2_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock4_layer4_s2_bn_relu_avx2(x, conv, m.pattern_ocblock4_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock4_layer4_kernel && can_use_pattern_ocblock4_layer4_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock4_layer4_bn_relu_avx2(x, conv, m.pattern_ocblock4_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock4_general_s1_kernel && can_use_pattern_ocblock4_s1_3x3_avx2(conv)) {
        const auto plan_it = m.pattern_ocblock4_plans.find(conv_name);
        if (plan_it != m.pattern_ocblock4_plans.end()) {
            conv3x3_pattern_ocblock4_s1_bn_relu_avx2(x, conv, plan_it->second, bn_fused, workspace, y_out);
            return;
        }
    }

    if (enable_pattern_ocblock2_spatial4_kernel && can_use_pattern_ocblock2_spatial4_stride1_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock2_spatial4_bn_relu_avx2(x, conv, m.pattern_ocblock2_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer2_s2_kernel &&
        can_use_q8_ocblock2_spatial4_layer2_s2_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 8;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer2_s2_bn_relu_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_pattern_ocblock2_spatial2_kernel && can_use_pattern_ocblock2_spatial2_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock2_spatial2_bn_relu_avx2(x, conv, m.pattern_ocblock2_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock2_kernel && can_use_pattern_ocblock2_3x3_avx2(conv)) {
        conv3x3s1_pattern_ocblock2_bn_relu_avx2(x, conv, m.pattern_ocblock2_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_ocblock_kernel && can_use_sparse_ocblock4_3x3s1_avx2(conv)) {
        conv3x3s1_sparse_ocblock4_bn_relu_avx2(x, conv, m.sparse_ocblock4_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (can_use_avx2_width_kernel(conv)) {
        conv2d_ternary_sparse_padded_bn_relu_avx2_width(x, conv, plan, bn_fused, workspace, y_out);
        return;
    }

    conv2d_ternary_sparse_padded_bn_relu_scalar(x, conv, plan, bn_fused, y_out);
}

void conv_bn(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name, EngineWorkspace& workspace, Tensor& y_out) {
    const Conv2DParams& conv = m.convs.at(conv_name);
    const ConvPlan& plan = m.conv_plans.at(conv_name);
    const BNFusedParams& bn_fused = m.fused_bns.at(bn_name);
    constexpr bool enable_ocblock_kernel = false;
    constexpr bool enable_pattern_ocblock2_kernel = true;
    constexpr bool enable_pattern_ocblock2_spatial4_kernel = true;
    constexpr bool enable_pattern_ocblock2_spatial2_kernel = true;
    constexpr bool enable_pattern_ocblock4_layer4_kernel = true;
    constexpr bool enable_pattern_ocblock4_general_s1_kernel = false;
    constexpr bool enable_q8_layer4_stride1_kernel = true;
    constexpr bool enable_q8_layer3_s1_kernel = true;
    constexpr bool enable_q8_layer2_s1_kernel = false;
    constexpr bool enable_q8_layer1_s1_kernel = false;
    constexpr bool enable_q8_ocblock2_layer1_s1_kernel = true;
    constexpr bool enable_q8_ocblock2_layer2_s1_kernel = true;
    constexpr bool enable_q8_ocblock2_layer2_s2_kernel = true;
    constexpr bool enable_q8_ocblock2_layer3_s1_kernel = true;
    constexpr bool enable_q8_ocblock2_layer4_s1_kernel = true;
    if (enable_q8_ocblock2_layer4_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer4_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_q8_layer4_stride1_kernel && can_use_q8_ocblock4_layer4_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int l4_extra_right_pad = 1;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + l4_extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer4_s1_bn_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer3_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer3_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_q8_layer3_s1_kernel &&
        can_use_q8_ocblock4_layer3_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int extra_right_pad = 2;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer3_s1_bn_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_layer2_s1_kernel &&
        can_use_q8_ocblock4_layer2_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer2_s1_bn_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_layer1_s1_kernel &&
        can_use_q8_ocblock4_layer1_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan = m.pattern_ocblock4_plans.at(conv_name);
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w, workspace.q8_padded_input);
        conv3x3_q8_ocblock4_layer1_s1_bn_avx2(xq, conv, q8_plan, bn_fused, workspace, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer2_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer2_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 4;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w + extra_right_pad, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer2_s1_bn_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_q8_ocblock2_layer1_s1_kernel &&
        can_use_q8_ocblock2_spatial4_layer1_s1_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, conv.pad_h, conv.pad_h, conv.pad_w, conv.pad_w, workspace.q8_padded_input);
        conv3x3_q8_ocblock2_spatial4_layer1_s1_bn_avx2(xq, conv, q8_plan, bn_fused, y_out);
        return;
    }

    if (enable_pattern_ocblock4_layer4_kernel && can_use_pattern_ocblock4_layer4_s2_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock4_layer4_s2_bn_avx2(x, conv, m.pattern_ocblock4_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock4_layer4_kernel && can_use_pattern_ocblock4_layer4_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock4_layer4_bn_avx2(x, conv, m.pattern_ocblock4_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock4_general_s1_kernel && can_use_pattern_ocblock4_s1_3x3_avx2(conv)) {
        const auto plan_it = m.pattern_ocblock4_plans.find(conv_name);
        if (plan_it != m.pattern_ocblock4_plans.end()) {
            conv3x3_pattern_ocblock4_s1_bn_avx2(x, conv, plan_it->second, bn_fused, workspace, y_out);
            return;
        }
    }

    if (enable_pattern_ocblock2_spatial4_kernel && can_use_pattern_ocblock2_spatial4_stride1_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock2_spatial4_bn_avx2(x, conv, m.pattern_ocblock2_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock2_spatial2_kernel && can_use_pattern_ocblock2_spatial2_3x3_avx2(conv)) {
        conv3x3_pattern_ocblock2_spatial2_bn_avx2(x, conv, m.pattern_ocblock2_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_pattern_ocblock2_kernel && can_use_pattern_ocblock2_3x3_avx2(conv)) {
        conv3x3s1_pattern_ocblock2_bn_avx2(x, conv, m.pattern_ocblock2_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (enable_ocblock_kernel && can_use_sparse_ocblock4_3x3s1_avx2(conv)) {
        conv3x3s1_sparse_ocblock4_bn_avx2(x, conv, m.sparse_ocblock4_plans.at(conv_name), bn_fused, workspace, y_out);
        return;
    }

    if (can_use_q8_1x1_s2_avx2(conv)) {
        const PatternOCBlock2ConvPlan& q8_plan = m.pattern_ocblock2_plans.at(conv_name);
        constexpr int extra_right_pad = 8;
        const int out_h = x.h / conv.stride_h;
        const int out_w = x.w / conv.stride_w;
        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x, 0, 0, 0, extra_right_pad, workspace.q8_padded_input);
        conv1x1_q8_s2_bn_avx2(xq, conv, q8_plan, bn_fused, out_h, out_w, y_out);
        return;
    }

    if (can_use_avx2_width_kernel(conv)) {
        conv2d_ternary_sparse_padded_bn_avx2_width(x, conv, plan, bn_fused, workspace, y_out);
        return;
    }
    conv2d_ternary_sparse_padded_bn_scalar(x, conv, plan, bn_fused, y_out);
}

// In-place basic block: reads from ws.buf_x, writes result back to ws.buf_x.
// Uses ws.buf_a and ws.buf_b as intermediates; ws.buf_identity for downsample.
void basic_block_ws(
    const Model& m,
    const BasicBlockSpec& block,
    EngineWorkspace& ws
) {
    if (block.has_downsample) {
        conv_bn(m, ws.buf_x, block.downsample_conv, block.downsample_bn, ws, ws.buf_identity);
    }
    conv_bn_relu(m, ws.buf_x, block.conv1, block.bn1, ws, ws.buf_a);
    conv_bn(m, ws.buf_a, block.conv2, block.bn2, ws, ws.buf_b);

    const Tensor& identity = block.has_downsample ? ws.buf_identity : ws.buf_x;
    add_inplace(ws.buf_b, identity);
    relu_inplace(ws.buf_b);
    std::swap(ws.buf_x, ws.buf_b);
}

// Timed versions of blocks for profiling
Tensor conv_bn_relu_timed(
    const Model& m,
    const Tensor& x,
    const std::string& conv_name,
    const std::string& bn_name,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
) {
    if (should_dump_intermediate_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_input_fp32.bin",
            x
        );
    }

    Tensor y;
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        conv_bn_relu(m, x, conv_name, bn_name, workspace, y);
        const auto t1 = std::chrono::high_resolution_clock::now();
        char name_buf[64];
        std::snprintf(name_buf, sizeof(name_buf), "%s+%s+relu", conv_name.c_str(), bn_name.c_str());
        timings.push_back(make_timing_entry(name_buf,
            std::chrono::duration<double, std::milli>(t1 - t0).count()));
    }

    if (should_dump_intermediate_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_output_fp32.bin",
            y
        );
    }

    return y;
}
Tensor conv_bn_timed(
    const Model& m,
    const Tensor& x,
    const std::string& conv_name,
    const std::string& bn_name,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
) {
    if (should_dump_intermediate_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_input_fp32.bin",
            x
        );
    }

    Tensor y;
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        conv_bn(m, x, conv_name, bn_name, workspace, y);
        const auto t1 = std::chrono::high_resolution_clock::now();
        char name_buf[64];
        std::snprintf(name_buf, sizeof(name_buf), "%s+%s", conv_name.c_str(), bn_name.c_str());
        timings.push_back(make_timing_entry(name_buf,
            std::chrono::duration<double, std::milli>(t1 - t0).count()));
    }

    if (should_dump_intermediate_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_output_fp32.bin",
            y
        );
    }

    return y;
}
Tensor add_relu_timed(const Tensor& a, const Tensor& b, const std::string& name, std::vector<TimingEntry>& timings) {
    char label[64];
    std::snprintf(label, sizeof(label), "%s+add+relu", name.c_str());
    return timed_op(timings, label, [&]() {
        Tensor y = add_tensors(a, b);
        y = relu(y);
        return y;
    });
}
Tensor basic_block_timed(const Model& m, const Tensor& x, const std::string& prefix, bool has_downsample, EngineWorkspace& workspace, std::vector<TimingEntry>& timings) {
    Tensor identity = x;

    Tensor y = conv_bn_relu_timed(m,x,(prefix + ".conv1").c_str(),(prefix + ".bn1").c_str(),workspace,timings);

    // if (prefix == "layer4.0") {
    //     write_tensor_f32("artifacts/layer4_0_conv2_input_fp32.bin", y);
    // }

    y = conv_bn_timed(m,y,(prefix + ".conv2").c_str(),(prefix + ".bn2").c_str(),workspace,timings);

    // if (prefix == "layer4.0") {
    //     write_tensor_f32("artifacts/layer4_0_conv2_output_fp32.bin", y);
    // }

    if (has_downsample) {
        identity = conv_bn_timed(m,x,(prefix + ".downsample.0").c_str(),(prefix + ".downsample.1").c_str(),workspace,timings);
    }

    y = add_relu_timed(y, identity, prefix, timings);

    return y;
}
// Workspace-mirrored timed block: same buffer/swap pattern as basic_block_ws, no return-by-value.
static void basic_block_ws_timed(
    const Model& m,
    const BasicBlockSpec& block,
    EngineWorkspace& ws,
    std::vector<TimingEntry>& timings
) {
    auto time_op = [&](const char* name, auto fn) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        timings.push_back(make_timing_entry(name,
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count()));
    };

    if (block.has_downsample) {
        time_op(block.downsample_conv, [&]() {
            conv_bn(m, ws.buf_x, block.downsample_conv, block.downsample_bn, ws, ws.buf_identity);
        });
    }

    time_op(block.conv1, [&]() {
        conv_bn_relu(m, ws.buf_x, block.conv1, block.bn1, ws, ws.buf_a);
    });

    time_op(block.conv2, [&]() {
        conv_bn(m, ws.buf_a, block.conv2, block.bn2, ws, ws.buf_b);
    });

    const Tensor& identity = block.has_downsample ? ws.buf_identity : ws.buf_x;
    char add_relu_label[64];
    std::snprintf(add_relu_label, sizeof(add_relu_label), "%s.add_relu", block.name);
    time_op(add_relu_label, [&]() {
        add_inplace(ws.buf_b, identity);
        relu_inplace(ws.buf_b);
    });

    std::swap(ws.buf_x, ws.buf_b);
}

} // namespace

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input, EngineWorkspace& ws) {
    conv_bn_relu(model, input, "conv1", "bn1", ws, ws.buf_a);
    maxpool2d_3x3_s2_p1_into(ws.buf_a, ws.buf_x);

    basic_block_ws(model, kLayer1_0, ws);
    basic_block_ws(model, kLayer1_1, ws);

    basic_block_ws(model, kLayer2_0, ws);
    basic_block_ws(model, kLayer2_1, ws);

    basic_block_ws(model, kLayer3_0, ws);
    basic_block_ws(model, kLayer3_1, ws);

    basic_block_ws(model, kLayer4_0, ws);
    basic_block_ws(model, kLayer4_1, ws);

    adaptive_avgpool_1x1_into(ws.buf_x, ws.buf_avgpool);
    linear_into(ws.buf_avgpool, model.linears.at("fc"), ws.buf_fc);
    return ws.buf_fc;
}

Tensor forward_resnet18_workspace_timed(
    const Model& model,
    const Tensor& input,
    EngineWorkspace& ws,
    std::vector<TimingEntry>& timings
) {
    auto time_op = [&](const char* name, auto fn) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        timings.push_back(make_timing_entry(name,
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count()));
    };

    time_op("conv1+bn1+relu", [&]() {
        conv_bn_relu(model, input, "conv1", "bn1", ws, ws.buf_a);
    });
    time_op("maxpool", [&]() {
        maxpool2d_3x3_s2_p1_into(ws.buf_a, ws.buf_x);
    });

    basic_block_ws_timed(model, kLayer1_0, ws, timings);
    basic_block_ws_timed(model, kLayer1_1, ws, timings);
    basic_block_ws_timed(model, kLayer2_0, ws, timings);
    basic_block_ws_timed(model, kLayer2_1, ws, timings);
    basic_block_ws_timed(model, kLayer3_0, ws, timings);
    basic_block_ws_timed(model, kLayer3_1, ws, timings);
    basic_block_ws_timed(model, kLayer4_0, ws, timings);
    basic_block_ws_timed(model, kLayer4_1, ws, timings);

    time_op("avgpool", [&]() { adaptive_avgpool_1x1_into(ws.buf_x, ws.buf_avgpool); });
    time_op("fc", [&]() { linear_into(ws.buf_avgpool, model.linears.at("fc"), ws.buf_fc); });
    return ws.buf_fc;
}

// Legacy debug-only timing path.
// Not used for reported benchmark numbers because it uses return-by-value Tensor temporaries.
// Use forward_resnet18_workspace_timed for profiling the optimized workspace-backed engine.
Tensor forward_resnet18_scalar_timed(
    const Model& model,
    const Tensor& input,
    EngineWorkspace& workspace,
    std::vector<TimingEntry>& timings
) {
    Tensor x = conv_bn_relu_timed(
        model,
        input,
        "conv1",
        "bn1",
        workspace,
        timings
    );

    x = timed_op(timings, "maxpool", [&]() {
        return maxpool2d_3x3_s2_p1(x);
    });

    x = basic_block_timed(model, x, "layer1.0", false, workspace, timings);
    x = basic_block_timed(model, x, "layer1.1", false, workspace, timings);

    x = basic_block_timed(model, x, "layer2.0", true, workspace, timings);
    x = basic_block_timed(model, x, "layer2.1", false, workspace, timings);

    x = basic_block_timed(model, x, "layer3.0", true, workspace, timings);
    x = basic_block_timed(model, x, "layer3.1", false, workspace, timings);

    x = basic_block_timed(model, x, "layer4.0", true, workspace, timings);
    x = basic_block_timed(model, x, "layer4.1", false, workspace, timings);

    x = timed_op(timings, "avgpool", [&]() {
        return adaptive_avgpool_1x1(x);
    });

    x = timed_op(timings, "fc", [&]() {
        return linear(x, model.linears.at("fc"));
    });

    return x;
}

void set_tensor_dump_layers(const std::vector<std::string>& layer_names) {
    g_tensor_dump_layers.clear();
    for (const auto& name : layer_names) {
        g_tensor_dump_layers.insert(name);
    }
    g_tensor_dump_enabled = !g_tensor_dump_layers.empty();
}

void clear_tensor_dump_layers() {
    g_tensor_dump_enabled = false;
    g_tensor_dump_layers.clear();
}