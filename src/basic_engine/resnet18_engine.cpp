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

#include <stdexcept>

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
constexpr bool enable_tensor_dumps = false;

bool should_dump_layer4_q8_tensor(const std::string& name) {
    if constexpr (!enable_tensor_dumps) {
        return false;
    }

    return name == "layer3.0.conv1";
}
std::string sanitize_tensor_name(std::string name) {
    for (char& c : name) {
        if (c == '.') {
            c = '_';
        }
    }

    return name;
}

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
Tensor conv_bn_relu(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name, EngineWorkspace& workspace) {
    const Conv2DParams& conv = m.convs.at(conv_name);
    const ConvPlan& plan = m.conv_plans.at(conv_name);
    const BNFusedParams bn_fused = m.fused_bns.at(bn_name);
    constexpr bool enable_ocblock_kernel = false;
    constexpr bool enable_pattern_ocblock2_kernel = true;
    constexpr bool enable_pattern_ocblock2_spatial2_kernel = true;
    constexpr bool enable_pattern_ocblock4_layer4_kernel = true;
    constexpr bool enable_conv1_ocblock4_kernel = true;
    constexpr bool enable_q8_layer4_stride1_kernel = true;
    constexpr bool enable_q8_layer3_s2_kernel = false;

    if (enable_q8_layer4_stride1_kernel &&
        can_use_q8_ocblock4_layer4_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_into(
            x,
            conv.pad_h,
            conv.pad_w,
            workspace.q8_padded_input
        );

        return conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
            xq,
            conv,
            q8_plan,
            bn_fused,
            workspace
        );
    }

    const bool is_q8_layer3_0_conv1 = conv_name == "layer3.0.conv1";
    if (enable_q8_layer3_s2_kernel &&
        is_q8_layer3_0_conv1 &&
        can_use_q8_ocblock4_s2_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        constexpr int extra_right_pad = 4;

        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x,
            conv.pad_h,
            conv.pad_h,
            conv.pad_w,
            conv.pad_w + extra_right_pad,
            workspace.q8_padded_input_s2
        );

        return conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
            xq,
            14,
            14,
            conv,
            q8_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_conv1_ocblock4_kernel &&
        can_use_conv1_ocblock4_7x7s2_avx2(conv)) {
        const auto plan_it = m.conv1_ocblock4_plans.find(conv_name);

        if (plan_it == m.conv1_ocblock4_plans.end()) {
            throw std::runtime_error(
                "Missing conv1 OC-Block4 plan for conv: " + conv_name
            );
        }

        return conv1_ocblock4_7x7s2_bn_relu_avx2(
            x,
            conv,
            plan_it->second,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock4_layer4_kernel &&
        can_use_pattern_ocblock4_layer4_s2_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& ocblock4_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        return conv3x3_pattern_ocblock4_layer4_s2_bn_relu_avx2(
            x,
            conv,
            ocblock4_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock4_layer4_kernel &&
        can_use_pattern_ocblock4_layer4_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& ocblock4_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        return conv3x3_pattern_ocblock4_layer4_bn_relu_avx2(
            x,
            conv,
            ocblock4_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock2_spatial2_kernel &&
        can_use_pattern_ocblock2_spatial2_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& pattern_plan =
            m.pattern_ocblock2_plans.at(conv_name);

        return conv3x3_pattern_ocblock2_spatial2_bn_relu_avx2(
            x,
            conv,
            pattern_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock2_kernel &&
        can_use_pattern_ocblock2_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& pattern_plan =
            m.pattern_ocblock2_plans.at(conv_name);

        return conv3x3s1_pattern_ocblock2_bn_relu_avx2(
            x,
            conv,
            pattern_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_ocblock_kernel && can_use_sparse_ocblock4_3x3s1_avx2(conv)) {
        const SparseOCBlock4ConvPlan& ocblock_plan = m.sparse_ocblock4_plans.at(conv_name);
        return conv3x3s1_sparse_ocblock4_bn_relu_avx2(x, conv, ocblock_plan, bn_fused, workspace);
    }

    if (can_use_avx2_width_kernel(conv)) {
        return conv2d_ternary_sparse_padded_bn_relu_avx2_width(x, conv, plan, bn_fused, workspace);
    }

    return conv2d_ternary_sparse_padded_bn_relu_scalar(x, conv, plan, bn_fused);
}

Tensor conv_bn(const Model& m, const Tensor& x, const std::string& conv_name, const std::string& bn_name, EngineWorkspace& workspace) {
    const Conv2DParams& conv = m.convs.at(conv_name);
    const ConvPlan& plan = m.conv_plans.at(conv_name);
    const BNFusedParams bn_fused = m.fused_bns.at(bn_name);
    constexpr bool enable_ocblock_kernel = false;
    constexpr bool enable_pattern_ocblock2_kernel = true;
    constexpr bool enable_pattern_ocblock2_spatial2_kernel = true;
    constexpr bool enable_pattern_ocblock4_layer4_kernel = true;
    constexpr bool enable_q8_layer4_stride1_kernel = true;

    if (enable_q8_layer4_stride1_kernel &&
        can_use_q8_ocblock4_layer4_s1_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& q8_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        QuantTensorI8& xq = quantize_pad_tensor_symmetric_i8_into(
            x,
            conv.pad_h,
            conv.pad_w,
            workspace.q8_padded_input
        );

        return conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
            xq,
            conv,
            q8_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock4_layer4_kernel &&
        can_use_pattern_ocblock4_layer4_s2_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& ocblock4_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        return conv3x3_pattern_ocblock4_layer4_s2_bn_avx2(
            x,
            conv,
            ocblock4_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock4_layer4_kernel &&
        can_use_pattern_ocblock4_layer4_3x3_avx2(conv)) {
        const PatternOCBlock4ConvPlan& ocblock4_plan =
            m.pattern_ocblock4_plans.at(conv_name);

        return conv3x3_pattern_ocblock4_layer4_bn_avx2(
            x,
            conv,
            ocblock4_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock2_spatial2_kernel &&
        can_use_pattern_ocblock2_spatial2_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& pattern_plan =
            m.pattern_ocblock2_plans.at(conv_name);

        return conv3x3_pattern_ocblock2_spatial2_bn_avx2(
            x,
            conv,
            pattern_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_pattern_ocblock2_kernel &&
        can_use_pattern_ocblock2_3x3_avx2(conv)) {
        const PatternOCBlock2ConvPlan& pattern_plan =
            m.pattern_ocblock2_plans.at(conv_name);

        return conv3x3s1_pattern_ocblock2_bn_avx2(
            x,
            conv,
            pattern_plan,
            bn_fused,
            workspace
        );
    }

    if (enable_ocblock_kernel && can_use_sparse_ocblock4_3x3s1_avx2(conv)) {
        const SparseOCBlock4ConvPlan& ocblock_plan = m.sparse_ocblock4_plans.at(conv_name);
        return conv3x3s1_sparse_ocblock4_bn_avx2(x, conv, ocblock_plan, bn_fused, workspace);
    }

    if (can_use_avx2_width_kernel(conv)) {
        return conv2d_ternary_sparse_padded_bn_avx2_width(x, conv, plan, bn_fused, workspace);
    }
    return conv2d_ternary_sparse_padded_bn_scalar(x, conv, plan, bn_fused);
}

Tensor basic_block(
    const Model& m,
    const Tensor& x,
    const char* prefix,
    bool has_downsample,
    EngineWorkspace& workspace
) {
    const std::string p(prefix);

    Tensor identity = x;

    Tensor y = conv_bn_relu(
        m,
        x,
        (p + ".conv1").c_str(),
        (p + ".bn1").c_str(),
        workspace
    );

    y = conv_bn(
        m,
        y,
        (p + ".conv2").c_str(),
        (p + ".bn2").c_str(),
        workspace
    );

    if (has_downsample) {
        identity = conv_bn(
            m,
            x,
            (p + ".downsample.0").c_str(),
            (p + ".downsample.1").c_str(),
            workspace
        );
    }

    y = add_tensors(y, identity);
    y = relu(y);

    return y;
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
    if (should_dump_layer4_q8_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_input_fp32.bin",
            x
        );
    }

    Tensor y = timed_op(timings, conv_name + "+" + bn_name + "+relu", [&]() {
        return conv_bn_relu(m, x, conv_name, bn_name, workspace);
    });

    if (should_dump_layer4_q8_tensor(conv_name)) {
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
    if (should_dump_layer4_q8_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_input_fp32.bin",
            x
        );
    }

    Tensor y = timed_op(timings, conv_name + "+" + bn_name, [&]() {
        return conv_bn(m, x, conv_name, bn_name, workspace);
    });

    if (should_dump_layer4_q8_tensor(conv_name)) {
        write_tensor_f32(
            "artifacts/" + sanitize_tensor_name(conv_name) + "_output_fp32.bin",
            y
        );
    }

    return y;
}
Tensor add_relu_timed(const Tensor& a, const Tensor& b, const std::string& name, std::vector<TimingEntry>& timings) {
    return timed_op(timings, name + "+add+relu", [&]() {
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
} // namespace

Tensor forward_resnet18_scalar(const Model& model, const Tensor& input, EngineWorkspace& workspace) {
    Tensor x = conv_bn_relu(model, input, "conv1", "bn1", workspace);
    x = maxpool2d_3x3_s2_p1(x);

    x = basic_block(model, x, "layer1.0", false, workspace);
    x = basic_block(model, x, "layer1.1", false, workspace);

    x = basic_block(model, x, "layer2.0", true, workspace);
    x = basic_block(model, x, "layer2.1", false, workspace);

    x = basic_block(model, x, "layer3.0", true, workspace);
    x = basic_block(model, x, "layer3.1", false, workspace);

    x = basic_block(model, x, "layer4.0", true, workspace);
    x = basic_block(model, x, "layer4.1", false, workspace);

    x = adaptive_avgpool_1x1(x);
    x = linear(x, model.linears.at("fc"));

    return x;
}

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