#include "blocked_resnet18.hpp"

#include "blocked_conv1.hpp"
#include "blocked_conv3x3.hpp"
#include "blocked_ops.hpp"

#include <algorithm>
#include <stdexcept>

BlockedModel build_blocked_model(const Model& m) {
    BlockedModel bm;
    bm.model = &m;
    for (const auto& [name, conv] : m.convs) {
        // Skip conv1: IC=3 is not divisible by 16 — handled by blocked_conv1_7x7_s2.
        if (conv.in_channels % 16 != 0 || conv.out_channels % 16 != 0) continue;
        bm.blocked_convs.emplace(name, build_blocked_conv_weights(conv));
    }
    return bm;
}

std::vector<float> forward_blocked_resnet18(
    const BlockedModel& bm,
    const Tensor& input,
    BlockedEngineWorkspace& ws,
    std::unordered_map<std::string, double>& /*timings*/)
{
    const Model& m = *bm.model;

    // Lookup helpers — avoid repeated at() calls in the hot path.
    auto& bc = bm.blocked_convs;
    auto& fb = m.fused_bns;

    // Shorthand: stride-1 conv3x3 + BN + optional ReLU.
    auto s1 = [&](const BlockedTensorFP32& x, const char* cn, const char* bn,
                   bool relu, BlockedTensorFP32& y) {
        blocked_conv3x3_s1_ref(x, bc.at(cn), fb.at(bn), relu, y);
    };
    // Stride-2 conv3x3.
    auto s2 = [&](const BlockedTensorFP32& x, const char* cn, const char* bn,
                   bool relu, BlockedTensorFP32& y) {
        blocked_conv3x3_s2_ref(x, bc.at(cn), fb.at(bn), relu, y);
    };
    // 1×1 stride-2 downsample conv.
    auto ds = [&](const BlockedTensorFP32& x, const char* cn, const char* bn,
                   bool relu, BlockedTensorFP32& y) {
        blocked_conv1x1_s2_ref(x, bc.at(cn), fb.at(bn), relu, y);
    };

    // ---- conv1 + bn1 + relu → maxpool ----
    blocked_conv1_7x7_s2(input, m.convs.at("conv1"), fb.at("bn1"), ws.conv1_out);
    blocked_maxpool_3x3_s2(ws.conv1_out, ws.pool_out);

    // ---- layer1.0 (no downsample, stride-1) ----
    s1(ws.pool_out,  "layer1.0.conv1", "layer1.0.bn1", true,  ws.l1_0_c1);
    s1(ws.l1_0_c1,  "layer1.0.conv2", "layer1.0.bn2", false, ws.l1_0_c2);
    blocked_residual_add_relu(ws.l1_0_c2, ws.pool_out);

    // ---- layer1.1 (no downsample, stride-1) ----
    s1(ws.l1_0_c2,  "layer1.1.conv1", "layer1.1.bn1", true,  ws.l1_1_c1);
    s1(ws.l1_1_c1,  "layer1.1.conv2", "layer1.1.bn2", false, ws.l1_1_c2);
    blocked_residual_add_relu(ws.l1_1_c2, ws.l1_0_c2);

    // ---- layer2.0 (downsample: conv1 stride-2, 1×1 skip stride-2) ----
    s2(ws.l1_1_c2,  "layer2.0.conv1", "layer2.0.bn1",          true,  ws.l2_0_c1);
    s1(ws.l2_0_c1,  "layer2.0.conv2", "layer2.0.bn2",          false, ws.l2_0_c2);
    ds(ws.l1_1_c2,  "layer2.0.downsample.0", "layer2.0.downsample.1", false, ws.l2_0_ds);
    blocked_residual_add_relu(ws.l2_0_c2, ws.l2_0_ds);

    // ---- layer2.1 (no downsample, stride-1) ----
    s1(ws.l2_0_c2,  "layer2.1.conv1", "layer2.1.bn1", true,  ws.l2_1_c1);
    s1(ws.l2_1_c1,  "layer2.1.conv2", "layer2.1.bn2", false, ws.l2_1_c2);
    blocked_residual_add_relu(ws.l2_1_c2, ws.l2_0_c2);

    // ---- layer3.0 (downsample) ----
    s2(ws.l2_1_c2,  "layer3.0.conv1", "layer3.0.bn1",          true,  ws.l3_0_c1);
    s1(ws.l3_0_c1,  "layer3.0.conv2", "layer3.0.bn2",          false, ws.l3_0_c2);
    ds(ws.l2_1_c2,  "layer3.0.downsample.0", "layer3.0.downsample.1", false, ws.l3_0_ds);
    blocked_residual_add_relu(ws.l3_0_c2, ws.l3_0_ds);

    // ---- layer3.1 (no downsample, stride-1) ----
    s1(ws.l3_0_c2,  "layer3.1.conv1", "layer3.1.bn1", true,  ws.l3_1_c1);
    s1(ws.l3_1_c1,  "layer3.1.conv2", "layer3.1.bn2", false, ws.l3_1_c2);
    blocked_residual_add_relu(ws.l3_1_c2, ws.l3_0_c2);

    // ---- layer4.0 (downsample) ----
    s2(ws.l3_1_c2,  "layer4.0.conv1", "layer4.0.bn1",          true,  ws.l4_0_c1);
    s1(ws.l4_0_c1,  "layer4.0.conv2", "layer4.0.bn2",          false, ws.l4_0_c2);
    ds(ws.l3_1_c2,  "layer4.0.downsample.0", "layer4.0.downsample.1", false, ws.l4_0_ds);
    blocked_residual_add_relu(ws.l4_0_c2, ws.l4_0_ds);

    // ---- layer4.1 (no downsample, stride-1) ----
    s1(ws.l4_0_c2,  "layer4.1.conv1", "layer4.1.bn1", true,  ws.l4_1_c1);
    s1(ws.l4_1_c1,  "layer4.1.conv2", "layer4.1.bn2", false, ws.l4_1_c2);
    blocked_residual_add_relu(ws.l4_1_c2, ws.l4_0_c2);

    // ---- global avgpool → FC → softmax ----
    blocked_avgpool_global(ws.l4_1_c2, ws.pool_final);

    std::vector<float> logits;
    blocked_fc(ws.pool_final, m.linears.at("fc"), logits);
    softmax_inplace(logits);

    return logits;
}
