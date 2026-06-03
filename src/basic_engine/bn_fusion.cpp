#include "bn_fusion.hpp"

#include <cmath>

BNFusedParams build_fused_bn(const BatchNormParams& bn) {
    BNFusedParams fused;

    fused.name = bn.name;
    fused.num_features = bn.num_features;

    fused.scale.resize(static_cast<size_t>(bn.num_features));
    fused.bias.resize(static_cast<size_t>(bn.num_features));

    for (int c = 0; c < bn.num_features; ++c) {
        const size_t idx = static_cast<size_t>(c);

        const float scale =
            bn.gamma[idx] / std::sqrt(bn.running_var[idx] + bn.eps);

        const float bias =
            bn.beta[idx] - bn.running_mean[idx] * scale;

        fused.scale[idx] = scale;
        fused.bias[idx] = bias;
    }

    return fused;
}

void build_fused_bns(Model& model) {
    model.fused_bns.clear();

    for (const auto& [name, bn] : model.bns) {
        model.fused_bns.emplace(name, build_fused_bn(bn));
    }
}