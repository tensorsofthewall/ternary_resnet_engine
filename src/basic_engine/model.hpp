#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct Conv2DParams {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;

    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int dilation_h = 0;
    int dilation_w = 0;

    bool has_bias = false;

    // Shape: [out_channels, in_channels, kh, kw]
    std::vector<int8_t> weight;
    std::vector<float> bias;
};

struct BatchNormParams {
    std::string name;

    int num_features = 0;
    float eps = 0.0f;

    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> running_mean;
    std::vector<float> running_var;
};

struct BNFusedParams {
    std::string name;
    int num_features=0;
    
    // y = x*scale[c]+bias[c]
    std::vector<float> scale;
    std::vector<float> bias;
};

struct LinearParams {
    std::string name;

    int in_features = 0;
    int out_features = 0;
    bool has_bias = false;

    // Shape: [out_features, in_features]
    std::vector<float> weight;
    std::vector<float> bias;
};

struct WeightIndex {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;
};

struct ConvOCPlan {
    std::vector<WeightIndex> pos;
    std::vector<WeightIndex> neg;
};

struct ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<ConvOCPlan> oc_plans;
};

// Sparse OC-Block4 definitions
struct OCBlock4Action {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;

    int8_t w0 = 0;
    int8_t w1 = 0;
    int8_t w2 = 0;
    int8_t w3 = 0;
};

struct OCBlock4Plan {
    int oc_start = 0; // starting output channel index of the block
    std::vector<OCBlock4Action> actions;
};

struct SparseOCBlock4ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<OCBlock4Plan> oc_blocks;
};

// Sparse OC-Block2 sign-pattern definitions
struct OCBlock2Action {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;
};

struct OCBlock2PatternBucket {
    int8_t w0 = 0;
    int8_t w1 = 0;

    std::vector<OCBlock2Action> actions;
};

struct OCBlock2PatternPlan {
    int oc_start = 0;

    std::vector<OCBlock2PatternBucket> buckets;
};

struct PatternOCBlock2ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<OCBlock2PatternPlan> oc_blocks;
};

struct Model {
    std::unordered_map<std::string, Conv2DParams> convs;
    std::unordered_map<std::string, ConvPlan> conv_plans;
    std::unordered_map<std::string, SparseOCBlock4ConvPlan> sparse_ocblock4_plans;
    std::unordered_map<std::string, PatternOCBlock2ConvPlan> pattern_ocblock2_plans;
    
    std::unordered_map<std::string, BatchNormParams> bns;
    std::unordered_map<std::string, BNFusedParams> fused_bns;
    std::unordered_map<std::string, LinearParams> linears;
};