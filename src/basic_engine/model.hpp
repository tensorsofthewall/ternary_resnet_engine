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

struct LinearParams {
    std::string name;

    int in_features = 0;
    int out_features = 0;
    bool has_bias = false;

    // Shape: [out_features, in_features]
    std::vector<float> weight;
    std::vector<float> bias;
};

struct Model {
    std::unordered_map<std::string, Conv2DParams> convs;
    std::unordered_map<std::string, BatchNormParams> bns;
    std::unordered_map<std::string, LinearParams> linears;
};