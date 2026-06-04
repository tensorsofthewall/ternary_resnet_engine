#include "model_loader.hpp"
#include "conv_plan.hpp"
#include "bn_fusion.hpp"
#include "sparse_ocblock_plan.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint32_t read_u32(std::ifstream& f) {
    uint32_t value = 0;
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!f) {
        throw std::runtime_error("Failed to read uint32");
    }
    return value;
}

float read_f32(std::ifstream& f) {
    float value = 0.0f;
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!f) {
        throw std::runtime_error("Failed to read float32");
    }
    return value;
}

bool read_bool(std::ifstream& f) {
    bool value = false;
    f.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!f) {
        throw std::runtime_error("Failed to read bool");
    }
    return value;
}

std::string read_string(std::ifstream& f) {
    const uint32_t len = read_u32(f);
    std::string s(len, '\0');
    f.read(s.data(), static_cast<std::streamsize>(len));
    if (!f) {
        throw std::runtime_error("Failed to read string");
    }
    return s;
}

template <typename T>
std::vector<T> read_array(std::ifstream& f) {
    const uint32_t ndim = read_u32(f);

    uint64_t total = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        total *= read_u32(f);
    }

    std::vector<T> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(total * sizeof(T)));
    if (!f) {
        throw std::runtime_error("Failed to read array payload");
    }

    return data;
}

Conv2DParams read_conv(std::ifstream& f, const std::string& name) {
    Conv2DParams p;
    p.name = name;

    p.in_channels = static_cast<int>(read_u32(f));
    p.out_channels = static_cast<int>(read_u32(f));

    p.kh = static_cast<int>(read_u32(f));
    p.kw = static_cast<int>(read_u32(f));
    p.stride_h = static_cast<int>(read_u32(f));
    p.stride_w = static_cast<int>(read_u32(f));
    p.pad_h = static_cast<int>(read_u32(f));
    p.pad_w = static_cast<int>(read_u32(f));
    p.dilation_h = static_cast<int>(read_u32(f));
    p.dilation_w = static_cast<int>(read_u32(f));

    p.has_bias = read_bool(f);
    p.weight = read_array<int8_t>(f);

    if (p.has_bias) {
        p.bias = read_array<float>(f);
    }

    return p;
}

BatchNormParams read_bn(std::ifstream& f, const std::string& name) {
    BatchNormParams p;
    p.name = name;

    p.num_features = static_cast<int>(read_u32(f));
    p.eps = read_f32(f);

    p.gamma = read_array<float>(f);
    p.beta = read_array<float>(f);
    p.running_mean = read_array<float>(f);
    p.running_var = read_array<float>(f);

    return p;
}

LinearParams read_linear(std::ifstream& f, const std::string& name) {
    LinearParams p;
    p.name = name;

    p.in_features = static_cast<int>(read_u32(f));
    p.out_features = static_cast<int>(read_u32(f));
    p.has_bias = read_bool(f);

    p.weight = read_array<float>(f);

    if (p.has_bias) {
        p.bias = read_array<float>(f);
    }

    return p;
}

} // namespace

Model load_model(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open model file: " + path.string());
    }

    char magic[8] = {};
    f.read(magic, 8);
    if (std::string(magic, 8) != "TRNRES18") {
        throw std::runtime_error("Invalid model magic");
    }

    const uint32_t version = read_u32(f);
    if (version != 1) {
        throw std::runtime_error("Unsupported model version");
    }

    const uint32_t num_records = read_u32(f);

    Model model;

    for (uint32_t i = 0; i < num_records; ++i) {
        const std::string record_type = read_string(f);
        const std::string name = read_string(f);

        if (record_type == "conv2d") {
            model.convs.emplace(name, read_conv(f, name));
        } else if (record_type == "batchnorm2d") {
            model.bns.emplace(name, read_bn(f, name));
        } else if (record_type == "linear") {
            model.linears.emplace(name, read_linear(f, name));
        } else {
            throw std::runtime_error("Unknown record type: " + record_type);
        }
    }

    build_conv_plans(model);
    build_sparse_ocblock4_conv_plans(model);
    build_fused_bns(model);
    return model;
}
