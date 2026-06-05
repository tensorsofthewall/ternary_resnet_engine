#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include "model_loader.hpp"
#include "quantized/quant_utils.hpp"
#include "quantized/conv3x3_q8_pattern_ocblock4_avx2.hpp"
#include "conv3x3_pattern_ocblock4_avx2.hpp"

struct LatencyStats {
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double p90_ms = 0.0;
};

LatencyStats compute_latency_stats(std::vector<double> times_ms) {
    std::sort(times_ms.begin(), times_ms.end());

    const double sum =
        std::accumulate(times_ms.begin(), times_ms.end(), 0.0);

    LatencyStats s;
    s.min_ms = times_ms.front();
    s.median_ms = times_ms[times_ms.size() / 2];
    s.mean_ms = sum / static_cast<double>(times_ms.size());

    const size_t p90_idx =
        static_cast<size_t>(0.9 * static_cast<double>(times_ms.size() - 1));
    s.p90_ms = times_ms[p90_idx];

    return s;
}

std::string sanitize_tensor_name(std::string name) {
    for (char& c : name) {
        if (c == '.') {
            c = '_';
        }
    }

    return name;
}

std::vector<float> read_f32_file(
    const std::filesystem::path& path,
    size_t expected_count
) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::vector<float> data(expected_count);
    f.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(expected_count * sizeof(float))
    );

    if (!f) {
        throw std::runtime_error("Failed to read expected float count from: " + path.string());
    }

    return data;
}

void print_stats(const char* name, const LatencyStats& s) {
    std::cout << name << "\n";
    std::cout << "  min_ms:    " << s.min_ms << "\n";
    std::cout << "  median_ms: " << s.median_ms << "\n";
    std::cout << "  mean_ms:   " << s.mean_ms << "\n";
    std::cout << "  p90_ms:    " << s.p90_ms << "\n";
}

void benchmark_q8_layer4_conv(
    Model& model,
    const std::string& conv_name,
    const std::string& bn_name,
    bool apply_relu,
    int runs,
    int warmup
) {
    EngineWorkspace workspace;

    const Conv2DParams& conv = model.convs.at(conv_name);
    const BNFusedParams& bn = model.fused_bns.at(bn_name);
    const PatternOCBlock4ConvPlan& plan =
        model.pattern_ocblock4_plans.at(conv_name);

    const std::string stem = sanitize_tensor_name(conv_name);

    Tensor x(512, 7, 7);
    x.data = read_f32_file(
        "artifacts/" + stem + "_input_fp32.bin",
        static_cast<size_t>(512 * 7 * 7)
    );

    const std::vector<float> golden = read_f32_file(
        "artifacts/" + stem + "_output_fp32.bin",
        static_cast<size_t>(512 * 7 * 7)
    );

    QuantTensorI8 xq_raw = quantize_tensor_symmetric_i8(x);
    QuantTensorI8 xq = pad_quant_tensor_i8(xq_raw, conv.pad_h, conv.pad_w);

    Tensor y_q8;

    for (int i = 0; i < warmup; ++i) {
        if (apply_relu) {
            y_q8 = conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
                xq,
                conv,
                plan,
                bn,
                workspace
            );
        } else {
            y_q8 = conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
                xq,
                conv,
                plan,
                bn,
                workspace
            );
        }
    }

    std::vector<double> q8_times;
    q8_times.reserve(static_cast<size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();

        if (apply_relu) {
            y_q8 = conv3x3_q8_ocblock4_layer4_s1_bn_relu_avx2(
                xq,
                conv,
                plan,
                bn,
                workspace
            );
        } else {
            y_q8 = conv3x3_q8_ocblock4_layer4_s1_bn_avx2(
                xq,
                conv,
                plan,
                bn,
                workspace
            );
        }

        const auto end = std::chrono::high_resolution_clock::now();

        q8_times.push_back(
            std::chrono::duration<double, std::milli>(end - start).count()
        );
    }

    float max_abs = 0.0f;
    double mean_abs = 0.0;
    double mean_abs_golden = 0.0;

    for (size_t i = 0; i < golden.size(); ++i) {
        const float err = std::abs(y_q8.data[i] - golden[i]);
        max_abs = std::max(max_abs, err);
        mean_abs += static_cast<double>(err);
        mean_abs_golden += std::abs(static_cast<double>(golden[i]));
    }

    mean_abs /= static_cast<double>(golden.size());
    mean_abs_golden /= static_cast<double>(golden.size());

    std::cout << "\nQ8 " << conv_name << " vs FP32 golden\n";
    std::cout << "  input_scale:              " << xq_raw.scale << "\n";
    std::cout << "  max_abs_error:            " << max_abs << "\n";
    std::cout << "  mean_abs_error:           " << mean_abs << "\n";
    std::cout << "  mean_abs_golden:          " << mean_abs_golden << "\n";
    std::cout << "  relative_mean_abs_error:  "
              << (mean_abs_golden > 0.0 ? mean_abs / mean_abs_golden : 0.0)
              << "\n\n";

    const LatencyStats q8_stats = compute_latency_stats(q8_times);

    print_stats(("Q8 " + conv_name + " kernel only").c_str(), q8_stats);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_layer4_q8 <model.bin>\n";
        return 1;
    }

    const auto model_path = std::filesystem::path(argv[1]);

    Model model = load_model(model_path);

    constexpr int warmup = 20;
    constexpr int runs = 100;

    benchmark_q8_layer4_conv(
        model,
        "layer4.0.conv2",
        "layer4.0.bn2",
        false,
        runs,
        warmup
    );

    benchmark_q8_layer4_conv(
        model,
        "layer4.1.conv1",
        "layer4.1.bn1",
        true,
        runs,
        warmup
    );

    benchmark_q8_layer4_conv(
        model,
        "layer4.1.conv2",
        "layer4.1.bn2",
        false,
        runs,
        warmup
    );

    return 0;
}