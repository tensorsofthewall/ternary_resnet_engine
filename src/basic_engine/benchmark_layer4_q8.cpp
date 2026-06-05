#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
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
    if (times_ms.empty()) {
        throw std::runtime_error("Cannot compute stats for empty latency vector");
    }

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
        throw std::runtime_error(
            "Failed to read expected float count from: " + path.string()
        );
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

struct ConvShape {
    int input_c = 0;
    int input_h = 0;
    int input_w = 0;
    int output_h = 0;
    int output_w = 0;
};

ConvShape infer_q8_benchmark_shape(
    const Conv2DParams& conv,
    const std::string& conv_name
) {
    // layer3.0.conv1: 128 -> 256, 3x3 stride-2, 28x28 -> 14x14
    if (conv.in_channels == 128 &&
        conv.out_channels == 256 &&
        conv.stride_h == 2 &&
        conv.stride_w == 2) {
        return ConvShape{128, 28, 28, 14, 14};
    }

    throw std::runtime_error(
        "Unsupported q8 stride-2 benchmark shape for conv: " + conv_name
    );
}

QuantTensorI8 quantize_and_pad_for_q8_s2_benchmark(
    const Tensor& x,
    const Conv2DParams& conv
) {
    QuantTensorI8 xq_raw = quantize_tensor_symmetric_i8(x);

    // Input 28x28, normal pad gives 30x30.
    //
    // For output width 14:
    //   last vector block starts at ow = 8
    //   stride-2 input col base = 8 * 2 = 16
    //   kx max = 2
    //   stride-2 load uses byte offsets 0,2,4,...,14
    //   max read col = 16 + 2 + 14 = 32
    //
    // Need physical width >= 33. Normal padded width is 30.
    // Add 4 extra right columns -> physical width = 34.
    constexpr int extra_right_pad = 4;

    return pad_quant_tensor_i8_asymmetric(
        xq_raw,
        conv.pad_h,
        conv.pad_h,
        conv.pad_w,
        conv.pad_w + extra_right_pad
    );
}

void benchmark_q8_stride2_conv(
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

    if (!can_use_q8_ocblock4_s2_3x3_avx2(conv)) {
        throw std::runtime_error(
            "q8 stride-2 OC-Block4 predicate rejected conv: " + conv_name
        );
    }

    const ConvShape shape = infer_q8_benchmark_shape(conv, conv_name);
    const std::string stem = sanitize_tensor_name(conv_name);

    Tensor x(shape.input_c, shape.input_h, shape.input_w);
    x.data = read_f32_file(
        "artifacts/" + stem + "_input_fp32.bin",
        static_cast<size_t>(shape.input_c * shape.input_h * shape.input_w)
    );

    const std::vector<float> golden = read_f32_file(
        "artifacts/" + stem + "_output_fp32.bin",
        static_cast<size_t>(conv.out_channels * shape.output_h * shape.output_w)
    );

    QuantTensorI8 xq_raw = quantize_tensor_symmetric_i8(x);
    QuantTensorI8 xq = quantize_and_pad_for_q8_s2_benchmark(x, conv);

    Tensor y_q8;

    for (int i = 0; i < warmup; ++i) {
        if (apply_relu) {
            y_q8 = conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
                xq,
                shape.output_h,
                shape.output_w,
                conv,
                plan,
                bn,
                workspace
            );
        } else {
            y_q8 = conv3x3_q8_ocblock4_s2_bn_avx2_out_hw(
                xq,
                shape.output_h,
                shape.output_w,
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
            y_q8 = conv3x3_q8_ocblock4_s2_bn_relu_avx2_out_hw(
                xq,
                shape.output_h,
                shape.output_w,
                conv,
                plan,
                bn,
                workspace
            );
        } else {
            y_q8 = conv3x3_q8_ocblock4_s2_bn_avx2_out_hw(
                xq,
                shape.output_h,
                shape.output_w,
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

    std::cout << "\nQ8 " << conv_name << " stride-2 vs FP32 golden\n";
    std::cout << "  input_shape:              "
              << shape.input_c << "x"
              << shape.input_h << "x"
              << shape.input_w << "\n";
    std::cout << "  output_shape:             "
              << conv.out_channels << "x"
              << shape.output_h << "x"
              << shape.output_w << "\n";
    std::cout << "  padded_q8_shape:          "
              << xq.c << "x"
              << xq.h << "x"
              << xq.w << "\n";
    std::cout << "  input_scale:              " << xq_raw.scale << "\n";
    std::cout << "  max_abs_error:            " << max_abs << "\n";
    std::cout << "  mean_abs_error:           " << mean_abs << "\n";
    std::cout << "  mean_abs_golden:          " << mean_abs_golden << "\n";
    std::cout << "  relative_mean_abs_error:  "
              << (mean_abs_golden > 0.0 ? mean_abs / mean_abs_golden : 0.0)
              << "\n\n";

    const LatencyStats q8_stats = compute_latency_stats(q8_times);

    print_stats(("Q8 " + conv_name + " stride-2 kernel only").c_str(), q8_stats);
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

    benchmark_q8_stride2_conv(
        model,
        "layer3.0.conv1",
        "layer3.0.bn1",
        true,
        runs,
        warmup
    );

    return 0;
}