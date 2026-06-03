#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

// For timing end-to-end latency
#include <algorithm>
#include <chrono>
#include <numeric>


#include "model_loader.hpp"
#include "resnet18_engine.hpp"
#include "tensor.hpp"


struct LatencyStats {
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double p90_ms = 0.0;
};

LatencyStats compute_latency_stats(std::vector<double> times_ms) {
    if (times_ms.empty()) {
        throw std::runtime_error("Cannot compute latency stats from empty vector");
    }

    std::sort(times_ms.begin(), times_ms.end());

    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);

    LatencyStats stats;
    stats.min_ms = times_ms.front();
    stats.median_ms = times_ms[times_ms.size() / 2];
    stats.mean_ms = sum / static_cast<double>(times_ms.size());

    const size_t p90_index = static_cast<size_t>(
        0.9 * static_cast<double>(times_ms.size() - 1)
    );
    stats.p90_ms = times_ms[p90_index];

    return stats;
}

std::vector<float> read_f32_file(const std::filesystem::path& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Could not open file: " + path.string());
    }

    std::vector<float> data(expected_count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_count * sizeof(float)));

    if (!f) {
        throw std::runtime_error("Failed to read expected float count from: " + path.string());
    }

    return data;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ternary_resnet <model.bin> <input_f32.bin> <golden_logits_f32.bin>\n";
        return 1;
    }

    const auto model_path = std::filesystem::path(argv[1]);
    const auto input_path = std::filesystem::path(argv[2]);
    const auto logits_path = std::filesystem::path(argv[3]);

    const Model model = load_model(model_path);

    Tensor input(3, 224, 224);
    input.data = read_f32_file(input_path, 3 * 224 * 224);

    constexpr int warmup_runs = 2;
    constexpr int timed_runs = 5;

    std::cout << "Running scalar warmup...\n";

    for (int i = 0; i < warmup_runs; ++i) {
        Tensor warmup_output = forward_resnet18_scalar(model, input);

        if (warmup_output.data.empty()) {
            throw std::runtime_error("Warmup output is empty");
        }
    }

    std::cout << "Benchmarking scalar engine...\n";

    std::vector<double> times_ms;
    times_ms.reserve(timed_runs);

    Tensor output;

    for (int i = 0; i < timed_runs; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();

        output = forward_resnet18_scalar(model, input);

        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        times_ms.push_back(elapsed_ms);
    }

    const LatencyStats stats = compute_latency_stats(times_ms);

    std::cout << "C++ scalar latency\n";
    std::cout << "  runs:      " << timed_runs << "\n";
    std::cout << "  min_ms:    " << stats.min_ms << "\n";
    std::cout << "  median_ms: " << stats.median_ms << "\n";
    std::cout << "  mean_ms:   " << stats.mean_ms << "\n";
    std::cout << "  p90_ms:    " << stats.p90_ms << "\n";

    const std::vector<float> golden = read_f32_file(logits_path, 1000);

    float max_abs = 0.0f;
    double mean_abs = 0.0;

    for (int i = 0; i < 1000; ++i) {
        const float err = std::abs(output.data[static_cast<size_t>(i)] - golden[static_cast<size_t>(i)]);
        max_abs = std::max(max_abs, err);
        mean_abs += static_cast<double>(err);
    }

    mean_abs /= 1000.0;

    std::cout << "C++ scalar vs PyTorch golden\n";
    std::cout << "  max_abs_error:  " << max_abs << "\n";
    std::cout << "  mean_abs_error: " << mean_abs << "\n";

    std::cout << "Sample logits:\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "  y[" << i << "] cpp=" << output.data[static_cast<size_t>(i)]
                  << " golden=" << golden[static_cast<size_t>(i)] << "\n";
    }

    return 0;
}