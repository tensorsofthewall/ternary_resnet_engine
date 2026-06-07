// Blocked engine: NCHWc16 activations + mullo_epi32 ternary conv prototype.
// Full ResNet18 forward pass, with the hot conv3x3 paths on AVX2 mullo kernels
// and the remaining ops on reference scalar kernels.
//
// Usage: ./blocked_engine <model.bin> <input_fp32.bin>
//
// Note: the reference (scalar) kernels are unoptimized -- they establish
// correctness and end-to-end timing for the scalar baseline, while the AVX2
// mullo_epi32 kernel on conv3x3 brings total_ms close to basic_engine.

#include "blocked_model.hpp"
#include "blocked_resnet18.hpp"
#include "model_loader.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static std::vector<float> read_f32_file(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Short read: " + path);
    return data;
}

static double elapsed_ms(
    std::chrono::time_point<std::chrono::high_resolution_clock> t0,
    std::chrono::time_point<std::chrono::high_resolution_clock> t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> <input_fp32.bin>\n";
        return 1;
    }

    constexpr int N_WARMUP = 20;
    constexpr int N_TIMED  = 100;

    try {
        std::cout << "Loading model...\n";
        const Model model = load_model(argv[1]);
        std::cout << "Building blocked model...\n";
        const BlockedModel bm = build_blocked_model(model);
        std::cout << "Blocked convs built: " << bm.blocked_convs.size() << "\n";

        // Load input: 3×224×224 flat FP32.
        constexpr size_t INPUT_ELEMS = 3 * 224 * 224;
        const auto raw = read_f32_file(argv[2], INPUT_ELEMS);
        Tensor input(3, 224, 224);
        input.data = raw;

        BlockedEngineWorkspace ws;
        std::unordered_map<std::string, double> timings;

        // Warmup.
        std::cout << "Warmup (" << N_WARMUP << " runs)...\n" << std::flush;
        for (int i = 0; i < N_WARMUP; ++i)
            forward_blocked_resnet18(bm, input, ws, timings);

        // Timed runs.
        timings.clear();
        std::cout << "Timing (" << N_TIMED << " runs)...\n" << std::flush;
        std::vector<double> times;
        times.reserve(N_TIMED);

        std::vector<float> probs;
        for (int i = 0; i < N_TIMED; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            probs = forward_blocked_resnet18(bm, input, ws, timings);
            const auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(elapsed_ms(t0, t1));
        }

        const int top1 = static_cast<int>(
            std::max_element(probs.begin(), probs.end()) - probs.begin());

        std::sort(times.begin(), times.end());
        const double min_ms    = times.front();
        const double median_ms = times[times.size() / 2];

        std::cout << "\n=== blocked_engine (AVX2 sign_epi32 mullo) ===\n";
        std::cout << "  top-1 class: " << top1
                  << "  (p=" << probs[static_cast<size_t>(top1)] << ")\n";
        std::cout << "  min_ms:    " << min_ms    << "\n";
        std::cout << "  median_ms: " << median_ms << "\n";
        std::cout << "  scalar ref baseline: ~397 ms\n";
        std::cout << "  bucket baseline (layer2.1.conv2, basic_engine): 2.67 ms\n";

        // Per-layer timing (averaged over N_TIMED runs), sorted by descending cost.
        std::cout << "\n--- per-layer avg_ms (over " << N_TIMED << " runs) ---\n";
        std::vector<std::pair<double, std::string>> layer_times;
        layer_times.reserve(timings.size());
        for (const auto& [name, total] : timings)
            layer_times.emplace_back(total / N_TIMED, name);
        std::sort(layer_times.rbegin(), layer_times.rend());
        for (const auto& [t, name] : layer_times)
            std::cout << "  " << name << ": " << t << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
