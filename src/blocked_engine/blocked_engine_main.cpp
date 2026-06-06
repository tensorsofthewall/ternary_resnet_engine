// Blocked engine: NCHWc16 activations + mullo_epi32 ternary conv prototype.
// Phase 4: full ResNet18 forward pass with reference scalar kernels.
// Phase 5 will replace the hot conv3x3 paths with AVX2 mullo kernels.
//
// Usage: ./blocked_engine <model.bin> <input_fp32.bin>
//
// Note: the reference (scalar) kernels are unoptimized. Phase 4 establishes
// correctness and end-to-end timing for the scalar baseline. Phase 5 adds
// the AVX2 mullo_epi32 kernel which should bring total_ms close to basic_engine.

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

    // Scalar reference is ~10–20s/pass. Phase 5 (AVX2) will use 20 warmup + 100 timed.
    constexpr int N_WARMUP = 1;
    constexpr int N_TIMED  = 3;

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
        std::cout << "Timing (" << N_TIMED << " runs)...\n" << std::flush;
        std::vector<double> times;
        times.reserve(N_TIMED);

        for (int i = 0; i < N_TIMED; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto probs = forward_blocked_resnet18(bm, input, ws, timings);
            const auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(elapsed_ms(t0, t1));

            // Print top-1 on first timed run.
            if (i == 0) {
                const int top1 = static_cast<int>(
                    std::max_element(probs.begin(), probs.end()) - probs.begin());
                std::cout << "  top-1 class: " << top1
                          << "  (p=" << probs[static_cast<size_t>(top1)] << ")\n";
            }
        }

        std::sort(times.begin(), times.end());
        const double min_ms    = times.front();
        const double median_ms = times[times.size() / 2];

        std::cout << "\n=== blocked_engine (scalar reference) ===\n";
        std::cout << "  min_ms:    " << min_ms    << "\n";
        std::cout << "  median_ms: " << median_ms << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
