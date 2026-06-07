#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

// For timing end-to-end latency
#include <algorithm>
#include <chrono>
#include <numeric>

#ifdef ENABLE_ALLOC_PROFILE
#include <atomic>
static std::atomic<size_t> g_alloc_count{0};
static std::atomic<size_t> g_alloc_bytes{0};
void* operator new(size_t n)                      { g_alloc_count.fetch_add(1, std::memory_order_relaxed); g_alloc_bytes.fetch_add(n, std::memory_order_relaxed); return std::malloc(n); }
void* operator new[](size_t n)                    { g_alloc_count.fetch_add(1, std::memory_order_relaxed); g_alloc_bytes.fetch_add(n, std::memory_order_relaxed); return std::malloc(n); }
void  operator delete(void* p) noexcept           { std::free(p); }
void  operator delete(void* p, size_t) noexcept   { std::free(p); }
void  operator delete[](void* p) noexcept         { std::free(p); }
void  operator delete[](void* p, size_t) noexcept { std::free(p); }
#endif


#include "model_loader.hpp"
#include "resnet18_engine.hpp"
#include "tensor.hpp"
#include "workspace.hpp"


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
    if (argc < 3) {
        std::cerr << "Usage: ternary_resnet <model.bin> <input_f32.bin>\n";
        return 1;
    }

    const auto model_path = std::filesystem::path(argv[1]);
    const auto input_path = std::filesystem::path(argv[2]);

    const Model model = load_model(model_path);

    Tensor input(3, 224, 224);
    input.data = read_f32_file(input_path, 3 * 224 * 224);

    constexpr int warmup_runs = 20;
    constexpr int timed_runs = 100;

    EngineWorkspace workspace;

    std::cout << "Running engine warmup...\n";

    for (int i = 0; i < warmup_runs; ++i) {
        Tensor warmup_output = forward_resnet18_scalar(model, input, workspace);

        if (warmup_output.data.empty()) {
            throw std::runtime_error("Warmup output is empty");
        }
    }

#ifdef ENABLE_ALLOC_PROFILE
    {
        g_alloc_count.store(0); g_alloc_bytes.store(0);
        Tensor alloc_out = forward_resnet18_scalar(model, input, workspace);
        (void)alloc_out;
        std::cout << "Alloc profile: forward_resnet18_scalar\n";
        std::cout << "  new_calls:       " << g_alloc_count.load() << "\n";
        std::cout << "  bytes_allocated: " << g_alloc_bytes.load() << "\n";

        std::vector<TimingEntry> alloc_timings;
        alloc_timings.reserve(64);
        g_alloc_count.store(0); g_alloc_bytes.store(0);
        Tensor alloc_timed_out = forward_resnet18_workspace_timed(model, input, workspace, alloc_timings);
        (void)alloc_timed_out;
        std::cout << "Alloc profile: forward_resnet18_workspace_timed\n";
        std::cout << "  new_calls:       " << g_alloc_count.load() << "\n";
        std::cout << "  bytes_allocated: " << g_alloc_bytes.load() << "\n";
    }
#endif

    std::cout << "Benchmarking C++ ternary engine...\n";

    std::vector<double> times_ms;
    times_ms.reserve(timed_runs);

    Tensor output;

    for (int i = 0; i < timed_runs; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();

        output = forward_resnet18_scalar(model, input, workspace);

        const auto end = std::chrono::high_resolution_clock::now();

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        times_ms.push_back(elapsed_ms);
    }

    const LatencyStats stats = compute_latency_stats(times_ms);

    std::cout << "C++ engine latency\n";
    std::cout << "  runs:      " << timed_runs << "\n";
    std::cout << "  min_ms:    " << stats.min_ms << "\n";
    std::cout << "  median_ms: " << stats.median_ms << "\n";
    std::cout << "  mean_ms:   " << stats.mean_ms << "\n";
    std::cout << "  p90_ms:    " << stats.p90_ms << "\n";

    std::vector<TimingEntry> timings;
    forward_resnet18_workspace_timed(model, input, workspace, timings);

    double total_profiled_ms = 0.0;

    std::cout << "\nC++ engine per-block profile\n";
    for (const TimingEntry& entry : timings) {
        total_profiled_ms += entry.ms;
        std::cout << "  " << entry.name.data() << ": " << entry.ms << " ms\n";
    }

    std::vector<TimingEntry> sorted_timings = timings;

    std::sort(
        sorted_timings.begin(),
        sorted_timings.end(),
        [](const TimingEntry& a, const TimingEntry& b) {
            return a.ms > b.ms;
        }
    );

    std::cout << "\nTop timed operations\n";
    for (size_t i = 0; i < std::min<size_t>(10, sorted_timings.size()); ++i) {
        std::cout
            << "  "
            << sorted_timings[i].name.data()
            << ": "
            << sorted_timings[i].ms
            << " ms\n";
    }

    std::cout << "  total_profiled_ms: " << total_profiled_ms << " ms\n";

    return 0;
}