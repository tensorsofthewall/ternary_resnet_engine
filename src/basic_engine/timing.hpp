#pragma once


#include <chrono>
#include <string>
#include <utility>
#include <vector>

struct TimingEntry {
    std::string name;
    double ms = 0.0;
};

template <typename Fn>
auto timed_op(
    std::vector<TimingEntry>& timings,
    const std::string& name,
    Fn&& fn
) {
    const auto start = std::chrono::high_resolution_clock::now();

    auto result = fn();

    const auto end = std::chrono::high_resolution_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    timings.push_back(TimingEntry{name, ms});

    return result;
}