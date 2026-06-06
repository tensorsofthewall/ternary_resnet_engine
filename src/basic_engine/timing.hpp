#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

struct TimingEntry {
    std::array<char, 64> name{};
    double ms = 0.0;
};

inline TimingEntry make_timing_entry(const char* name, double ms) {
    TimingEntry e;
    std::snprintf(e.name.data(), e.name.size(), "%s", name);
    e.ms = ms;
    return e;
}

template <typename Fn>
auto timed_op(
    std::vector<TimingEntry>& timings,
    const char* name,
    Fn&& fn
) {
    const auto start = std::chrono::high_resolution_clock::now();

    auto result = fn();

    const auto end = std::chrono::high_resolution_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    timings.push_back(make_timing_entry(name, ms));

    return result;
}
