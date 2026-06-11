#include "nhwc_common.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>

namespace nhwc_block32 {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

void print_stats(const char* label, std::vector<double> v, int N) {
    std::sort(v.begin(), v.end());
    const double mn  = v.front();
    const double med = v[N / 2];
    const double p90 = v[N * 9 / 10];
    const double avg = std::accumulate(v.begin(), v.end(), 0.0) / N;
    std::cout << "  " << label
              << ":  min=" << mn << "  med=" << med
              << "  p90=" << p90 << "  avg=" << avg << " ms\n";
}

} // namespace nhwc_block32
