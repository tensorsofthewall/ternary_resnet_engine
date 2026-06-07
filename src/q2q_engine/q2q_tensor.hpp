#pragma once

#include <cstdint>
#include <vector>

namespace q2q {

struct QTensorNHWC {
    int H = 0, W = 0, C = 0;
    int border = 0;
    float scale = 1.0f;
    std::vector<uint8_t> data;
};

QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, int border, float scale);

} // namespace q2q
