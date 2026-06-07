#pragma once

#include <cstdint>
#include <vector>

namespace q2q {

// A quantized activation tensor in NHWC layout: uint8 values plus a single
// per-tensor dequantization scale (real_value = data * scale). H/W/C are the
// PADDED extents (logical_extent + 2*border); `border` is the halo width
// baked into the storage so that conv kernels can read out-of-bounds zeros
// without bounds checks. Indexing is row-major: data[((h*W + w)*C) + c].
struct QTensorNHWC {
    int H = 0, W = 0, C = 0;
    int border = 0;
    float scale = 1.0f;
    std::vector<uint8_t> data;
};

// Allocates a zero-filled QTensorNHWC sized for a logical H_logical x W_logical x C
// image plus a `border`-wide zero halo on every side (H = H_logical + 2*border, etc.).
QTensorNHWC make_q8_tensor(int H_logical, int W_logical, int C, int border, float scale);

} // namespace q2q
