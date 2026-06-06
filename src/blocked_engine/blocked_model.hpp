#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>
#include "model.hpp"

// 16×16 column-major weight tiles for an entire conv layer.
//
// Storage layout (one entry per (oc_cb, ic_cb, ky, kx) tile):
//   data[oc_cb * (ic_blocks * kh * kw * 256)
//       + ic_cb * (kh * kw * 256)
//       + ky    * (kw * 256)
//       + kx    * 256
//       + ci_in * 16 + ci_out]
//
// Each 256-byte tile is a 16×16 matrix stored column-major:
//   tile[ci_in * 16 + ci_out]  →  weight[oc_cb*16 + ci_out][ic_cb*16 + ci_in][ky][kx]
//
// Requires IC % 16 == 0 and OC % 16 == 0.

struct BlockedConvWeights {
    int oc_blocks = 0;  // OC / 16
    int ic_blocks = 0;  // IC / 16
    int kh        = 0;
    int kw        = 0;
    std::vector<int8_t> data;

    const int8_t* block(int oc_cb, int ic_cb, int ky, int kx) const {
        const size_t idx =
            (static_cast<size_t>(oc_cb * ic_blocks * kh * kw)
           + static_cast<size_t>(ic_cb * kh * kw)
           + static_cast<size_t>(ky * kw)
           + static_cast<size_t>(kx)) * 256;
        return data.data() + idx;
    }

    int8_t* block(int oc_cb, int ic_cb, int ky, int kx) {
        const size_t idx =
            (static_cast<size_t>(oc_cb * ic_blocks * kh * kw)
           + static_cast<size_t>(ic_cb * kh * kw)
           + static_cast<size_t>(ky * kw)
           + static_cast<size_t>(kx)) * 256;
        return data.data() + idx;
    }
};

BlockedConvWeights build_blocked_conv_weights(const Conv2DParams& conv);
