#pragma once

#include "model.hpp"

#include <cstdint>
#include <vector>

namespace q2q {

// =====================================================================
// Dense panel + PanelStrides -- generic, OC_TILE=4 / K_CHUNK=32 layout
// shared by the baseline and apanel-optimized dense conv kernels.
// =====================================================================
std::vector<int8_t> build_oc4_k32_panel(const Conv2DParams& conv);

struct PanelStrides {
    size_t st_icc, st_kx, st_ky, st_oct;
    static PanelStrides make(int IC, int KH, int KW);
};

// =====================================================================
// conv1 im2col panel -- panel flattening reordered from
// k=ic*49+ky*7+kx to k=(ky*7+kx)*IC+ic so each (ky,kx) position's IC=3
// source channels are contiguous in NHWC, enabling a 4-byte gather.
// =====================================================================
struct Conv1Panel {
    int K_PAD = 0, IC_CHUNKS = 0, OC_TILES = 0;
    std::vector<int8_t> data;
};

Conv1Panel build_conv1_im2col_panel(const Conv2DParams& conv);

} // namespace q2q
