#pragma once

#include "model.hpp"

#include <cstdint>
#include <vector>

namespace q2q {

// =====================================================================
// Dense panel + PanelStrides -- generic, OC_TILE=4 / K_CHUNK=32 layout
// shared by the baseline and apanel-optimized dense conv kernels.
// =====================================================================
// Repacks `conv.weight` (OC x IC x KH x KW, ternary int8) into the
// [oc_tile][ky][kx][icc][j in OC_TILE][k in K_CHUNK] layout the VNNI dense
// kernels expect: each (ky,kx,icc) slot holds OC_TILE contiguous output
// channels x K_CHUNK=32 contiguous reduction elements, ready to load
// straight into a dpbusd operand. Requires IC a multiple of K_CHUNK and OC a
// multiple of OC_TILE (true for every dense q8 layer in this engine).
std::vector<int8_t> build_oc4_k32_panel(const Conv2DParams& conv);

// Byte strides into a build_oc4_k32_panel-packed buffer for stepping by one
// reduction chunk (st_icc), kernel column (st_kx), kernel row (st_ky), or
// output-channel tile (st_oct). `make` derives all four from (IC, KH, KW).
struct PanelStrides {
    size_t st_icc, st_kx, st_ky, st_oct;
    static PanelStrides make(int IC, int KH, int KW);
};

// =====================================================================
// conv1 im2col panel -- panel flattening reordered from
// k=ic*49+ky*7+kx to k=(ky*7+kx)*IC+ic so each (ky,kx) position's IC=3
// source channels are contiguous in NHWC, enabling a 4-byte gather.
// =====================================================================
// Packed conv1 weight panel in [oc_tile][icc][j in OC_TILE][k in K_CHUNK]
// layout, reduction axis zero-padded from K_FULL=IC*KH*KW up to K_PAD (a
// multiple of K_CHUNK=32, spanning IC_CHUNKS chunks) so VNNI loads never run
// past the real weights. OC_TILES = OC / OC_TILE.
struct Conv1Panel {
    int K_PAD = 0, IC_CHUNKS = 0, OC_TILES = 0;
    std::vector<int8_t> data;
};

// Builds a Conv1Panel from `conv.weight`, re-flattening the reduction index
// from k=ic*KH*KW+ky*KW+kx to k=(ky*KW+kx)*IC+ic (matching conv1_gather_fast's
// im2col order) and zero-padding the tail [K_FULL, K_PAD) of each oc_tile/icc
// slot.
Conv1Panel build_conv1_im2col_panel(const Conv2DParams& conv);

} // namespace q2q
