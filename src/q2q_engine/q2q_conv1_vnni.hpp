#pragma once

#include "bn_fusion.hpp"
#include "model.hpp"
#include "q2q_tensor.hpp"
#include "q2q_weight_panel.hpp"

#include <cstdint>
#include <cstring>

namespace q2q {

// Gathers im2col columns for one output pixel. For each (ky,kx), the IC=3
// source channels are contiguous in NHWC and the destination slot is
// contiguous in the panel-B layout above -- one 4-byte load + 4-byte store
// per (ky,kx) (49 total) instead of 3 strided scalar byte copies x 49 (147
// total). The 4th (garbage) byte written by iteration i is overwritten by
// iteration i+1's valid byte 0, except the LAST iteration's spill into
// a_panel[K_FULL], which the zero-fill (always run after the gather)
// resets to 0. Reads stay in-bounds: max (iw0+KW-1)*C+4 = 228*3+4 = 688
// < x.W*x.C = 230*3 = 690.
static inline void conv1_gather_fast(
    const uint8_t* base_in, int W, int C, int ih0, int iw0,
    int KH, int KW, int IC, uint8_t* a_panel)
{
    for (int ky = 0; ky < KH; ++ky) {
        const uint8_t* row = base_in + ((size_t)(ih0 + ky) * W + iw0) * C;
        uint8_t* dst_row = a_panel + (size_t)ky * KW * IC;
        for (int kx = 0; kx < KW; ++kx) {
            uint32_t v;
            std::memcpy(&v, row + (size_t)kx * C, sizeof(v));
            std::memcpy(dst_row + (size_t)kx * IC, &v, sizeof(v));
        }
    }
}

// conv1 (7x7/stride=2) as im2col + VNNI dot products, with fused BN and a
// ReLU+requant epilogue (q8 in, q8 out). OpenMP-parallel over output rows;
// each thread gathers its own a_panel via conv1_gather_fast and reuses it
// across every row it owns -- no allocation inside the parallel loop body.
void q2q_conv1_im2col_bn_relu_requant(
    const QTensorNHWC& x, const Conv1Panel& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    float out_scale, QTensorNHWC& y, int OH, int OW);

} // namespace q2q
