#include "q2q_pool.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>

namespace q2q {

void q8_maxpool_3x3_s2_p1(const QTensorNHWC& x, int OH, int OW, QTensorNHWC& y) {
    constexpr int stride = 2, pad = 1;
    const int C = x.C;
    const int read_offset = x.border - pad;
    const uint8_t* base_in  = x.data.data();
    uint8_t*       base_out = y.data.data();

    // Phase 12: parallelize over output rows -- each row reads an overlapping
    // (read-only, safe) 3-row band of the input and writes a disjoint output
    // row, so there are no races.
    #pragma omp parallel for schedule(static)
    for (int oh = 0; oh < OH; ++oh) {
        uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * C;
        for (int ow = 0; ow < OW; ++ow) {
            uint8_t* out_px = out_row + (size_t)ow * C;
            const uint8_t* p00 = base_in + ((size_t)(oh * stride + 0 + read_offset) * x.W + (ow * stride + 0 + read_offset)) * C;
            const uint8_t* p01 = p00 + (size_t)1 * C;
            const uint8_t* p02 = p00 + (size_t)2 * C;
            const uint8_t* p10 = p00 + (size_t)x.W * C;
            const uint8_t* p11 = p10 + (size_t)1 * C;
            const uint8_t* p12 = p10 + (size_t)2 * C;
            const uint8_t* p20 = p10 + (size_t)x.W * C;
            const uint8_t* p21 = p20 + (size_t)1 * C;
            const uint8_t* p22 = p20 + (size_t)2 * C;
            for (int c = 0; c < C; ++c) {
                uint8_t m = p00[c];
                m = std::max(m, p01[c]); m = std::max(m, p02[c]);
                m = std::max(m, p10[c]); m = std::max(m, p11[c]); m = std::max(m, p12[c]);
                m = std::max(m, p20[c]); m = std::max(m, p21[c]); m = std::max(m, p22[c]);
                out_px[c] = m;
            }
        }
    }
}

} // namespace q2q
