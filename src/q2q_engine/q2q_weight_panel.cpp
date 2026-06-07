#include "q2q_weight_panel.hpp"

#include "q2q_common.hpp"

namespace q2q {

std::vector<int8_t> build_oc4_k32_panel(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const size_t base   = (size_t)IC * KH * KW;

    std::vector<int8_t> panel(
        (size_t)OC_TILES * KH * KW * IC_CHUNKS * OC_TILE * K_CHUNK, int8_t(0));

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile)
        for (int ky = 0; ky < KH; ++ky)
            for (int kx = 0; kx < KW; ++kx)
                for (int icc = 0; icc < IC_CHUNKS; ++icc)
                    for (int j = 0; j < OC_TILE; ++j) {
                        const int oc = oc_tile * OC_TILE + j;
                        for (int k = 0; k < K_CHUNK; ++k) {
                            const int ic = icc * K_CHUNK + k;
                            const int8_t w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                            const size_t idx =
                                ((((((size_t)oc_tile * KH + ky) * KW + kx) * IC_CHUNKS + icc)
                                  * OC_TILE + j) * K_CHUNK) + k;
                            panel[idx] = w;
                        }
                    }
    return panel;
}

PanelStrides PanelStrides::make(int IC, int KH, int KW) {
    const int IC_CHUNKS = IC / K_CHUNK;
    PanelStrides s;
    s.st_icc = (size_t)OC_TILE * K_CHUNK;
    s.st_kx  = (size_t)IC_CHUNKS * s.st_icc;
    s.st_ky  = (size_t)KW * s.st_kx;
    s.st_oct = (size_t)KH * s.st_ky;
    return s;
}

Conv1Panel build_conv1_im2col_panel(const Conv2DParams& conv) {
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int K_FULL = IC * KH * KW;
    const int IC_CHUNKS = (K_FULL + K_CHUNK - 1) / K_CHUNK;
    const int K_PAD = IC_CHUNKS * K_CHUNK;
    const int OC_TILES = OC / OC_TILE;
    const size_t base = (size_t)IC * KH * KW;

    Conv1Panel panel;
    panel.K_PAD = K_PAD; panel.IC_CHUNKS = IC_CHUNKS; panel.OC_TILES = OC_TILES;
    panel.data.assign((size_t)OC_TILES * IC_CHUNKS * OC_TILE * K_CHUNK, int8_t(0));

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile)
        for (int icc = 0; icc < IC_CHUNKS; ++icc)
            for (int j = 0; j < OC_TILE; ++j) {
                const int oc = oc_tile * OC_TILE + j;
                for (int k = 0; k < K_CHUNK; ++k) {
                    const int kg = icc * K_CHUNK + k;
                    int8_t w = 0;
                    if (kg < K_FULL) {
                        const int ky = kg / (KW * IC);
                        const int rem = kg % (KW * IC);
                        const int kx = rem / IC;
                        const int ic = rem % IC;
                        w = conv.weight[oc * base + (size_t)ic * KH * KW + ky * KW + kx];
                    }
                    const size_t idx = (((size_t)oc_tile * IC_CHUNKS + icc) * OC_TILE + j) * K_CHUNK + k;
                    panel.data[idx] = w;
                }
            }
    return panel;
}

} // namespace q2q
