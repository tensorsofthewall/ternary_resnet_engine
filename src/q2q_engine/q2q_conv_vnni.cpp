#include "q2q_conv_vnni.hpp"

#include "q2q_epilogue.hpp"
#include "q2q_quant.hpp"

#include <omp.h>

#include <vector>

namespace q2q {

void q2q_conv_bn_requant(
    const QTensorNHWC& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    bool relu, float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const int stride = conv.stride_h;
    const int read_offset = x.border - conv.pad_h;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const PanelStrides st = PanelStrides::make(IC, KH, KW);

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data();
    uint8_t*       base_out   = y.data.data();

    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        const int oc_base = oc_tile * OC_TILE;
        float s[OC_TILE], b[OC_TILE];
        for (int j = 0; j < OC_TILE; ++j) {
            s[j] = bn.scale[oc_base + j] * dq;
            b[j] = bn.bias[oc_base + j];
        }
        const int8_t* panel_tile = base_panel + (size_t)oc_tile * st.st_oct;

        for (int oh = 0; oh < OH; ++oh) {
            uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C + oc_base;
            for (int ow = 0; ow < OW; ++ow) {
                int32_t d0, d1, d2, d3;
                vnni_oc4_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW,
                              oh, ow, stride, read_offset, d0, d1, d2, d3);

                float v0 = static_cast<float>(d0) * s[0] + b[0];
                float v1 = static_cast<float>(d1) * s[1] + b[1];
                float v2 = static_cast<float>(d2) * s[2] + b[2];
                float v3 = static_cast<float>(d3) * s[3] + b[3];
                if (relu) {
                    v0 = v0 < 0.f ? 0.f : v0; v1 = v1 < 0.f ? 0.f : v1;
                    v2 = v2 < 0.f ? 0.f : v2; v3 = v3 < 0.f ? 0.f : v3;
                }

                uint8_t* out_px = out_row + (size_t)ow * y.C;
                out_px[0] = requantize_u8(v0, inv_out_scale);
                out_px[1] = requantize_u8(v1, inv_out_scale);
                out_px[2] = requantize_u8(v2, inv_out_scale);
                out_px[3] = requantize_u8(v3, inv_out_scale);
            }
        }
    }
}

// =====================================================================
// OPTIMIZED dense conv -- Phase 10 items 3/5, byte-for-byte from
// benchmark_q2q_microkernel_opt.cpp (validated there: ~2.08x / ~17x
// paired wins). border-generic (q2q_conv_bn_requant_apanel uses x.border /
// y.border instead of a fixed PAD, matching the baseline's generalization;
// every tensor it touches here has border=PAD_Q8=1, so this is a no-op).
// epilogue_oc4_simd is shared with conv1.
// =====================================================================
void q2q_conv_bn_requant_apanel(
    const QTensorNHWC& x, const std::vector<int8_t>& panel,
    const Conv2DParams& conv, const BNFusedParams& bn,
    bool relu, float out_scale, QTensorNHWC& y, int OH, int OW)
{
    const int IC = conv.in_channels, OC = conv.out_channels;
    const int KH = conv.kh, KW = conv.kw;
    const int IC_CHUNKS = IC / K_CHUNK;
    const int OC_TILES  = OC / OC_TILE;
    const int stride = conv.stride_h;
    const int read_offset = x.border - conv.pad_h;
    const float dq = x.scale;
    const float inv_out_scale = 1.0f / out_scale;
    const PanelStrides st = PanelStrides::make(IC, KH, KW);
    const bool use_n2 = (stride == 1) && (OW % 2 == 0);

    const uint8_t* base_in    = x.data.data();
    const int8_t*  base_panel = panel.data();
    uint8_t*       base_out   = y.data.data();

    // Precompute per-OC scale/bias once -- tiny (OC floats each), read-only,
    // shared across threads. This keeps the (oc_tile, oh) loop nest perfectly
    // collapsible (OpenMP collapse() requires no code between the collapsed
    // loop heads).
    std::vector<float> s_all((size_t)OC), b_all((size_t)OC);
    for (int oc = 0; oc < OC; ++oc) {
        s_all[oc] = bn.scale[oc] * dq;
        b_all[oc] = bn.bias[oc];
    }

    // Phase 12: parallelize over the flattened (oc_tile, oh) task space via
    // collapse(2). Each task writes a disjoint OC_TILE-channel x one-output-row
    // region (oc_base..oc_base+3, row oh) -- no overlap, no races. Flattening
    // both dimensions (rather than just oh) gives every thread count plenty of
    // independent units even for layer4's small spatial extent (OH=7,
    // OC_TILES=128 -> 896 tasks vs. just 7 if parallelizing over oh alone).
    #pragma omp parallel for collapse(2) schedule(static)
    for (int oc_tile = 0; oc_tile < OC_TILES; ++oc_tile) {
        for (int oh = 0; oh < OH; ++oh) {
            const int oc_base = oc_tile * OC_TILE;
            const float* s = &s_all[oc_base];
            const float* b = &b_all[oc_base];
            const int8_t* panel_tile = base_panel + (size_t)oc_tile * st.st_oct;
            uint8_t* out_row = base_out + ((size_t)(oh + y.border) * y.W + y.border) * y.C + oc_base;
            if (use_n2) {
                for (int ow = 0; ow < OW; ow += 2) {
                    int32_t d[2][OC_TILE];
                    vnni_oc4_n2_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW,
                                     oh, ow, read_offset, d);
                    epilogue_oc4_simd(d[0][0], d[0][1], d[0][2], d[0][3], s, b, relu, inv_out_scale,
                                      out_row + (size_t)ow * y.C);
                    epilogue_oc4_simd(d[1][0], d[1][1], d[1][2], d[1][3], s, b, relu, inv_out_scale,
                                      out_row + (size_t)(ow + 1) * y.C);
                }
            } else {
                for (int ow = 0; ow < OW; ++ow) {
                    int32_t d0, d1, d2, d3;
                    vnni_oc4_dots(base_in, x.W, x.C, panel_tile, st, IC_CHUNKS, KH, KW,
                                  oh, ow, stride, read_offset, d0, d1, d2, d3);
                    epilogue_oc4_simd(d0, d1, d2, d3, s, b, relu, inv_out_scale,
                                      out_row + (size_t)ow * y.C);
                }
            }
        }
    }
}

} // namespace q2q
