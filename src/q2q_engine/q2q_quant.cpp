#include "q2q_quant.hpp"

#include "q2q_common.hpp"

#include <cmath>

namespace q2q {

void quantize_nchw_fp32_to_q8_nhwc(const float* src, int H, int W, int C, QTensorNHWC& dst) {
    const float inv_scale = 1.0f / ACT_SCALE;
    const int b = dst.border;
    for (int y = 0; y < H; ++y) {
        uint8_t* row = dst.data.data() + ((size_t)(y + b) * dst.W + b) * C;
        for (int x = 0; x < W; ++x) {
            uint8_t* px = row + (size_t)x * C;
            for (int c = 0; c < C; ++c) {
                const float v = src[((size_t)c * H + y) * W + x];
                int32_t q = static_cast<int32_t>(std::lround(v * inv_scale));
                q = q < 0 ? 0 : (q > 255 ? 255 : q);
                px[c] = static_cast<uint8_t>(q);
            }
        }
    }
}

void dequantize_q8_nhwc_to_fp32_nchw(const QTensorNHWC& src, int H, int W, int C, Tensor& dst) {
    const int b = src.border;
    for (int y = 0; y < H; ++y) {
        const uint8_t* row = src.data.data() + ((size_t)(y + b) * src.W + b) * C;
        for (int x = 0; x < W; ++x) {
            const uint8_t* px = row + (size_t)x * C;
            for (int c = 0; c < C; ++c)
                dst.data[((size_t)c * H + y) * W + x] = static_cast<float>(px[c]) * src.scale;
        }
    }
}

} // namespace q2q
