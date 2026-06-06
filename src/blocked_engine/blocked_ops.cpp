#include "blocked_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void blocked_maxpool_3x3_s2(const BlockedTensorFP32& input, BlockedTensorFP32& output) {
    // pad=1, stride=2, kernel=3×3
    const int OH = (input.H + 2 * 1 - 3) / 2 + 1;
    const int OW = (input.W + 2 * 1 - 3) / 2 + 1;
    output.reuse(input.CB, OH, OW);

    for (int cb = 0; cb < input.CB; ++cb) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float maxv[16];
                std::fill(maxv, maxv + 16, -std::numeric_limits<float>::infinity());

                for (int ky = 0; ky < 3; ++ky) {
                    const int ih = oh * 2 + ky - 1;
                    if (ih < 0 || ih >= input.H) continue;
                    for (int kx = 0; kx < 3; ++kx) {
                        const int iw = ow * 2 + kx - 1;
                        if (iw < 0 || iw >= input.W) continue;
                        const float* pix = input.pixel(cb, ih, iw);
                        for (int ci = 0; ci < 16; ++ci)
                            if (pix[ci] > maxv[ci]) maxv[ci] = pix[ci];
                    }
                }

                float* out = output.pixel(cb, oh, ow);
                for (int ci = 0; ci < 16; ++ci) out[ci] = maxv[ci];
            }
        }
    }
}

void blocked_avgpool_global(const BlockedTensorFP32& input, BlockedTensorFP32& output) {
    output.reuse(input.CB, 1, 1);
    const float inv_n = 1.0f / static_cast<float>(input.H * input.W);

    for (int cb = 0; cb < input.CB; ++cb) {
        float sum[16] = {};
        for (int h = 0; h < input.H; ++h) {
            for (int w = 0; w < input.W; ++w) {
                const float* pix = input.pixel(cb, h, w);
                for (int ci = 0; ci < 16; ++ci) sum[ci] += pix[ci];
            }
        }
        float* out = output.pixel(cb, 0, 0);
        for (int ci = 0; ci < 16; ++ci) out[ci] = sum[ci] * inv_n;
    }
}

void blocked_fc(const BlockedTensorFP32& input, const LinearParams& fc,
                std::vector<float>& logits) {
    // Input is (CB, 1, 1) after global avgpool — C = CB*16 flat feature values.
    const int C = input.CB * 16;
    if (C != fc.in_features)
        throw std::runtime_error("blocked_fc: input C mismatch");

    logits.assign(static_cast<size_t>(fc.out_features), 0.0f);

    // Gather flat feature vector from NCHWc16 (CB, 1, 1) layout.
    std::vector<float> feat(static_cast<size_t>(C));
    for (int cb = 0; cb < input.CB; ++cb) {
        const float* pix = input.pixel(cb, 0, 0);
        for (int ci = 0; ci < 16; ++ci)
            feat[static_cast<size_t>(cb * 16 + ci)] = pix[ci];
    }

    // y = W * x + b   (W: [out_features × in_features] row-major)
    for (int o = 0; o < fc.out_features; ++o) {
        float acc = fc.has_bias ? fc.bias[static_cast<size_t>(o)] : 0.0f;
        const float* row = fc.weight.data() + static_cast<size_t>(o * fc.in_features);
        for (int i = 0; i < fc.in_features; ++i)
            acc += row[i] * feat[static_cast<size_t>(i)];
        logits[static_cast<size_t>(o)] = acc;
    }
}

void blocked_residual_add_relu(BlockedTensorFP32& x, const BlockedTensorFP32& skip) {
    for (size_t i = 0; i < x.data.size(); ++i) {
        float v = x.data[i] + skip.data[i];
        x.data[i] = v > 0.0f ? v : 0.0f;
    }
}

void blocked_residual_add(BlockedTensorFP32& x, const BlockedTensorFP32& skip) {
    for (size_t i = 0; i < x.data.size(); ++i)
        x.data[i] += skip.data[i];
}

void softmax_inplace(std::vector<float>& x) {
    if (x.empty()) return;
    const float max_v = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& v : x) { v = std::exp(v - max_v); sum += v; }
    const float inv_sum = 1.0f / sum;
    for (float& v : x) v *= inv_sum;
}
