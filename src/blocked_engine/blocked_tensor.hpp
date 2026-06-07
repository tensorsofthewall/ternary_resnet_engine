#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

// NCHWc16 activation tensor: CB channel-blocks of 16, H rows, W cols.
// Layout: data[(cb * H * W + h * W + w) * 16 + ci]
// C must be divisible by 16; CB = C / 16.

struct BlockedTensorFP32 {
    int CB = 0;
    int H  = 0;
    int W  = 0;
    std::vector<float> data;

    BlockedTensorFP32() = default;
    BlockedTensorFP32(int CB_, int H_, int W_)
        : CB(CB_), H(H_), W(W_),
          data(static_cast<size_t>(CB_ * H_ * W_) * 16, 0.0f) {}

    static BlockedTensorFP32 alloc(int C, int H, int W) {
        if (C % 16 != 0)
            throw std::runtime_error("BlockedTensorFP32: C must be divisible by 16");
        return BlockedTensorFP32(C / 16, H, W);
    }

    float* pixel(int cb, int h, int w) {
        return data.data() + (static_cast<size_t>(cb * H + h) * static_cast<size_t>(W) + static_cast<size_t>(w)) * 16;
    }
    const float* pixel(int cb, int h, int w) const {
        return data.data() + (static_cast<size_t>(cb * H + h) * static_cast<size_t>(W) + static_cast<size_t>(w)) * 16;
    }

    void reuse(int new_CB, int new_H, int new_W) {
        CB = new_CB; H = new_H; W = new_W;
        const size_t req = static_cast<size_t>(CB * H * W) * 16;
        if (data.size() < req) data.assign(req, 0.0f);
    }

    int C() const { return CB * 16; }
};

struct BlockedTensorI8 {
    int CB    = 0;
    int H     = 0;
    int W     = 0;
    float scale = 1.0f;
    std::vector<int8_t> data;

    BlockedTensorI8() = default;
    BlockedTensorI8(int CB_, int H_, int W_, float scale_ = 1.0f)
        : CB(CB_), H(H_), W(W_), scale(scale_),
          data(static_cast<size_t>(CB_ * H_ * W_) * 16, 0) {}

    int8_t* pixel(int cb, int h, int w) {
        return data.data() + (static_cast<size_t>(cb * H + h) * static_cast<size_t>(W) + static_cast<size_t>(w)) * 16;
    }
    const int8_t* pixel(int cb, int h, int w) const {
        return data.data() + (static_cast<size_t>(cb * H + h) * static_cast<size_t>(W) + static_cast<size_t>(w)) * 16;
    }

    void reuse(int new_CB, int new_H, int new_W, float new_scale = 1.0f) {
        CB = new_CB; H = new_H; W = new_W; scale = new_scale;
        const size_t req = static_cast<size_t>(CB * H * W) * 16;
        if (data.size() < req) data.assign(req, 0);
    }

    int C() const { return CB * 16; }
};
