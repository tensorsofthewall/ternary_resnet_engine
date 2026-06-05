#pragma once

#include <cstdint>
#include <vector>

struct QuantTensorI8 {
    int c = 0;
    int h = 0;
    int w = 0;

    // Symmetric quantization:
    // fp32 ~= int8 * scale
    float scale = 1.0f;

    std::vector<int8_t> data;

    QuantTensorI8() = default;

    QuantTensorI8(int channels, int height, int width, float s)
        : c(channels),
          h(height),
          w(width),
          scale(s),
          data(static_cast<size_t>(channels * height * width), 0) {}

    inline int8_t& operator()(int ch, int y, int x) {
        return data[static_cast<size_t>((ch * h + y) * w + x)];
    }

    inline const int8_t& operator()(int ch, int y, int x) const {
        return data[static_cast<size_t>((ch * h + y) * w + x)];
    }

    inline int8_t* ptr(int ch, int y, int x) {
        return data.data() + static_cast<size_t>((ch * h + y) * w + x);
    }

    inline const int8_t* ptr(int ch, int y, int x) const {
        return data.data() + static_cast<size_t>((ch * h + y) * w + x);
    }

    inline size_t size() const {
        return data.size();
    }
};