#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

struct Tensor {
    int c = 0;
    int h = 0;
    int w = 0;
    std::vector<float> data;

    Tensor() = default;

    Tensor(int channels, int height, int width): c(channels), h(height), w(width), data(static_cast<size_t>(channels * height * width), 0.0f) {}

    // Reuse this tensor for new shape: no-op (no alloc, no zero-init) if size unchanged.
    void reuse(int channels, int height, int width) {
        c = channels; h = height; w = width;
        const size_t n = static_cast<size_t>(c * h * w);
        if (data.size() != n) data.resize(n);
    }

    // Pre-allocate capacity so future reuse() calls within this size never reallocate.
    void reserve(int channels, int height, int width) {
        data.reserve(
            static_cast<size_t>(channels) *
            static_cast<size_t>(height) *
            static_cast<size_t>(width)
        );
    }

    size_t size() const {
        return data.size();
    }

    float& operator()(int ic, int iy, int ix) {
        assert(ic >= 0 && ic < c);
        assert(iy >= 0 && iy < h);
        assert(ix >= 0 && ix < w);
        return data[static_cast<size_t>((ic * h + iy)* w + ix)];
    }

    const float& operator()(int ic, int iy, int ix) const {
        assert(ic >=0 && ic < c);
        assert(iy >= 0 && iy < h);
        assert(ix >= 0 && ix < w);
        return data[static_cast<size_t>((ic * h + iy)* w + ix)];
    }

    float* ptr(int ic, int iy, int ix) {
        return &data[static_cast<size_t>((ic * h + iy)* w + ix)];
    }

    const float* ptr(int ic, int iy, int ix) const {
        return &data[static_cast<size_t>((ic * h + iy)* w + ix)];
    }
};