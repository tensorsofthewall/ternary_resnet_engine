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