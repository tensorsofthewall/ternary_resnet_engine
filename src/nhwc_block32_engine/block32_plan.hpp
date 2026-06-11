#pragma once

#include "model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nhwc_block32 {

// Reduction-axis chunk width: one AVX2 register holds 32 int8 ternary
// weights / uint8 activations.
constexpr int BLOCK32 = 32;

// One (ky, kx, ic_block) reduction-axis block of ternary weights for a
// single output channel: w[ci] in {-1,0,+1} is the weight at kernel position
// (ky,kx) for input channel ic_block*BLOCK32 + ci. build_block32_plan
// drops blocks where every w[ci] == 0, but for ternary weights at width 32
// this essentially never fires (P(all-zero) = (1/3)^32) -- in practice this
// is a dense block32 ternary plan, one block per (ky,kx,ic_block) tuple.
struct Block32 {
    uint8_t ky = 0;
    uint8_t kx = 0;
    uint16_t ic_block = 0;
    int8_t w[BLOCK32] = {};
};

// One output channel's block32 ternary weight blocks, in (ky, kx, ic_block)
// iteration order.
struct OCPlan {
    std::vector<Block32> blocks;
};

// Block32 ternary rewrite of a Conv2DParams: same shape/stride/pad metadata,
// with the dense ternary weight tensor replaced by one OCPlan per
// output channel (built by build_block32_plan from a Conv2DParams
// whose in_channels is a multiple of BLOCK32 and whose weights are
// verified ternary).
struct Block32ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int ic_blocks = 0;  // in_channels / BLOCK32

    std::vector<OCPlan> oc_plans;  // size == out_channels
};

// True if `conv.in_channels` is a multiple of BLOCK32 (i.e. every
// (ky,kx) reduction axis can be split into whole 32-wide blocks). False for
// conv1 (in_channels == 3).
bool can_use_block32_plan(const Conv2DParams& conv);

// Builds a Block32ConvPlan from `conv`. Throws if conv.in_channels is
// not a multiple of BLOCK32, if the weight tensor size doesn't match
// the declared shape, or if any weight value is outside {-1,0,+1}.
Block32ConvPlan build_block32_plan(const Conv2DParams& conv);

// Throws unless every output channel's block list has the same length and
// (ky,kx,ic_block) sequence as oc_plans[0]. OC4 (and wider) kernels walk the
// blocks of several output channels in lockstep using only one channel's
// (ky,kx,ic_block) coordinates, which is only valid under this invariant --
// true in practice because build_block32_plan iterates all (ky,kx,ic_block)
// combos in a fixed order and never drops one as all-zero for ternary
// weights at width 32 (see Block32 above).
void require_uniform_block_layout(const Block32ConvPlan& plan);

} // namespace nhwc_block32
