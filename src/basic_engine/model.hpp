#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Dense 2D convolution layer parameters as loaded from the model file:
// ternary {-1,0,+1} weights stored as int8 in [out_channels, in_channels,
// kh, kw] order, plus the usual stride/pad/dilation and an optional bias.
struct Conv2DParams {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;

    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int dilation_h = 0;
    int dilation_w = 0;

    bool has_bias = false;

    // Shape: [out_channels, in_channels, kh, kw]
    std::vector<int8_t> weight;
    std::vector<float> bias;
};

// Raw batch-norm parameters (gamma/beta/running mean & variance) as loaded
// from the model file, before fusion into a single affine scale+bias.
struct BatchNormParams {
    std::string name;

    int num_features = 0;
    float eps = 0.0f;

    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> running_mean;
    std::vector<float> running_var;
};

// BatchNormParams folded into a single per-channel affine transform, so
// inference applies one multiply-add instead of the full BN formula.
struct BNFusedParams {
    std::string name;
    int num_features=0;

    // y = x*scale[c]+bias[c]
    std::vector<float> scale;
    std::vector<float> bias;
};

// Fully-connected layer parameters (the final FC head): dense FP32 weights
// in [out_features, in_features] order, plus an optional bias.
struct LinearParams {
    std::string name;

    int in_features = 0;
    int out_features = 0;
    bool has_bias = false;

    // Shape: [out_features, in_features]
    std::vector<float> weight;
    std::vector<float> bias;
};

// Spatial position (input channel, kernel row, kernel column) of one
// nonzero ternary weight -- the weight's own value (+1 or -1) is implied by
// which of ConvOCPlan::pos / ::neg it's stored in.
struct WeightIndex {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;
};

// One output channel's ternary weights split into "+1" and "-1" position
// lists, so a sparse conv kernel can compute the dot product as
// sum(pos activations) - sum(neg activations) with no multiplies.
struct ConvOCPlan {
    std::vector<WeightIndex> pos;
    std::vector<WeightIndex> neg;
};

// Sparse rewrite of a Conv2DParams: same shape/stride/pad metadata, but the
// dense ternary weight tensor is replaced by one ConvOCPlan per output
// channel (built by build_conv_plan from a Conv2DParams whose weights are
// verified ternary).
struct ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<ConvOCPlan> oc_plans;
};

// Sparse OC-Block4 definitions
//
// One nonzero spatial position (ic,ky,kx) for a group of 4 consecutive
// output channels, carrying that position's ternary weight for each of the
// 4 channels (w0..w3, each in {-1,0,+1}; an action is only emitted when at
// least one is nonzero -- see build_sparse_ocblock4_conv_plan).
struct OCBlock4Action {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;

    int8_t w0 = 0;
    int8_t w1 = 0;
    int8_t w2 = 0;
    int8_t w3 = 0;
};

// All nonzero actions for one 4-output-channel block (channels
// [oc_start, oc_start+4)), in spatial scan order.
struct OCBlock4Plan {
    int oc_start = 0; // starting output channel index of the block
    std::vector<OCBlock4Action> actions;
};

// Sparse rewrite of a 3x3/stride-1 Conv2DParams (out_channels % 4 == 0):
// output channels are grouped into blocks of 4 and each block's nonzero
// ternary weights are listed as OCBlock4Actions, so a kernel can accumulate
// 4 output channels per activation load. Built by
// build_sparse_ocblock4_conv_plan.
struct SparseOCBlock4ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<OCBlock4Plan> oc_blocks;
};

// Sparse OC-Block2 sign-pattern grouping definitions
//
// One nonzero spatial position (ic,ky,kx) shared by both output channels in
// a 2-channel block -- the pair's ternary weights live on the enclosing
// OCBlock2PatternBucket, not per-action, since every action in a bucket has
// the same (w0,w1) pattern.
struct OCBlock2Action {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;
};

// All spatial positions sharing one (w0,w1) ternary sign pattern
// (pattern_id encodes the pair as a base-3 digit pair), grouped together so
// a kernel can apply one pattern-specific add/sub combination to every
// action in the bucket without per-position branching.
struct OCBlock2PatternBucket {
    int pattern_id = 0;
    int8_t w0 = 0;
    int8_t w1 = 0;

    std::vector<OCBlock2Action> actions;
};

// All sign-pattern buckets for one 2-output-channel block
// (channels [oc_start, oc_start+2)).
struct OCBlock2PatternPlan {
    int oc_start = 0;

    std::vector<OCBlock2PatternBucket> buckets;
};

// Sign-pattern-bucketed sparse rewrite of a Conv2DParams: output channels
// are grouped into blocks of 2 and each block's nonzero weight positions are
// bucketed by their shared (w0,w1) ternary pattern. Built by
// build_pattern_ocblock2_plan.
struct PatternOCBlock2ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<OCBlock2PatternPlan> oc_blocks;
};

// Sparse OC-Block4 sign-pattern grouping definitions
//
// One nonzero spatial position (ic,ky,kx) shared by all 4 output channels in
// a 4-channel block -- like OCBlock2Action, the ternary weights live on the
// enclosing OCBlock4PatternBucket (every action in a bucket shares the same
// (w0,w1,w2,w3) pattern).
struct OCBlock4PatternAction {
    uint16_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;
};

// All spatial positions sharing one (w0,w1,w2,w3) ternary sign pattern
// (pattern_id = base-3 encoding of the 4-tuple, one of 3^4 = 81 patterns,
// via pattern_index/pattern_from_index in pattern_ocblock4_plan.cpp).
struct OCBlock4PatternBucket {
    int pattern_id = 0;

    int8_t w0 = 0;
    int8_t w1 = 0;
    int8_t w2 = 0;
    int8_t w3 = 0;

    std::vector<OCBlock4PatternAction> actions;
};

// Flat per-action record storing the 4 weights alongside the spatial position.
// Actions are stored sorted by (w0,w1,w2,w3) pattern so consecutive actions with
// the same pattern are together — branch prediction is perfect within each run.
struct OCBlock4FlatAction {
    uint16_t ic = 0;
    uint8_t  ky = 0;
    uint8_t  kx = 0;
    int8_t   w0 = 0;
    int8_t   w1 = 0;
    int8_t   w2 = 0;
    int8_t   w3 = 0;
    // 8 bytes total: naturally aligned, one cache line holds 8 actions.
};

// All sign-pattern buckets for one 4-output-channel block
// (channels [oc_start, oc_start+4)), plus a flat_actions view of the same
// data for kernels that prefer to scan one contiguous, pattern-sorted array
// over walking the bucket list.
struct OCBlock4PatternPlan {
    int oc_start = 0;
    std::vector<OCBlock4PatternBucket> buckets;
    // Flat view: all non-(all-zero) actions in one contiguous block.
    // Sorted by bucket pattern for branch-prediction benefit without pointer chasing.
    std::vector<OCBlock4FlatAction> flat_actions;
};

// Sign-pattern-bucketed sparse rewrite of a 3x3, stride-1-or-2
// Conv2DParams (out_channels % 4 == 0): output channels are grouped into
// blocks of 4 and each block's nonzero weight positions are bucketed by
// their shared (w0,w1,w2,w3) ternary pattern. Built by
// build_pattern_ocblock4_plan.
struct PatternOCBlock4ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<OCBlock4PatternPlan> oc_blocks;
};

// Specialized Sparse Conv1 impl
//
// Conv1-specific analogue of OCBlock4PatternAction: identical layout, but
// `ic` is narrowed to uint8_t since conv1's in_channels is always 3.
struct Conv1OCBlock4Action {
    uint8_t ic = 0;
    uint8_t ky = 0;
    uint8_t kx = 0;
};

// Conv1-specific analogue of OCBlock4PatternBucket (see model.hpp's
// OCBlock4PatternBucket for the shared (w0..w3) sign-pattern-bucketing idea).
struct Conv1OCBlock4PatternBucket {
    int pattern_id = 0;

    int8_t w0 = 0;
    int8_t w1 = 0;
    int8_t w2 = 0;
    int8_t w3 = 0;

    std::vector<Conv1OCBlock4Action> actions;
};

// All sign-pattern buckets for one 4-output-channel block of conv1
// (channels [oc_start, oc_start+4)).
struct Conv1OCBlock4Plan {
    int oc_start = 0;
    std::vector<Conv1OCBlock4PatternBucket> buckets;
};

// Sign-pattern-bucketed sparse rewrite of the conv1 layer specifically
// (7x7/stride-2, in_channels==3): same grouping idea as
// PatternOCBlock4ConvPlan, specialized for conv1's fixed small IC. Built by
// build_conv1_ocblock4_plan.
struct Conv1OCBlock4ConvPlan {
    std::string name;

    int in_channels = 0;
    int out_channels = 0;
    int kh = 0;
    int kw = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;

    std::vector<Conv1OCBlock4Plan> oc_blocks;
};

// The full loaded ResNet18 model: dense conv/BN/linear params keyed by layer
// name (as read from the model file by model_loader), plus every alternative
// sparse/pattern-based ConvPlan representation built on demand by the
// build_*_plan(s) helpers (each map is populated only for the layers that
// support that representation -- see each plan struct's supports_* predicate).
struct Model {
    std::unordered_map<std::string, Conv2DParams> convs;
    std::unordered_map<std::string, ConvPlan> conv_plans;
    std::unordered_map<std::string, SparseOCBlock4ConvPlan> sparse_ocblock4_plans;
    std::unordered_map<std::string, PatternOCBlock2ConvPlan> pattern_ocblock2_plans;
    std::unordered_map<std::string, PatternOCBlock4ConvPlan> pattern_ocblock4_plans;
    std::unordered_map<std::string, Conv1OCBlock4ConvPlan> conv1_ocblock4_plans;
    
    std::unordered_map<std::string, BatchNormParams> bns;
    std::unordered_map<std::string, BNFusedParams> fused_bns;
    std::unordered_map<std::string, LinearParams> linears;
};