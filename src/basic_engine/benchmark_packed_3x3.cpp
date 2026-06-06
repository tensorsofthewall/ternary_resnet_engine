// Phase 1: standalone benchmark harness for the packed sparse 3x3 kernel.
//
// Compares the current q8 OCBlock2 bucket kernel against the packed sparse
// activation-panel kernel (Phase 2) on a single captured layer activation.
//
// Usage:
//   ./build/benchmark_packed_3x3 <model.bin> [<input_fp32.bin>]
//
// If input is omitted, defaults to artifacts/golden_input_fp32.bin.
// Dump artifacts are written to artifacts/ and reused on subsequent runs.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_loader.hpp"
#include "resnet18_engine.hpp"
#include "tensor.hpp"
#include "workspace.hpp"
#include "quantized/quant_utils.hpp"
#include "quantized/conv3x3_q8_pattern_ocblock2_avx2.hpp"
#include "quantized/packed_q8_3x3_kernel.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string sanitize(std::string s) {
    for (char& c : s) if (c == '.') c = '_';
    return s;
}

static std::vector<float> read_f32_file(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f) throw std::runtime_error("Short read: " + path);
    return data;
}

static Tensor load_tensor(const std::string& path, int c, int h, int w) {
    Tensor t(c, h, w);
    t.data = read_f32_file(path, static_cast<size_t>(c * h * w));
    return t;
}

static double duration_ms(
    std::chrono::time_point<std::chrono::high_resolution_clock> t0,
    std::chrono::time_point<std::chrono::high_resolution_clock> t1
) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Stats {
    double min_ms = 0.0;
    double median_ms = 0.0;
};

static Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.min_ms    = v.front();
    s.median_ms = v[v.size() / 2];
    return s;
}

// ---------------------------------------------------------------------------
// Layer specification
// ---------------------------------------------------------------------------

struct LayerSpec {
    const char* conv_name;   // key into model.convs and model.pattern_ocblock2_plans
    const char* bn_name;     // key into model.fused_bns
    bool        has_relu;    // conv1 in a block (conv_bn_relu); false = conv2 (conv_bn)
    int         extra_right_pad;  // right pad added beyond conv.pad_w in the engine
    int         ic, oc;      // input/output channels
    int         oh, ow;      // output height/width
};

static const LayerSpec LAYERS[] = {
    // layer3 stride-1 (IC=256, spatial 14x14)
    {"layer3.1.conv2", "layer3.1.bn2", false, 4, 256, 256, 14, 14},
    {"layer3.0.conv2", "layer3.0.bn2", false, 4, 256, 256, 14, 14},
    {"layer3.1.conv1", "layer3.1.bn1", true,  4, 256, 256, 14, 14},
    // layer4 stride-1 (IC=512, spatial 7x7)
    {"layer4.1.conv2", "layer4.1.bn2", false, 4, 512, 512,  7,  7},
    {"layer4.0.conv2", "layer4.0.bn2", false, 4, 512, 512,  7,  7},
    {"layer4.1.conv1", "layer4.1.bn1", true,  4, 512, 512,  7,  7},
};
static constexpr int N_LAYERS = static_cast<int>(sizeof(LAYERS) / sizeof(LAYERS[0]));

// ---------------------------------------------------------------------------
// Bucket kernel dispatch (mirrors conv_bn / conv_bn_relu in the engine)
// ---------------------------------------------------------------------------

static void run_bucket_kernel(
    const LayerSpec&              spec,
    const Conv2DParams&           conv,
    const PatternOCBlock2ConvPlan& plan,
    const BNFusedParams&          bn_fused,
    const QuantTensorI8&          xq,
    Tensor&                       y_out
) {
    if (spec.ic == 256) {
        if (spec.has_relu)
            conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_relu_avx2(xq, conv, plan, bn_fused, y_out);
        else
            conv3x3_q8_ocblock2_spatial4_layer3_s1_bn_avx2(xq, conv, plan, bn_fused, y_out);
    } else {  // IC=512 (layer4)
        if (spec.has_relu)
            conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_relu_avx2(xq, conv, plan, bn_fused, y_out);
        else
            conv3x3_q8_ocblock2_spatial4_layer4_s1_bn_avx2(xq, conv, plan, bn_fused, y_out);
    }
}

// ---------------------------------------------------------------------------
// Error stats between two flat float arrays
// ---------------------------------------------------------------------------

struct ErrorStats {
    float max_abs  = 0.0f;
    float mean_abs = 0.0f;
    float rel_mean = 0.0f;  // mean_abs / max(|golden|)
};

static ErrorStats compute_error(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size())
        throw std::runtime_error("Size mismatch in compute_error");

    float max_abs  = 0.0f;
    double sum_abs = 0.0;
    float max_golden = 0.0f;

    for (size_t i = 0; i < a.data.size(); ++i) {
        const float err = std::abs(a.data[i] - b.data[i]);
        max_abs  = std::max(max_abs, err);
        sum_abs += static_cast<double>(err);
        max_golden = std::max(max_golden, std::abs(b.data[i]));
    }

    ErrorStats e;
    e.max_abs  = max_abs;
    e.mean_abs = static_cast<float>(sum_abs / static_cast<double>(a.data.size()));
    e.rel_mean = max_golden > 0.0f ? e.mean_abs / max_golden : 0.0f;
    return e;
}

// ---------------------------------------------------------------------------
// Dump: run one timed forward pass to capture layer inputs/outputs
// ---------------------------------------------------------------------------

static std::string input_path(const LayerSpec& s) {
    return std::string("artifacts/") + sanitize(s.conv_name) + "_input_fp32.bin";
}
static std::string output_path(const LayerSpec& s) {
    return std::string("artifacts/") + sanitize(s.conv_name) + "_output_fp32.bin";
}

static bool dumps_exist(const LayerSpec& s) {
    return std::filesystem::exists(input_path(s)) &&
           std::filesystem::exists(output_path(s));
}

static void run_dump_pass(
    const Model&   model,
    const Tensor&  full_input,
    EngineWorkspace& ws,
    const std::vector<std::string>& targets
) {
    set_tensor_dump_layers(targets);
    std::vector<TimingEntry> dummy_timings;
    forward_resnet18_scalar_timed(model, full_input, ws, dummy_timings);
    clear_tensor_dump_layers();
}

// ---------------------------------------------------------------------------
// Packed kernel dispatch
// ---------------------------------------------------------------------------

static void run_packed_kernel_row(
    const LayerSpec&          spec,
    const PackedSparseQ8Plan& packed_plan,
    const BNFusedParams&      bn,
    const QuantTensorI8&      xq,
    int8_t*                   A_panel,
    int                       oh,
    Tensor&                   y_out
) {
    if (spec.ic == 256) {
        pack_a_panel_layer3(xq, oh, A_panel);
        if (spec.has_relu)
            packed_q8_3x3_s1_layer3_bn_relu_avx2(A_panel, packed_plan, bn, xq.scale, oh, y_out);
        else
            packed_q8_3x3_s1_layer3_bn_avx2(A_panel, packed_plan, bn, xq.scale, oh, y_out);
    } else {
        pack_a_panel_layer4(xq, oh, A_panel);
        if (spec.has_relu)
            packed_q8_3x3_s1_layer4_bn_relu_avx2(A_panel, packed_plan, bn, xq.scale, oh, y_out);
        else
            packed_q8_3x3_s1_layer4_bn_avx2(A_panel, packed_plan, bn, xq.scale, oh, y_out);
    }
}

// ---------------------------------------------------------------------------
// Benchmark one layer
// ---------------------------------------------------------------------------

static void benchmark_layer(
    const Model&    model,
    const LayerSpec& spec,
    int             n_runs
) {
    // Load captured FP32 input and bucket-kernel golden output
    const Tensor x_fp32 = load_tensor(
        input_path(spec), spec.ic, spec.oh, spec.ow);
    const Tensor golden = load_tensor(
        output_path(spec), spec.oc, spec.oh, spec.ow);

    const Conv2DParams&            conv        = model.convs.at(spec.conv_name);
    const PatternOCBlock2ConvPlan& bucket_plan = model.pattern_ocblock2_plans.at(spec.conv_name);
    const BNFusedParams&           bn_fused    = model.fused_bns.at(spec.bn_name);

    // Build packed plan (one-time, outside timing)
    const PackedSparseQ8Plan packed_plan = build_packed_sparse_q8_plan(conv);

    // Pre-allocate workspace buffers reused across timing iterations
    QuantTensorI8 xq;
    const size_t padded_sz = static_cast<size_t>(spec.ic)
                           * static_cast<size_t>(spec.oh + 2)
                           * static_cast<size_t>(spec.ow + 2 + spec.extra_right_pad);
    xq.data.reserve(padded_sz);

    Tensor y_bucket(spec.oc, spec.oh, spec.ow);
    Tensor y_packed(spec.oc, spec.oh, spec.ow);

    // A_panel buffer: K * N_PAD bytes, reused across all oh iterations
    std::vector<int8_t> A_panel(static_cast<size_t>(packed_plan.K) * 16);

    // Warmup — bucket
    for (int i = 0; i < 5; ++i) {
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w,
            conv.pad_w + spec.extra_right_pad, xq);
        run_bucket_kernel(spec, conv, bucket_plan, bn_fused, xq, y_bucket);
    }

    // Warmup — packed
    for (int i = 0; i < 5; ++i) {
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w,
            conv.pad_w + spec.extra_right_pad, xq);
        for (int oh = 0; oh < spec.oh; ++oh)
            run_packed_kernel_row(spec, packed_plan, bn_fused, xq, A_panel.data(), oh, y_packed);
    }

    // Timed runs — bucket
    std::vector<double> bq_times(n_runs), bc_times(n_runs), bt_times(n_runs);
    for (int i = 0; i < n_runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w,
            conv.pad_w + spec.extra_right_pad, xq);
        const auto t1 = std::chrono::high_resolution_clock::now();
        run_bucket_kernel(spec, conv, bucket_plan, bn_fused, xq, y_bucket);
        const auto t2 = std::chrono::high_resolution_clock::now();
        bq_times[i] = duration_ms(t0, t1);
        bc_times[i] = duration_ms(t1, t2);
        bt_times[i] = duration_ms(t0, t2);
    }

    // Timed runs — packed (quantize + interleaved pack+compute per oh)
    std::vector<double> pq_times(n_runs), pc_times(n_runs), pt_times(n_runs);
    for (int i = 0; i < n_runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantize_pad_tensor_symmetric_i8_asymmetric_into(
            x_fp32, conv.pad_h, conv.pad_h, conv.pad_w,
            conv.pad_w + spec.extra_right_pad, xq);
        const auto t1 = std::chrono::high_resolution_clock::now();
        for (int oh = 0; oh < spec.oh; ++oh)
            run_packed_kernel_row(spec, packed_plan, bn_fused, xq, A_panel.data(), oh, y_packed);
        const auto t2 = std::chrono::high_resolution_clock::now();
        pq_times[i] = duration_ms(t0, t1);
        pc_times[i] = duration_ms(t1, t2);
        pt_times[i] = duration_ms(t0, t2);
    }

    const Stats bqs = compute_stats(bq_times);
    const Stats bcs = compute_stats(bc_times);
    const Stats bts = compute_stats(bt_times);
    const Stats pqs = compute_stats(pq_times);
    const Stats pcs = compute_stats(pc_times);
    const Stats pts = compute_stats(pt_times);

    const double speedup_total  = bts.min_ms / pts.min_ms;
    const double speedup_compute = bcs.min_ms / pcs.min_ms;

    const ErrorStats err_bucket = compute_error(y_bucket, golden);

    const int K = spec.ic * 9;
    const int N = spec.oh * spec.ow;
    const int M = spec.oc;

    std::cout << "\n--- " << spec.conv_name << " ---\n";
    std::cout << "  input:   " << spec.ic << " x " << spec.oh << " x " << spec.ow << "\n";
    std::cout << "  output:  " << spec.oc << " x " << spec.oh << " x " << spec.ow << "\n";
    std::cout << "  K=" << K << "  N=" << N << "  M=" << M << "\n";

    std::cout << "\n  current bucket kernel (" << n_runs << " runs):\n";
    std::cout << "    quantize:  min=" << bqs.min_ms << " ms  median=" << bqs.median_ms << " ms\n";
    std::cout << "    compute:   min=" << bcs.min_ms << " ms  median=" << bcs.median_ms << " ms\n";
    std::cout << "    total:     min=" << bts.min_ms << " ms  median=" << bts.median_ms << " ms\n";

    std::cout << "\n  packed sparse kernel (OC_TILE=1, N_TILE=" << spec.ow
              << ", N_PAD=16):\n";
    std::cout << "    quantize:  min=" << pqs.min_ms << " ms  median=" << pqs.median_ms << " ms\n";
    std::cout << "    compute:   min=" << pcs.min_ms << " ms  median=" << pcs.median_ms << " ms\n";
    std::cout << "    total:     min=" << pts.min_ms << " ms  median=" << pts.median_ms << " ms\n";
    std::cout << "    speedup (compute): " << speedup_compute << "x\n";
    std::cout << "    speedup (total):   " << speedup_total   << "x\n";

    std::cout << "\n  correctness (standalone bucket vs golden):\n";
    std::cout << "    max_abs_error:          " << err_bucket.max_abs  << "\n";
    std::cout << "    mean_abs_error:         " << err_bucket.mean_abs << "\n";
    std::cout << "    relative_mean_abs_error " << err_bucket.rel_mean << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_packed_3x3 <model.bin> [<input_fp32.bin>]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_bin  = argc >= 3 ? argv[2] : "artifacts/golden_input_fp32.bin";

    std::cout << "Loading model: " << model_path << "\n";
    const Model model = load_model(model_path);

    std::cout << "Loading input: " << input_bin << "\n";
    Tensor full_input(3, 224, 224);
    full_input.data = read_f32_file(input_bin, 3 * 224 * 224);

    EngineWorkspace ws;

    // Warmup the engine workspace (lets ws buffers grow to steady-state size)
    std::cout << "Warmup forward passes...\n";
    for (int i = 0; i < 5; ++i) {
        forward_resnet18_scalar(model, full_input, ws);
    }

    // Determine which layers still need their dump artifacts
    std::vector<std::string> to_dump;
    for (int i = 0; i < N_LAYERS; ++i) {
        if (!dumps_exist(LAYERS[i])) {
            to_dump.push_back(LAYERS[i].conv_name);
        }
    }

    if (!to_dump.empty()) {
        std::cout << "Running dump pass for " << to_dump.size() << " layer(s)...\n";
        for (const auto& name : to_dump) std::cout << "  " << name << "\n";
        run_dump_pass(model, full_input, ws, to_dump);
        std::cout << "Dump complete. Artifacts written to artifacts/\n";
    } else {
        std::cout << "All dump artifacts already exist — skipping dump pass.\n";
    }

    constexpr int N_RUNS = 100;

    for (int i = 0; i < N_LAYERS; ++i) {
        benchmark_layer(model, LAYERS[i], N_RUNS);
    }

    std::cout << "\nDone.\n";
    return 0;
}
