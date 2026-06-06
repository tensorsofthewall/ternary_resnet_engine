# Ternary ResNet-18 Inference Engine

A from-scratch C++ CPU inference engine for a Ternary Weight Network (TWN) variant of
ResNet-18. Convolution weights are constrained to `{-1, 0, +1}`, so multiply-accumulate
can be replaced with add/subtract over non-zero weights. The engine targets x86-64 CPUs
with AVX2/FMA and uses no external ML inference library in the C++ runtime.

This repo is intentionally an engineering study: it builds a working inference engine,
profiles bottlenecks, implements specialized kernels, compares against ONNX Runtime, and
documents both successful and reverted optimization attempts.

The current engine is still slower than ONNX Runtime's mature MLAS/GEMM kernels, but the
custom path improves from a multi-second scalar baseline to low-millisecond multi-thread
latency on the test machine.

---

## Assignment Scope

This project implements a standalone C++ inference engine for a ResNet-18 model with
ternary weights (`{-1, 0, +1}`), targeting the following requirements:

- **No ML inference framework in the C++ runtime.** The engine uses no PyTorch, ONNX
  Runtime, MLAS, oneDNN, or OpenBLAS in its inference path. ORT and PyTorch are used
  exclusively as benchmarking and profiling references.
- **SIMD convolution kernels.** AVX2/FMA kernels cover all major layer families using
  ternary-sparse dispatch, pattern grouping, and dynamic INT8 activation quantization.
- **Ternary weight packing.** Conv weights are stored as `int8_t` values in
  `{-1, 0, +1}`. A dense 2-bit packed representation is provided in
  `artifacts/model_ternary_weights_2bit_packed.bin` for footprint reporting. At runtime,
  weights are expanded into sparse/pattern plans for AVX2 execution.
- **Benchmark comparison.** End-to-end latency is measured against PyTorch FP32, ORT
  FP32, and ORT INT8 (static quantized) across 1–24 threads.
- **Per-layer profiling.** The engine reports per-block latency in each run.
- **Memory footprint reporting.** Artifact sizes, theoretical weight encodings, and
  runtime workspace are reported by `scripts/report_memory_footprint.py`.

---

## Hardware

All benchmark numbers in this README were measured on:

- **CPU**: Intel Core i7-12800HX, 12th Gen Alder Lake-HX
  - 8 P-cores + 8 E-cores / 24 hardware threads
  - AVX2 + FMA supported
  - No AVX-512 on this machine
- **OS**: Linux, kernel 6.18 series
- **Compiler**: GCC with `-O3 -march=native -mavx2 -mfma`
- **Runtime threading**: OpenMP for the C++ engine, ORT intra-op threading for ONNX Runtime

Useful C++ scaling extends to about 8 P-cores, after which additional threads land on
E-cores with different clock characteristics. At 16–24 threads the engine's best-observed
min latency beats ORT INT8 (which degrades due to QDQ graph fragmentation), but ORT FP32
remains faster throughout. High-thread median variance reflects hybrid-core (P+E) scheduler
effects.

---

## Prerequisites

### System packages

Ubuntu / Debian:

```bash
sudo apt install build-essential cmake libgomp1
```

Arch / EndeavourOS:

```bash
sudo pacman -S base-devel cmake gcc
```

### Python environment

The repo uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management.

```bash
uv sync
```

Alternatively, use any Python 3.11+ environment and install the required packages manually:

```bash
pip install numpy onnx onnxruntime torch torchvision onnxscript
```

---

## Build

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

This produces:

```text
build/basic_engine
build/benchmark_layer4_q8
build/benchmark_packed_3x3
```

The main executable is `build/basic_engine`.

To build with heap-allocation profiling enabled:

```bash
cmake -S . -B build_alloc -DENABLE_ALLOC_PROFILE=ON
cmake --build build_alloc -j$(nproc)
```

---

## Generating Artifacts

All benchmarks require generated model and input artifacts:

```bash
uv run python tools/export_model.py
```

This writes the following into `artifacts/`:

| File | Description |
|---|---|
| `model_ternary_i8_weights_v1.bin` | Custom binary format containing ternary conv weights, BN params, FC params, and kernel plans |
| `golden_input_fp32.bin` | Fixed random input image, shape `3×224×224`, float32 |
| `golden_logits_fp32.bin` | PyTorch reference logits for correctness checks |
| `golden_input.npy` | Same input in NumPy format for ORT benchmarks |
| `resnet18_ternary_fp32.onnx` | FP32 ONNX export of the same model |

The INT8 ONNX model is generated lazily by the ORT benchmark script:

```text
artifacts/resnet18_ternary_int8.onnx
```

The default export seeds are deterministic, so benchmark runs are comparable as long as the
same artifacts are used.

---

## Running the C++ Engine

Single-thread run:

```bash
OMP_NUM_THREADS=1 ./build/basic_engine \
  artifacts/model_ternary_i8_weights_v1.bin \
  artifacts/golden_input_fp32.bin \
  artifacts/golden_logits_fp32.bin
```

Eight-thread run:

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
./build/basic_engine \
  artifacts/model_ternary_i8_weights_v1.bin \
  artifacts/golden_input_fp32.bin \
  artifacts/golden_logits_fp32.bin
```

The output contains:

- `C++ engine latency`: min/median/mean/p90 over 100 timed runs after 20 warmup runs
- `C++ engine per-block profile`: one additional workspace-backed profiling pass
- `Top timed operations`: largest per-layer costs
- `total_profiled_ms`: sum of the per-block profiling pass
- PyTorch-golden logit error and sample logits

The per-block profile uses `forward_resnet18_workspace_timed`, which mirrors the same
workspace buffers and in-place add/ReLU behavior as the benchmarked fast path. The old
return-by-value timed path is kept only as a legacy/debug reference.

---

## Thread Scaling Benchmark

Run:

```bash
bash scripts/bench_threads.sh
```

The script builds the project, runs selected thread counts, and logs output to:

```text
logs/thread_scaling_<timestamp>.log
```

Edit `THREAD_COUNTS` in `scripts/bench_threads.sh` to sweep additional values.

---

## C++ vs ONNX Runtime Benchmark

Run:

```bash
uv run python scripts/benchmark_cpp_vs_ort.py \
  --threads 1 2 4 8 16 \
  --runs 100 \
  --warmup 20 \
  --build
```

Useful flags:

| Flag | Description |
|---|---|
| `--threads 1 2 4 8` | Thread counts to benchmark |
| `--runs 100` | Timed runs per configuration |
| `--warmup 20` | Warmup runs before timing |
| `--build` | Run `cmake --build build` before benchmarking |
| `--skip-ort-fp32` | Skip ORT FP32 baseline |
| `--skip-ort-int8` | Skip ORT INT8 baseline |
| `--force-quantize` | Regenerate cached ORT INT8 ONNX model |
| `--ort-profile` | Run an ORT INT8 profiling pass and print the Chrome trace JSON path |

Outputs are written to:

```text
logs/cpp_vs_ort_<timestamp>.json
logs/cpp_vs_ort_<timestamp>.csv
```

The script sets ORT intra-op threading and C++ OpenMP threading consistently and avoids
Python-side CPU affinity mutation.

---

## PyTorch FP32 Benchmark

```bash
python scripts/benchmark_pytorch_fp32.py \
  --threads 1 2 4 8 16 24 \
  --runs 100 \
  --warmup 20
```

Outputs are written to:

```text
logs/pytorch_fp32_<timestamp>.json
logs/pytorch_fp32_<timestamp>.csv
```

---

## Memory Footprint Report

```bash
python scripts/report_memory_footprint.py
```

Outputs a Markdown table of model artifact sizes, theoretical weight storage, and runtime
workspace breakdown. Writes to `logs/memory_footprint_<timestamp>.json` and `.md`.

To generate the dense 2-bit packed ternary weight artifact first:

```bash
python tools/pack_ternary_weights.py
```

This writes `artifacts/model_ternary_weights_2bit_packed.bin`.

---

## Allocation Profiling

Allocation profiling is optional and is compiled in with `ENABLE_ALLOC_PROFILE`:

```bash
cmake -S . -B build_alloc -DENABLE_ALLOC_PROFILE=ON
cmake --build build_alloc -j$(nproc)

OMP_NUM_THREADS=1 ./build_alloc/basic_engine \
  artifacts/model_ternary_i8_weights_v1.bin \
  artifacts/golden_input_fp32.bin \
  artifacts/golden_logits_fp32.bin
```

### Allocation findings

Before cleanup:

| Path | Allocations | Bytes |
|---|---:|---:|
| `forward_resnet18_scalar` | 52 | ~45.6 KB |
| `forward_resnet18_workspace_timed` | 75 | ~48.7 KB |

After cleanup:

| Path | Allocations | Bytes |
|---|---:|---:|
| `forward_resnet18_scalar` | 8 | ~5.0 KB |
| `forward_resnet18_workspace_timed` | 8 | ~5.0 KB |

The largest source was not activation tensor allocation. It was an accidental value copy:

```cpp
const BNFusedParams bn_fused = m.fused_bns.at(...);
```

`BNFusedParams` owns vectors for fused scale/bias, so this copied parameter buffers on
every convolution dispatch. Changing this to a const reference eliminated 40 allocations:

```cpp
const BNFusedParams& bn_fused = m.fused_bns.at(...);
```

Additional cleanup:

- Added workspace-backed `adaptive_avgpool_1x1_into` and `linear_into`
- Added `buf_avgpool` and `buf_fc` to `EngineWorkspace`
- Replaced `TimingEntry::name` from `std::string` with fixed-size `char[64]`/`std::array`
- Pre-reserved workspace buffers via `EngineWorkspace::reserve_max()`

The remaining 8 allocations are small dispatch/string binding artifacts, primarily from
long downsample layer names being bound to `const std::string&` parameters. Eliminating
them would require a broader heterogeneous lookup / pre-resolved layer descriptor refactor,
which is not currently worth the complexity.

---

## Current Benchmark Results

All numbers use:

```text
runs = 100
warmup = 20
batch size = 1
input = 3×224×224
```

### Current best comparison

All values are **min ms / median ms** over 100 timed runs after 20 warmup runs.
Numbers from `logs/cpp_vs_ort_20260606_183157.csv` and `logs/pytorch_fp32_20260606_183255.csv`.

| Threads | C++ engine | ORT INT8 | ORT FP32 | PyTorch FP32 |
|---:|---:|---:|---:|---:|
| 1 | 41.3 / 41.9 | 19.8 / 20.3 | 25.8 / 26.3 | 35.3 / 38.6 |
| 2 | 22.4 / 23.1 | 12.4 / 13.0 | 13.1 / 13.5 | 18.9 / 19.7 |
| 4 | 12.7 / 13.0 | 7.8 / 8.0 | 6.9 / 7.2 | 10.7 / 11.2 |
| 8 | 8.3 / 8.6 | 6.3 / 6.8 | 4.4 / 4.4 | 7.0 / 7.3 |
| 16 | 9.1 / 9.8 | 8.5 / 8.9 | 3.8 / 4.1 | 9.3 / 9.8 |
| 24 | **7.3 / 7.6** | 8.6 / 10.4 | 3.7 / 4.0 | 8.3 / 10.5† |

† PyTorch 24T mean=75 ms due to a thermal/scheduler spike during that run; median is
representative.

At 1–8 threads the C++ engine trails all three baselines. At 16–24 threads its best-observed
min beats ORT INT8 — ORT INT8 degrades past 8 threads because QDQ graph fragmentation
(QuantizeLinear/DequantizeLinear ops) does not scale well under contention. ORT FP32 is the
fastest and most stable baseline at all thread counts.

The min/median spread at 16T (9.1 / 9.8 ms) is narrower than previously observed; the
large median variance from the prior run was due to a mis-tuned benchmark that ran at a
different system load. Both metrics are consistent at 24T (7.3 / 7.6 ms).

### Optimization timeline

These values are approximate and were collected across the optimization process. They are
useful for showing progression rather than as a single controlled benchmark table.

| Stage | Approx latency | Notes |
|---|---:|---|
| Dense scalar baseline | ~2.4–2.7 s | Direct nested-loop conv, no ternary sparsity exploitation |
| Sparse ternary scalar | ~1.04 s | Positive/negative weight plans; skipped zeros |
| Padded sparse scalar | ~626 ms | Removed repeated padding boundary checks |
| AVX2 width kernel | ~309–364 ms | Vectorized output-width paths; generalized stride-1/stride-2 |
| Pattern OC-Block2/4 kernels | ~160 ms | Grouped actions by ternary patterns across output channels |
| q8 layer4 path | ~119 ms | Dynamic activation quantization + INT32 accumulation for deep blocks |
| OpenMP over OC blocks | ~29 ms at 8T | Output-channel-block parallelism across hot convs |
| Expanded q8 OCBlock2/4 + q8 1×1 | ~41–42 ms at 1T, ~8–9 ms at 8T | q8 paths extended to layer1/2/3/4 and downsample |
| Allocation cleanup | ~41 ms at 1T | Eliminated BN fused-param copies and final return-by-value ops |

---

## Architecture Overview

### Tensor layout

The engine uses NCHW tensors:

```text
data[(c * h + y) * w + x]
```

This layout makes per-output-channel writes contiguous, which is important for the current
OC-block parallel strategy. An attempted output-height tile parallel schedule regressed
because it scattered output writes across channel planes.

### Ternary weight execution

Weights are stored as `int8_t` values in `{-1, 0, +1}`.

The engine precomputes sparse plans so inner loops can iterate only over non-zero weight
positions. For multiple output channels, it groups actions by ternary pattern:

- OC-Block2: weight patterns across 2 output channels
- OC-Block4: weight patterns across 4 output channels

This reduces branching and allows AVX2 kernels to update multiple output channels or
spatial lanes efficiently.

### q8 activation path

The q8 kernels dynamically quantize FP32 activations per layer using symmetric int8
quantization, perform INT32 accumulation, and apply fused BN/output scaling back to FP32.

The current q8 family includes:

- q8 OC-Block2 spatial4 kernels for layer1/layer2/layer3/layer4 3×3 convs
- q8 OC-Block4 kernels for selected deeper layers
- q8 stride-2 3×3 kernels
- q8 1×1 stride-2 downsample kernel
- fused quantize+pad helpers with reusable workspace buffers

### Workspace execution

The fast path uses reusable workspace buffers:

- `buf_x`
- `buf_a`
- `buf_b`
- `buf_identity`
- `buf_avgpool`
- `buf_fc`
- q8 padded buffers

Residual blocks execute in-place where possible:

```text
buf_x -> conv1 -> buf_a
buf_a -> conv2 -> buf_b
identity branch -> buf_identity if needed
buf_b += identity
relu(buf_b)
swap(buf_x, buf_b)
```

This avoids most activation tensor construction and lets the timed profiling path mirror
the benchmark path.

---

## Kernel Dispatch Strategy

For a convolution, the engine tries specialized kernels first and falls back to generic
paths:

1. q8 OC-Block2 / OC-Block4 conv kernels
2. q8 1×1 stride-2 downsample kernel
3. dedicated conv1 7×7 stride-2 OC-Block4 kernel
4. FP32 pattern OC-Block2 / OC-Block4 kernels
5. generic AVX2 width kernel
6. scalar fallback

Some q8 or experimental paths are intentionally left disabled when end-to-end benchmarks
show no improvement.

---

## ONNX Runtime Profiling Findings

ORT INT8 is only partially quantized for this exported graph:

| Op type | Count/pass | Time/pass | Avg/op |
|---|---:|---:|---:|
| `Conv` FP32 | 11 | ~15.7 ms | ~1.42 ms |
| `QLinearConv` INT8 | 9 | ~5.3 ms | ~0.59 ms |
| `DequantizeLinear` | 52 | ~2.0 ms | ~0.038 ms |
| `QuantizeLinear` | 22 | ~0.54 ms | ~0.024 ms |
| Everything else | — | ~0.65 ms | — |

Key interpretation:

- ORT INT8 is not fully INT8 end-to-end.
- ORT FP32 can outperform ORT INT8 at higher thread counts because QDQ overhead and graph
  fragmentation limit INT8 scaling.
- ORT's quantized 3×3 convs are still much faster than the custom bucket kernels.

Representative comparison:

| Layer type | ORT | Custom engine |
|---|---:|---:|
| quantized 3×3 conv | ~0.52–0.64 ms | ~2–3 ms |
| 1×1 downsample | ~0.14 ms FP32 Conv | ~0.42 ms q8 1×1 |

The main reason is kernel strategy. ORT uses packed GEMM/MLAS-style kernels with high
arithmetic intensity and cache-friendly packing. The custom engine's sparse bucket dispatch
reduces ternary arithmetic but pays for indirect bucket traversal, strided activation loads,
and lower SIMD dot-product throughput.

---

## What Worked

### Sparse ternary plans

Skipping zero weights gave the first major improvement over dense scalar convolution.

### Padding outside the inner loop

Padded inputs removed repeated boundary checks from hot convolution loops.

### BN fusion

Precomputing fused BN scale/bias and applying them inside conv kernels removed separate
BatchNorm passes.

### AVX2 width vectorization

Vectorizing across output-width lanes gave a large speedup over scalar sparse conv.

### Pattern OC-Block2 / OC-Block4

Grouping actions by weight pattern across output channels reduced inner-loop branching and
enabled specialized AVX2 updates.

### q8 activation kernels

Dynamic activation quantization helped for many 3×3 layers, especially deeper layers and
later OCBlock2 kernels.

### q8 1×1 stride-2 downsample

Although slower than ORT's MLAS FP32 1×1, the custom q8 1×1 path is still much faster than
the available FP32 ternary fallback and therefore remains enabled.

### OpenMP over output-channel blocks

OC-block parallelism gave the most useful multi-thread speedup and brought the engine into
single-digit millisecond latency at 8 threads.

### Workspace-backed forward

Reusable buffers and in-place residual add/ReLU made the fast path and profiler path
consistent and greatly reduced allocation noise.

### Allocation cleanup

Changing `BNFusedParams` from value copy to const reference eliminated 40 allocations per
forward. Fixed-size timing names and final `*_into` ops made the profiled path match the
fast path.

---

## What Did Not Work / Reverted Experiments

### q8 layer3 stride-1 / stride-2 early attempts

Standalone q8 layer3 kernels were numerically reasonable and sometimes faster in isolation,
but initial end-to-end integration was neutral or marginal once quantize/pad overhead was
included. Later q8 OCBlock2 kernels for layer3 were integrated as part of the broader q8
path, but the early isolated attempts were not kept as-is.

### q8 layer2/layer1 OCBlock4 paths

q8 OCBlock4 was not a clear win for smaller layers. OCBlock2 spatial4 kernels are a better
fit for those layers.

### q8 1×1 disable

Disabling q8 1×1 downsample was tested and reverted. The fallback path was much slower.

### OH-tile parallelism

Flattening tasks over output-height tiles and OC blocks was tested and reverted. The input
reuse benefit did not compensate for scattered NCHW output writes.

### Layer4 OC-Block4 action offset precompute

Precomputing action base offsets for the layer4 OC-Block4 path was attempted but did not
produce enough benefit to justify the added complexity.

### if-constexpr / pattern-switch dispatch for q8 OCBlock4

Replacing runtime branch checks with compile-time pattern dispatch did not produce a
measurable speedup. The bottleneck appeared to be memory bandwidth / access pattern rather
than branch misprediction.

### Packed sparse activation-panel kernel (all layers, OC_TILE=1/2/4)

**Hypothesis**: pre-pack A_panel[K × N_PAD] of quantized int8 activations for N_TILE output
columns; each k position is loaded once and reused across all OC. Sparse per-OC pos/neg
index lists skip zero-weight positions. This should reduce total SIMD loads compared to the
bucket kernel's per-action input loads.

**Implementation**:

- `PackedSparseQ8Plan`: pos_idx[oc] / neg_idx[oc] vectors of uint16_t K-indices; built once
  at model load.
- Per-OH-row: quantize input → `pack_a_panel_layerN` copies N_TILE i8 values per K position
  into a contiguous panel; then inner kernel accumulates with `_mm256_cvtepi8_epi32` + add/sub.
- Layer2: N_TILE=28, N_PAD=32, A_panel = 1152 × 32 = 36 KB (fits L1)
- Layer3: N_TILE=14, N_PAD=16, A_panel = 2304 × 16 = 36 KB (fits L1)
- Layer4: N_TILE=7, N_PAD=16, A_panel = 4608 × 16 = 72 KB (exceeds L1)
- OC_TILE=2/4 variants used merged action lists + `_mm256_sign_epi32` for branch-free
  multi-OC accumulation.

**Results** (i7-12800HX, AVX2, 1T, 100 runs, min_ms):

| Layer | Bucket compute | Packed OC_TILE=1 | Speedup | Threshold |
|---|---|---|---|---|
| layer2.1.conv2 (IC=128, 28×28) | 2.50 ms | 1.97 ms | **1.27×** | ≥1.7× |
| layer3 (IC=256, 14×14) | ~2.52 ms | ~2.01 ms | **~1.25×** | ≥2.0× |
| layer4 (IC=512, 7×7) | ~1.78 ms | ~2.94 ms | **~0.60×** | ≥2.5× |
| layer3 OC_TILE=2 (sign_epi32) | ~2.52 ms | ~4.27 ms | **~0.60×** | >OC_TILE=1 |
| layer3 OC_TILE=4 (sign_epi32) | ~2.52 ms | ~4.47 ms | **~0.57×** | >OC_TILE=1 |

**Root causes**:

- *Layer4*: 72 KB panel exceeds L1 (48 KB); every k access is an L2 miss.
- *Layer3/layer2*: The bucket kernel's pattern-based grouping skips ~42% of k positions
  entirely (zero-weight patterns). The packed panel must visit every nonzero-weight k
  individually. The ~30% load reduction from panel reuse does not recover the lost
  pattern-sparsity benefit.
- *OC_TILE=2/4*: `sign_epi32` costs ~2 µops and is called 4×/8× per k for two/four OCs.
  Total µop count doubles vs OC_TILE=1 despite ~29% fewer k iterations from merging.

**Conclusion**: The Q8 OCBlock2 bucket kernel's pattern-sparsity exploitation outperforms
activation-panel packing on this architecture and layout. Not integrated. Files retained in
`quantized/` and `benchmark_packed_3x3.cpp` for reference.

### Naive conv1 OpenMP

Parallelizing conv1 with naive output-channel threading regressed, especially at high thread
counts. Conv1 remains single-threaded / specially handled rather than blindly parallelized.

---

## Memory Footprint

Generated by `scripts/report_memory_footprint.py` (run after `tools/pack_ternary_weights.py`).

| Component | Size |
|---|---:|
| ORT FP32 ONNX model (graph + external weights) | 44.72 MB |
| ORT INT8 ONNX model | 11.32 MB |
| Custom ternary model binary (INT8 weights + BN/FC FP32) | 12.68 MB |
| Conv weights, FP32 (4 B/weight, 11.2 M weights) | 42.60 MB |
| Conv weights, INT8 (1 B/weight) | 10.65 MB |
| Conv weights, 2-bit packed (ternary dense, on disk) | 2.66 MB |
| Conv weights, 1.58-bit theoretical (log₂3 bits/weight) | 2.11 MB |
| Runtime workspace buffers (reserved at init) | 6.28 MB |
| Heap allocations per inference (post-warmup) | 8 allocs / ~5 KB |

The custom binary is larger than the 2-bit theoretical minimum because it stores BN
parameters and FC weights in FP32, and uses INT8 (not packed bits) for the conv weights
to enable direct AVX2 loads without bit-unpacking overhead. At runtime, 2-bit packing
would require a bit-unpack step on the critical path, whereas the current INT8 storage
enables `_mm256_cvtepi8_epi32` directly.

The runtime workspace is pre-allocated at `EngineWorkspace` construction and never
reallocated during inference. The 8 remaining heap allocations per pass are small
string/map dispatch artifacts from downsample layer name lookups.

---

## Known Limitations

- **Still slower than ORT**: The current engine is about 2× slower than ORT INT8 at 1T and
  about 1.3× slower at 8T on this machine.
- **No GEMM / packed-matrix path**: The largest remaining gap is kernel arithmetic intensity.
  ORT's MLAS/GEMM kernels pack data and reuse it more effectively.
- **NCHW layout constraints**: NCHW works well for current OC-block writes but limits some
  spatial tiling strategies.
- **Fixed input shape**: Kernels are specialized for the 224×224 ResNet-18 shape.
- **AVX2 only**: No AVX-512/VNNI path is implemented.
- **Dynamic activation quantization overhead**: q8 helps many layers, but quantize+pad
  overhead limits benefit on smaller feature maps or low-channel layers.
- **Residual dispatch allocations remain**: 8 small heap allocations remain after warmup,
  mostly from string/map binding artifacts. They are not a current bottleneck.
- **Correctness tolerance**: q8 paths introduce numerical error relative to PyTorch FP32
  golden logits; this project focuses on inference latency and kernel design rather than
  calibrated accuracy.

---

## Project Structure

```text
.
├── CMakeLists.txt
├── models/
│   └── resnet18.py
├── tools/
│   ├── export_model.py
│   └── pack_ternary_weights.py
├── scripts/
│   ├── benchmark_cpp_vs_ort.py
│   ├── benchmark_pytorch_fp32.py
│   ├── report_memory_footprint.py
│   ├── reproduce_all.sh
│   └── bench_threads.sh
├── src/basic_engine/
│   ├── main.cpp
│   ├── resnet18_engine.cpp / .hpp
│   ├── model.hpp
│   ├── model_loader.cpp / .hpp
│   ├── tensor.hpp
│   ├── workspace.hpp
│   ├── timing.hpp
│   ├── ops.cpp / .hpp
│   ├── bn_fusion.cpp / .hpp
│   ├── conv_plan.cpp / .hpp
│   ├── conv_avx2.cpp / .hpp
│   ├── sparse_ocblock_plan.cpp / .hpp
│   ├── conv3x3_sparse_ocblock_avx2.cpp / .hpp
│   ├── pattern_ocblock2_plan.cpp / .hpp
│   ├── conv3x3_pattern_ocblock2_avx2.cpp / .hpp
│   ├── pattern_ocblock4_plan.cpp / .hpp
│   ├── conv3x3_pattern_ocblock4_avx2.cpp / .hpp
│   ├── conv1_ocblock4_plan.cpp / .hpp
│   ├── conv1_ocblock4_avx2.cpp / .hpp
│   └── quantized/
│       ├── quant_tensor.hpp
│       ├── quant_utils.cpp / .hpp
│       ├── conv3x3_q8_pattern_ocblock2_avx2.cpp / .hpp
│       ├── conv3x3_q8_pattern_ocblock4_avx2.cpp / .hpp
│       ├── conv1x1_q8_s2_avx2.cpp / .hpp
│       └── packed_q8_3x3_kernel.cpp / .hpp  (experimental; not wired into engine)
├── artifacts/
│   └── generated model/input/reference files
└── logs/
    └── benchmark outputs
```

---

## Next Steps

The current code is at the point where small dispatch/allocation cleanups are unlikely to
close the ORT gap. More meaningful future work would require a different kernel strategy:

1. The packed activation-panel approach was attempted and closed (see "What Did Not Work").
   A true GEMM-style kernel would need to pack both activations and weights into layouts with
   high arithmetic intensity — im2col + SGEMM or a hand-rolled microkernel. The ternary
   weight structure makes a standard im2col wasteful; a sparse microkernel with explicit
   register blocking over output-channel and output-width tiles would be the next avenue.
2. Explore blocked layouts that preserve output locality while improving input reuse.
3. Add an autotuning harness for tile sizes, OC block sizes, and OpenMP thresholds.
4. Investigate a VNNI / AVX-512 path on hardware that supports it.
5. Replace string-keyed runtime dispatch with pre-resolved layer descriptors if chasing
   the remaining tiny allocation overhead.
6. Calibrate q8 activation scales with real data if accuracy becomes a requirement.

---

## Reproducing Results

```bash
bash scripts/reproduce_all.sh
```

This runs all steps in order: artifact generation, 2-bit weight packing, C++ build,
single-thread smoke test, C++ vs ORT benchmark, PyTorch FP32 benchmark, and memory
footprint report. Results are written to `logs/`.

The script auto-detects `.venv/bin/python` if present. Set `PYTHON=/path/to/python` to
override.

---

## Summary

This engine started as a naive scalar ternary ResNet-18 implementation and progressively
added sparsity, padding, BN fusion, AVX2 vectorization, pattern grouping, q8 activation
kernels, OpenMP scaling, workspace execution, ORT profiling, and allocation cleanup.

The C++ runtime uses no PyTorch, ONNX Runtime, MLAS, oneDNN, or any third-party inference
library. ORT and PyTorch are used only as benchmarking references.

**Current performance** (i7-12800HX, 100 runs, batch=1):

- **1T**: C++ 41.3 ms vs ORT INT8 19.8 ms — engine is ~2× slower. The gap is kernel
  arithmetic intensity; MLAS/GEMM-style packing is out of scope for this project.
- **8T**: C++ 8.3 ms vs ORT INT8 6.3 ms — gap narrows to ~1.3×.
- **16–24T**: C++ beats ORT INT8 in best-observed min latency (7.3 ms vs 8.6 ms at 24T)
  because ORT INT8 degrades from QDQ graph fragmentation. This is a min-only result;
  ORT FP32 remains the fastest baseline at all thread counts (3.7 ms at 24T).

The final result is not an ORT replacement, but it is a useful kernel-design study:

- Demonstrates a large speedup over the original custom scalar baseline (~60× at 8T).
- Reaches single-digit millisecond latency at 8+ threads.
- Explains precisely why ORT INT8 remains faster at 1–8T.
- Documents both successful optimizations and failed hypotheses with measurements.
- Provides a clean basis for future packed/GEMM-style kernel experiments.
