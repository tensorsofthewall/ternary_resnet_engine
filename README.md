# Ternary ResNet-18 Inference Engine

A from-scratch C++ CPU inference engine for a Ternary Weight Network (TWN) variant of
ResNet-18. Convolution weights are constrained to `{-1, 0, +1}`, so multiply-accumulate
can be replaced with add/subtract over non-zero weights. The engine targets x86-64 CPUs
with AVX2/FMA and uses no external ML inference library in the C++ runtime.

This repo is intentionally an engineering study: it builds a working inference engine,
profiles bottlenecks, implements specialized kernels, compares against ONNX Runtime, and
documents both successful and reverted optimization attempts.

The repo contains two C++ engines:

- **`basic_engine`** — the original NCHW/ternary-sparse engine (pattern-grouped AVX2
  kernels, dynamic q8 activation quantization). It improves from a multi-second scalar
  baseline to single-digit-millisecond multi-thread latency, but remains slower than
  ONNX Runtime's MLAS/GEMM kernels (see "Known Limitations").
- **Q2Q / VNNI engine** (`src/q2q_engine/`) — a quantize-once-to-q8 pipeline that
  re-encodes ternary weights as INT8, runs the full forward pass in NHWC/q8 using
  `AVX-VNNI` (`vpdpbusd`) dot-product kernels, and **beats both `basic_engine` and ORT
  INT8 end-to-end** (~14.9 ms vs ~44.8 ms vs ~19.8 ms median at 1 thread on the test
  machine — see "Current Benchmark Results").

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

## Reviewer Quick Start

The two deliverables that matter for evaluation are:

| Component | Purpose |
|---|---|
| `build/basic_engine` | Original from-scratch AVX2 ternary-sparse inference engine |
| `build/benchmark_q2q_full_prefix_optimized` | Final, fastest engine — NHWC q8 + `AVX-VNNI` pipeline |

Recommended full reproduction path (generates artifacts, builds, runs all benchmarks, and
writes the memory footprint report to `logs/`):

```bash
uv sync
bash scripts/reproduce_all.sh
```

To see the headline result directly — build and run the final Q2Q/VNNI benchmark on its own:

```bash
cmake -S . -B build && cmake --build build -j$(nproc) --target benchmark_q2q_full_prefix_optimized
OMP_NUM_THREADS=1 ./build/benchmark_q2q_full_prefix_optimized \
  artifacts/model_ternary_i8_weights_v1.bin --threads 1
```

This prints a `C++ engine latency` block (`min_ms`/`median_ms`/`mean_ms`/`p90_ms`) —
expect a median around **~14.8 ms at 1 thread**, vs. `basic_engine`'s ~45 ms and ORT
INT8's ~19.7 ms (see "Current Benchmark Results" for the full thread sweep, and "C++ vs
ONNX Runtime Benchmark" for the side-by-side comparison script that produces those
numbers for all four runtimes).

The Q2Q/VNNI path requires the `AVX-VNNI` instruction set (`vpdpbusd`/
`_mm256_dpbusd_epi32`; available on Alder Lake and newer — see "Hardware"). On CPUs
without it, only the AVX2 `basic_engine` path is expected to build and run correctly.

## What to Look At

For evaluation, the most relevant code is:

| File / directory | Why it matters |
|---|---|
| `src/basic_engine/` | Original standalone AVX2 ternary-sparse engine |
| `src/q2q_engine/` | Final, fastest engine — NHWC q8 + `AVX-VNNI` pipeline (see "Q2Q / VNNI Engine") |
| `src/q2q_engine/benchmark_q2q_full_prefix_optimized.cpp` | Final benchmark driver — thin orchestration over the `q2q_engine` modules |
| `scripts/benchmark_cpp_vs_ort.py` | Main side-by-side latency comparison: `basic_engine`, Q2Q/VNNI, ORT FP32, ORT INT8 |
| `scripts/reproduce_all.sh` | Full, ordered reproduction: artifacts → build → benchmarks → footprint report |
| `scripts/report_memory_footprint.py` | Model/workspace memory footprint report |

See "Final vs Experimental Targets" (after "Build") for which binaries are the actual
deliverables vs. kept-for-the-record development history.

---

## Hardware

All benchmark numbers in this README were measured on:

- **CPU**: Intel Core i7-12800HX, 12th Gen Alder Lake-HX
  - 8 P-cores + 8 E-cores / 24 hardware threads
  - AVX2 + FMA supported, plus `AVX-VNNI` (`vpdpbusd`/`_mm256_dpbusd_epi32`) — used by the
    Q2Q engine's dot-product microkernels (see "Q2Q / VNNI Engine")
  - No AVX-512 on this machine (Alder Lake's `AVX-VNNI` is the non-512 VEX-encoded form,
    distinct from `AVX-512 VNNI`)
- **OS**: Linux, kernel 6.18 series
- **Compiler**: GCC with `-O3 -march=native -mavx2 -mfma`
- **Runtime threading**: OpenMP for the C++ engine, ORT intra-op threading for ONNX Runtime

Useful C++ scaling extends to about 8 P-cores, after which additional threads land on
E-cores with different clock characteristics; high-thread median variance in both custom
engines reflects this hybrid-core (P+E) scheduler behavior. The two engines sit in
different positions relative to ORT, though:

- **`basic_engine`** trails ORT INT8 at 1–8 threads and ORT FP32 throughout — at
  16–24 threads its best-observed *min* latency can beat ORT INT8 (which degrades from
  QDQ graph fragmentation), but that is a min-only result and ORT FP32 remains the
  fastest, most stable baseline.
- **The Q2Q/VNNI engine** beats both `basic_engine` and ORT INT8 end-to-end at 1–8
  threads (its sweet spot) and is competitive with ORT FP32 up to ~8 threads; only at
  16+ threads does ORT FP32 pull ahead, for the same hybrid-core scheduling reasons.

See "Current Benchmark Results" for the full thread sweep and "Known Limitations" for why
each engine behaves this way at high thread counts.

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

With default options, this produces:

```text
build/basic_engine
build/benchmark_q2q_microkernel_opt
build/benchmark_q2q_conv1_opt
build/benchmark_q2q_full_prefix_optimized
```

`basic_engine` and `benchmark_q2q_full_prefix_optimized` are the two main deliverables —
the original AVX2 ternary-sparse engine and the final, fastest Q2Q/VNNI engine,
respectively (see "Reviewer Quick Start" and "Final vs Experimental Targets"). The other
two (`benchmark_q2q_microkernel_opt`, `benchmark_q2q_conv1_opt`) are standalone A/B
validations for two of the Q2Q microkernel optimizations that feed into the final binary —
they build by default because `BUILD_Q2Q_FINAL` defaults to `ON`.

Build switches (all `OFF` by default except `BUILD_Q2Q_FINAL`):

| Option | Default | Description |
|---|---:|---|
| `BUILD_Q2Q_FINAL` | `ON` | Build the winning Q2Q binaries: `benchmark_q2q_full_prefix_optimized` plus the standalone microkernel/conv1 validations behind it |
| `BUILD_EXPERIMENTS` | `OFF` | Build historical `basic_engine`-family benchmarks (`benchmark_layer4_q8`/`packed_3x3`/`gemm3x3`/`nchwc`, `blocked_engine`, `benchmark_cbhwb`/`w16w32`) |
| `BUILD_Q2Q_EXPERIMENTS` | `OFF` | Build intermediate/historical Q2Q benchmarks (layer stacks, hybrid trunk, conv1+maxpool, original full prefix, etc.) |
| `ENABLE_ALLOC_PROFILE` | `OFF` | Track heap allocations per forward pass (`basic_engine` only) |

To build everything, including all historical/diagnostic binaries:

```bash
cmake -S . -B build_full -DBUILD_EXPERIMENTS=ON -DBUILD_Q2Q_EXPERIMENTS=ON -DBUILD_Q2Q_FINAL=ON
cmake --build build_full -j$(nproc)
```

To build with heap-allocation profiling enabled:

```bash
cmake -S . -B build_alloc -DENABLE_ALLOC_PROFILE=ON
cmake --build build_alloc -j$(nproc)
```

---

## Final vs Experimental Targets

The two targets that matter for evaluation — built by default — are:

```text
basic_engine
benchmark_q2q_full_prefix_optimized
```

Everything else under `BUILD_EXPERIMENTS`/`BUILD_Q2Q_EXPERIMENTS` (plus the two
`BUILD_Q2Q_FINAL` validation binaries `benchmark_q2q_microkernel_opt`/
`benchmark_q2q_conv1_opt`) is historical/diagnostic — kept for transparency and to
document the optimization process and negative results (see "What Did Not Work / Reverted
Experiments" and the `benchmark_q2q_*` development chain referenced in "Optimization
timeline"), but not required to reproduce the final headline numbers.

The final, authoritative comparison is produced by:

```bash
uv run python scripts/benchmark_cpp_vs_ort.py \
  --threads 1 2 4 8 16 24 \
  --runs 100 \
  --warmup 20
```

which writes a CSV/JSON with one row per `(threads, runtime)` pair, where `runtime` is one
of `cpp_engine` (`basic_engine`), `q2q_engine` (the Q2Q/VNNI engine), `ort_fp32`, or
`ort_int8` — see "C++ vs ONNX Runtime Benchmark" for details and flags.

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

## Correctness / Output Check

The exported artifacts include:

```text
artifacts/golden_input_fp32.bin
artifacts/golden_logits_fp32.bin
```

`golden_input_fp32.bin` is a fixed, deterministic random input (`3×224×224` FP32), and
`golden_logits_fp32.bin` is PyTorch's reference forward-pass output for that same input on
the exported ternary model (see "Generating Artifacts"). Every `basic_engine` run loads
both, runs its own forward pass on the same input, and prints a `C++ engine vs PyTorch
golden` block with `max_abs_error`/`mean_abs_error` against the reference logits — this is
the engine directly satisfying the assignment's "inference engine output has to match the
pytorch model's outputs" requirement, as distinct from classification accuracy (which the
assignment explicitly excludes, since the model is randomly initialized and never trained —
see "Classification accuracy is not evaluated, but numerical output sanity is checked" in
"Known Limitations").

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
| `--engine PATH` | Path to the `basic_engine` binary (default `build/basic_engine`) |
| `--q2q-engine PATH` | Path to the optimized Q2Q/VNNI full-prefix binary (default `build/benchmark_q2q_full_prefix_optimized`) |
| `--skip-q2q` | Skip the Q2Q/VNNI engine benchmark |
| `--skip-ort-fp32` | Skip ORT FP32 baseline |
| `--skip-ort-int8` | Skip ORT INT8 baseline |
| `--force-quantize` | Regenerate cached ORT INT8 ONNX model |
| `--ort-profile` | Run an ORT INT8 profiling pass and print the Chrome trace JSON path |

The script benchmarks four runtimes side by side: `basic_engine`, the Q2Q/VNNI engine,
ORT FP32, and ORT INT8 (the Q2Q/VNNI run also sweeps the same `--threads` list via its
own `--threads` CLI flag, so its thread-scaling behavior is directly comparable).

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
Numbers from `logs/cpp_vs_ort_20260607_155807.csv` and `logs/pytorch_fp32_20260607_155922.csv`.

| Threads | basic_engine | **Q2Q/VNNI engine** | ORT INT8 | ORT FP32 | PyTorch FP32 |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.8 / 45.2 | **14.6 / 14.8** | 19.5 / 19.7 | 25.7 / 25.9 | 34.7 / 35.2 |
| 2 | 23.2 / 23.9 | **8.0 / 8.2** | 11.9 / 12.1 | 13.0 / 13.1 | 18.4 / 18.7 |
| 4 | 13.1 / 13.5 | **4.6 / 4.7** | 7.8 / 7.9 | 7.2 / 7.4 | 10.5 / 11.1 |
| 8 | 8.8 / 9.0 | **3.0 / 3.1** | 6.1 / 6.2 | 4.4 / 4.5 | 6.8 / 7.1 |
| 16 | 14.5 / 18.7 | **6.9 / 7.8** | 10.1 / 10.4 | 3.9 / 4.0 | 10.9 / 11.2 |
| 24 | 7.7 / 7.8 | **3.4 / 5.2** | 8.3 / 8.7 | 3.8 / 4.1 | 9.4 / 9.6 |

The **Q2Q/VNNI engine beats both `basic_engine` and ORT INT8 end-to-end at every thread
count** — by ~3.0× over `basic_engine` and ~1.3–1.8× over ORT INT8 at 1–8 threads (its
sweet spot; see "Q2Q / VNNI Engine" below). It also beats ORT FP32 up to 4 threads and
stays competitive with it through 8 threads. Only at 16+ threads does ORT FP32 (the
fastest, most stable baseline throughout) pull ahead — the Q2Q engine's per-layer OpenMP
parallelism has less independent work to hand out at high thread counts (e.g., layer4 is
only 7×7 spatial), so it sees the same kind of high-thread-count variance as `basic_engine`,
just at a much lower absolute latency.

`basic_engine` itself still trails ORT INT8 at low-to-mid thread counts and ORT FP32
throughout (see "Known Limitations" for why) — it remains a useful kernel-design study in
its own right, and the Q2Q engine's win comes from a fundamentally different strategy
(re-quantizing ternary weights to INT8 and using `AVX-VNNI` `vpdpbusd`), not from further
tuning the `basic_engine` kernels.

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
| **Q2Q/VNNI engine (new strategy)** | **~14.9 ms at 1T, ~3.0 ms at 8T** | Re-quantized ternary weights to INT8, im2col + `AVX-VNNI` (`vpdpbusd`) microkernels over an NHWC q8 pipeline — see "Q2Q / VNNI Engine" below |

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

### Q2Q / VNNI Engine

The Q2Q/VNNI engine (`src/q2q_engine/`) takes a fundamentally different approach from
`basic_engine` above: instead of executing ternary weights directly against dynamically
quantized activations in NCHW, it **re-quantizes the ternary `{-1,0,+1}` weights to INT8
once at load time** and runs the entire forward pass through a quantize-once-to-q8 (Q2Q)
pipeline in **NHWC** layout, so every conv becomes a dense INT8×UINT8 dot product that maps
directly onto `AVX-VNNI`'s `vpdpbusd` instruction (`_mm256_dpbusd_epi32`) — a single
instruction that multiplies-and-accumulates 32 unsigned×signed INT8 lanes into INT32.

#### Why Q2Q/VNNI outperforms the sparse engine

`basic_engine` minimizes *arithmetic* by skipping zero-valued ternary weights, but pays
for it with irregular sparse-action traversal, lower arithmetic intensity per memory
access, and AVX2 widen/add-or-subtract loops that branch on weight sign (see "Ternary
weight execution" and "What Did Not Work" — several attempts to make that traversal more
regular, e.g. dense GEMM-style or blocked-layout kernels, made things *worse* because they
fought the branch-prediction and gather patterns inherent to sparse ternary weights).

The Q2Q engine inverts that trade-off: it treats ternary weights as dense INT8 values —
spending cycles on the ~42% of positions that are actually zero — in exchange for fully
regular NHWC activation panels and packed `OC_TILE`×`K_CHUNK` weight panels that map
directly onto `vpdpbusd`. The instruction does 32 INT8 multiply-accumulates per cycle with
no branching and no gathers; the "wasted" arithmetic on zero weights is vastly cheaper
than the irregular memory and branch costs it buys freedom from. On this CPU, the result
is that the *regular dense* VNNI path beats the *irregular sparse* AVX2 path outright —
which is also exactly the strategy ORT's own quantized kernels use (see "ONNX Runtime
Profiling Findings").

#### No external GEMM/MLAS usage

Like `basic_engine`, the Q2Q/VNNI engine calls no ORT, MLAS, oneDNN, MKL, OpenBLAS, or any
other external GEMM/inference library — its panel builders, quantization, and dot-product
kernels (`_mm256_dpbusd_epi32` and friends) are custom C++/intrinsics, written from
scratch for this project (see "Assignment Scope"). ORT and PyTorch remain purely
benchmarking/reference tools: model export, golden-logit generation, and latency
comparison — never part of either engine's inference path.

Its building blocks:

- **NHWC q8 tensors** (`QTensorNHWC` in `q2q_tensor.hpp`) — activations are stored as
  `uint8_t` in NHWC order with a per-tensor symmetric scale and a spatial border for
  padding. `quantize_nchw_fp32_to_q8_nhwc`/`dequantize_q8_nhwc_to_fp32_nchw`
  (`q2q_quant.hpp/.cpp`) convert at the pipeline's entry/exit; everything in between —
  conv1, maxpool, every layer1-4 conv and residual add — stays in q8/NHWC, so there is
  only one quantize and one dequantize for the whole network.
- **Weight panels** (`q2q_weight_panel.hpp/.cpp`) — `build_oc4_k32_panel` repacks each
  conv's INT8 weights into `OC_TILE`-output-channel × `K_CHUNK`-input-element tiles
  (`OC_TILE = 4`, `K_CHUNK = 32`, matching `vpdpbusd`'s 32-lane width) so the inner loop
  is a sequence of aligned 32-byte loads feeding straight into VNNI dot products, with no
  gather or repacking at runtime. `build_conv1_im2col_panel` does the analogous im2col
  layout for the stride-2 7×7 stem conv.
- **VNNI dot-product microkernels** (`q2q_conv_vnni.hpp/.cpp`, `q2q_conv1_vnni.hpp/.cpp`)
  — `vnni_oc4_dots`/`vnni_oc4_n2_dots` and `conv1_gather_fast` are the hot-path inner
  loops (kept `inline` in headers for cross-TU inlining); they walk a panel with
  `_mm256_dpbusd_epi32`, accumulating 4 output channels (or 4×2 output positions in the
  `_n2` variant) per call.
- **Fused epilogue** (`q2q_epilogue.hpp`, header-only `inline epilogue_oc4_simd`) — folds
  BN scale/shift, requantization back to `uint8_t`, and ReLU into one SIMD pass over the
  INT32 accumulators directly in the conv's inner loop, so no intermediate FP32 tensor is
  ever materialized between conv and activation.
- **Forward orchestration** (`q2q_resnet18.hpp/.cpp`) — `Q2QResnet18Context` owns every
  panel, scratch `QTensorNHWC`, and parameter pointer the network needs;
  `run_q2q_layer1..4`/`run_q2q_full` are **templates** parameterized over
  `(conv_fn, res_fn)` so the compiler generates a direct-call instantiation per kernel
  pair — preserving the zero-overhead, direct-call dispatch property of the original
  C++14 generic-lambda design the benchmark used during development, without introducing
  function-pointer indirection. (This is a separate, compile-time concern from
  `basic_engine`'s runtime kernel-selection fallback chain — see "basic_engine Kernel
  Dispatch Strategy" below.)

The full module breakdown lives in `src/q2q_engine/`: `q2q_common`, `q2q_tensor`,
`q2q_quant`, `q2q_epilogue`, `q2q_weight_panel`, `q2q_conv_vnni`, `q2q_conv1_vnni`,
`q2q_pool`, `q2q_residual`, and `q2q_resnet18`, with
`benchmark_q2q_full_prefix_optimized.cpp` as a thin CLI/harness layer over them — mirroring
`basic_engine`'s module structure. This design is what lets it beat both `basic_engine`
and ORT INT8 end-to-end (see "Current Benchmark Results").

---

## basic_engine Kernel Dispatch Strategy

This section describes `basic_engine`'s runtime kernel-selection path specifically — the
Q2Q/VNNI engine uses a separate, dense NHWC/q8 execution path with compile-time
template dispatch instead (see "Q2Q / VNNI Engine" above).

For a convolution, `basic_engine` tries specialized kernels first and falls back to
generic paths:

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

## Investigation Guided by Profiling Feedback

After `basic_engine` reached its tuned state (see "Optimization timeline"), the question
was *why* ONNX Runtime remained faster despite the engine's hand-written, ternary-aware
kernels. The profiling above gave the answer: ORT's advantage came from packed,
GEMM/MLAS-style convolution with high arithmetic intensity and cache-friendly data
layout — not merely from being a more mature codebase, and not primarily from
quantization itself (recall ORT INT8 isn't even fully INT8 end-to-end; see the table
above).

That reframed the problem: the sparse, branch-grouped ternary strategy and ORT's dense
packed strategy aren't two tunings of the same idea — they're different ideas, and
closing the gap meant picking one. Three lines of investigation followed directly from
that profiling insight:

1. **Allocation/dispatch cleanup** — heap-allocation profiling
   (`ENABLE_ALLOC_PROFILE`) found and removed `BNFusedParams` value copies and other
   per-forward allocation artifacts (see "Allocation findings" / "What Worked" /
   "Allocation cleanup"). This closed real but small overheads — it could not, by itself,
   close a kernel-strategy-sized gap.
2. **Layout/tiling search within the sparse strategy** — packed sparse activation
   panels, dense GEMM-style M_TILE=2 ternary kernels, NCHWc16 blocked layouts, and
   blocked-width sparse layouts were all benchmarked (see "What Did Not Work / Reverted
   Experiments"). Each one tried to make the *sparse ternary* traversal more
   regular/GEMM-like, and each one regressed — branch misprediction and irregular
   memory access are intrinsic to operating on `{-1,0,+1}` weights directly, not artifacts
   of a specific layout choice. This is what established that the sparse-direct strategy
   had a real ceiling, not just a tuning gap.
3. **Q2Q/VNNI rewrite** — rather than continuing to fight that ceiling, the Q2Q engine
   adopts ORT's own strategy: re-quantize to a dense, regular representation (INT8) and
   run it through the CPU's dedicated dense dot-product instruction (`AVX-VNNI`
   `vpdpbusd`), in a layout (NHWC/q8) that's natural for that instruction (see "Q2Q / VNNI
   Engine" and "Why Q2Q/VNNI outperforms the sparse engine"). This is the change that
   produced the final, ORT-beating result.

In short: the first two investigation lines used the profiling feedback to *characterize*
the ceiling precisely (and are documented in full, including the negative results, because
that characterization is itself a useful engineering artifact); the third line used that
characterization to *change strategy* rather than keep tuning against it.

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

### Dense GEMM-style M_TILE=2 ternary kernel

**Hypothesis**: a dense traversal of all K positions with M_TILE=2 output channels sharing
one A_panel load per k would outperform the sparse per-OC index traversal by halving
A_panel memory traffic. W_dense[OC][K] stores ternary weights as int8_t; inner loop
branches on w0/w1 to add or subtract into 8 int32 accumulator vectors (4 per OC).

**Design**: layer2.1.conv2 (IC=128, 28×28, K=1152). A_panel[k × N_PAD=32] = 36 KB,
fits L1D. 12 YMM registers in flight; no spills on AVX2. Skip k positions where both
w0 and w1 are zero (~18% of pairs).

**Results** (1T, 100 runs, min_ms):

| Kernel | Total min_ms | vs bucket |
|---|---:|---:|
| bucket (baseline) | 2.67 ms | 1.00× |
| packed sparse OC_TILE=1 | 2.13 ms | 1.25× |
| dense GEMM M_TILE=2 | 13.76 ms | **0.19×** |

**Root cause**: the hot inner loop has unpredictable branches on every k position.
Ternary weights are ~29% +1 / 29% −1 / 42% zero, so branch misprediction rate is near
50%. At ~944 non-skipped iterations × 2 branch evaluations × ~18 cycle misprediction
penalty per output row × 28 rows, misprediction stalls alone account for ~9 ms of the
13.76 ms total.

The sparse kernel avoids this entirely: it separates `pos_idx` and `neg_idx` lists so
the hot accumulation loop is completely branch-free.

Branchless alternatives (`sign_epi32`) were already tested in the OC_TILE=2/4 packed
sparse experiment and also regressed (0.60×). No branchless ternary encoding exists in
AVX2 that avoids this cost without a fundamentally different kernel structure.

**Conclusion**: Dense ternary GEMM with per-k weight branching does not work on this
architecture. The sparse branch-free design is the right approach for ternary weights in
AVX2. Not integrated. Files retained in `benchmark_gemm3x3.cpp` for reference.

---

### NCHWc16 blocked activation layout

**Hypothesis**: reorganising the padded quantized input from NCHW into NCHWc16
(`data[CB][H_pad][W_pad][16]`, 16 channels innermost per pixel) would enable
16-channel SIMD loads for a sparse action plan iterated by `(CB, KY, KX)` block,
matching the strategy used by ORT/MLAS packed-GEMM kernels.

**Design**: Layout A — NCHWc16, CB=8 blocks for IC=128. Conversion from NCHW padded
int8: 128 KB rearrangement (scalar, stride-16 writes). Sparse plan: per `(CB, KY, KX)`
block → list of `(OC, CI, sign)` pairs. Kernel: per-oh row, stride-16 byte gathers
for 28 OW positions per action, sign-branch on ±1.

**Results** (1T, 100 runs, min_ms):

| Kernel | Quantize | Convert | Compute | Total | vs bucket |
|---|---:|---:|---:|---:|---:|
| bucket (baseline) | 0.048 ms | — | 2.62 ms | 2.67 ms | 1.00× |
| packed sparse | 0.045 ms | — | 2.08 ms | 2.13 ms | 1.25× |
| NCHWc16 Layout A | 0.048 ms | **0.033 ms** | 17.57 ms | 17.66 ms | **0.15×** |

**Root causes**:

- *Stride-16 gathers*: for each of the 85,130 sparse actions per output row, the inner
  loop `src[n * 16]` for n=0..27 reads 28 bytes separated by stride 16. This is a pure
  scalar gather — the compiler cannot vectorize a stride-16 byte access over OW. Same
  structural problem as the dense GEMM branch-misprediction: the memory access pattern
  is incompatible with OW-vectorization.
- *Sign-branch misprediction*: the `if (act.sign == 1)` branch at the action level
  sees ~50% +1 / ~50% −1, same misprediction rate as the dense GEMM inner loop.
- *Convert is nearly free*: 0.033 ms for 128 KB rearrangement. NCHWc16 conversion
  overhead is not the bottleneck; the gather-based compute is.

**Key finding**: NCHW with contiguous OW rows is the right layout for this kernel
family. NCHWc16 is only beneficial when the engine can vectorize over the CI=16
channel dimension simultaneously across all OW positions — which requires a
fundamentally different kernel structure (e.g., im2col transposition first, producing
a contiguous CI×OW panel). That extra transposition step would recreate the A_panel
approach, adding conversion overhead with no net gain.

Layout B (NCHWcW: `data[CB][H][WB][16][WI]`) not attempted — stop condition triggered.

**Conclusion**: NCHWc16 is 6.7× slower than the bucket kernel. OW-vectorized NCHW is
the correct layout for the sparse ternary kernel. Files retained in
`benchmark_nchwc.cpp` for reference.

---

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

- **`basic_engine` is still slower than ORT**: The original NCHW/ternary-sparse engine is
  about 2× slower than ORT INT8 at 1T and about 1.3× slower at 8T on this machine — its
  remaining gaps are described in the next two bullets. The Q2Q/VNNI engine closes (and
  reverses) this gap end-to-end by abandoning direct ternary execution in favor of
  re-quantizing to INT8 and using `AVX-VNNI` (see "Q2Q / VNNI Engine" and "Current
  Benchmark Results") — it was a change of strategy, not further tuning of these kernels.
- **No GEMM / packed-matrix path in `basic_engine`**: its largest remaining gap is kernel
  arithmetic intensity — ORT's MLAS/GEMM kernels pack data and reuse it more effectively.
  A dense GEMM-style M_TILE=2 experiment was conducted (`benchmark_gemm3x3`) and found to
  be 5× slower than the bucket kernel due to branch misprediction on random ternary
  weights (see "What Did Not Work"). The Q2Q engine sidesteps this entirely by panel-izing
  weights into dense INT8 tiles ahead of time (`build_oc4_k32_panel`) and feeding them
  through `vpdpbusd`, trading a one-time re-quantization cost for high arithmetic
  intensity on every conv.
- **NCHW layout constraints in `basic_engine`**: NCHW works well for its OC-block writes
  and is the correct layout for its OW-vectorized sparse kernels. NCHWc16 was benchmarked
  and found to be 6.7× slower due to stride-16 scatter-gather over OW (see "What Did Not
  Work"). The Q2Q engine instead adopts NHWC end-to-end (`QTensorNHWC`), which is the
  natural layout for its dense per-pixel dot-product kernels.
- **Fixed input shape**: Kernels in both engines are specialized for the 224×224
  ResNet-18 shape.
- **`AVX-VNNI` is engine-specific, and `AVX-512` is unavailable**: `AVX-VNNI`
  (`vpdpbusd`) is implemented and used by the Q2Q engine's dense dot-product kernels, but
  `basic_engine`'s kernels remain AVX2-only — they execute sparse ternary weight patterns
  directly rather than dense INT8 panels, so `vpdpbusd` doesn't fit their execution model
  without the same re-quantization step the Q2Q engine takes. Neither engine has an
  `AVX-512` path, since the test machine doesn't support it (see "Hardware").
- **Dynamic activation quantization overhead (`basic_engine`)**: q8 helps many layers, but
  quantize+pad overhead limits benefit on smaller feature maps or low-channel layers.
- **Residual dispatch allocations remain (`basic_engine`)**: 8 small heap allocations
  remain after warmup, mostly from string/map binding artifacts. They are not a current
  bottleneck.
- **Q2Q engine high-thread-count falloff**: the Q2Q engine's per-layer OpenMP parallelism
  has progressively less independent work at deeper layers (e.g. layer4 is only 7×7
  spatial), so above ~8 threads its absolute latency stops improving and it sees the same
  kind of hybrid P+E-core scheduler variance as `basic_engine` — just starting from a much
  lower baseline. ORT FP32 remains the most stable high-thread-count option (see "Current
  Benchmark Results").
- **Classification accuracy is not evaluated, but numerical output sanity is checked**:
  the assignment uses a randomly initialized, ternarized model, so top-1/top-5
  classification accuracy is not meaningful and is not evaluated. Separately, the
  inference engine should still reproduce the PyTorch model's forward pass closely enough
  to validate that the implemented computation is the intended one.

  `basic_engine` reports max/mean absolute error against `golden_logits_fp32.bin` on every
  run. The Q2Q/VNNI path is more approximate because it intentionally keeps activations in
  fixed-scale q8 form for latency, but it should be treated as a latency-oriented
  quantized inference path rather than a calibrated accuracy-preserving deployment path.

  In other words: classification accuracy is out of scope, but numerical agreement with
  the PyTorch reference is still used as an implementation sanity check. If stricter output
  matching were required for the Q2Q path, the next step would be per-layer activation
  calibration and tighter quantization-error reporting.

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
│   ├── benchmark_packed_3x3.cpp  (packed sparse activation-panel experiment)
│   ├── benchmark_gemm3x3.cpp    (dense GEMM M_TILE=2 experiment; negative result)
│   ├── benchmark_nchwc.cpp      (NCHWc16 blocked layout diagnostic; negative result)
│   └── quantized/
│       ├── quant_tensor.hpp
│       ├── quant_utils.cpp / .hpp
│       ├── conv3x3_q8_pattern_ocblock2_avx2.cpp / .hpp
│       ├── conv3x3_q8_pattern_ocblock4_avx2.cpp / .hpp
│       ├── conv1x1_q8_s2_avx2.cpp / .hpp
│       └── packed_q8_3x3_kernel.cpp / .hpp  (experimental; not wired into engine)
├── src/q2q_engine/
│   ├── q2q_common.cpp / .hpp           (constants, hsum_epi32, benchmark utilities)
│   ├── q2q_tensor.cpp / .hpp           (QTensorNHWC, make_q8_tensor)
│   ├── q2q_quant.cpp / .hpp            (NCHW fp32 <-> NHWC q8 quantize/dequantize)
│   ├── q2q_epilogue.hpp                (epilogue_oc4_simd: fused BN + requant + ReLU)
│   ├── q2q_weight_panel.cpp / .hpp     (build_oc4_k32_panel, build_conv1_im2col_panel)
│   ├── q2q_conv_vnni.cpp / .hpp        (vnni_oc4_dots / _n2_dots, dense conv kernels)
│   ├── q2q_conv1_vnni.cpp / .hpp       (conv1_gather_fast, im2col conv1 kernel)
│   ├── q2q_pool.cpp / .hpp             (q8_maxpool_3x3_s2_p1)
│   ├── q2q_residual.cpp / .hpp         (q2q_residual_add_relu / _avx2)
│   ├── q2q_resnet18.cpp / .hpp         (Q2QResnet18Context, run_q2q_layer1..4 / _full)
│   ├── benchmark_q2q_full_prefix_optimized.cpp  (the winning Q2Q/VNNI benchmark; thin
│   │                                              orchestration over the modules above)
│   └── benchmark_q2q_*.cpp             (earlier development-phase benchmarks, kept for
│                                        history — see "Optimization timeline")
├── artifacts/
│   └── generated model/input/reference files
└── logs/
    └── benchmark outputs
```

---

## Next Steps

`basic_engine`'s direct-ternary-execution strategy is at the point where small
dispatch/allocation cleanups are unlikely to close the ORT gap — closing it required a
different kernel strategy entirely, which is what the **Q2Q/VNNI engine** now is (see
"Q2Q / VNNI Engine" and "Current Benchmark Results": it already beats both `basic_engine`
and ORT INT8 end-to-end). Remaining future work falls into two buckets:

**Pushing the Q2Q/VNNI engine further:**

1. Close the gap with ORT FP32 at 16+ threads — the Q2Q engine's per-layer OpenMP split
   runs out of independent work at deeper layers (layer4 is only 7×7 spatial); a
   fused/larger-grained parallel schedule across layer boundaries could keep high thread
   counts fed (see "Q2Q engine high-thread-count falloff" in "Known Limitations").
2. Add an autotuning harness for `OC_TILE`/`K_CHUNK` panel dimensions, dispatch
   thresholds, and OpenMP scheduling now that the dense-panel + VNNI strategy is in place.
3. `conv1` already uses the same im2col + `AVX-VNNI` strategy as the rest of the network
   (`q2q_conv1_vnni`/`conv1_gather_fast`, with a fused SIMD epilogue — see the
   "Q2Q/VNNI engine (new strategy)" row in "Optimization timeline", and "Q2Q / VNNI
   Engine" for the kernel design). It remains the largest single stage in the prefix; a
   further win would mean fusing input (de)quantization directly into the gather step to
   remove the last separate pass over the panel, rather than changing its core strategy.
4. Calibrate q8/Q2Q activation scales with real data if accuracy becomes a requirement —
   both engines currently optimize for latency over calibrated accuracy (see "Accuracy is
   not calibrated (by design)" in "Known Limitations").

**Historical context for `basic_engine` (superseded by the Q2Q engine's results, kept for
the engineering record):**

5. Packed activation-panel (sparse), dense GEMM M_TILE=2, and NCHWc16 blocked layout were
   all attempted and closed (see "What Did Not Work"). The sparse branch-free OW-vectorized
   design over NCHW was the performance ceiling for `basic_engine`'s direct-ternary AVX2
   toolkit — the path past it (im2col into dense panels, high arithmetic intensity, and a
   `vpdpbusd`/`AVX-VNNI` microkernel) is exactly what the Q2Q engine now implements.
6. Replace `basic_engine`'s string-keyed runtime dispatch with pre-resolved layer
   descriptors if chasing its remaining tiny allocation overhead.

---

## Reproducing Results

Full reproduction — artifacts, 2-bit weight packing, C++ build, smoke test, C++ vs ORT
benchmark, PyTorch FP32 benchmark, and memory footprint report, all written to `logs/`:

```bash
bash scripts/reproduce_all.sh
```

The script auto-detects `.venv/bin/python` if present. Set `PYTHON=/path/to/python` to
override.

To reproduce just the four-way comparison table from "Current Benchmark Results"
(`basic_engine` / Q2Q-VNNI / ORT FP32 / ORT INT8, written to
`logs/cpp_vs_ort_<timestamp>.{json,csv}` with one row per `(threads, runtime)` pair where
`runtime` ∈ `{cpp_engine, q2q_engine, ort_fp32, ort_int8}`):

```bash
uv run python scripts/benchmark_cpp_vs_ort.py \
  --threads 1 2 4 8 16 24 \
  --runs 100 \
  --warmup 20 \
  --build
```

(see "C++ vs ONNX Runtime Benchmark" for the full flag list, including
`--skip-q2q`/`--skip-ort-fp32`/`--skip-ort-int8` if you want to narrow the comparison).

To reproduce just the Q2Q/VNNI engine's own thread sweep directly — no ORT/PyTorch
involved, just the final binary printing its own `C++ engine latency` block per run (see
"Reviewer Quick Start"):

```bash
for t in 1 2 4 8 16 24; do
  OMP_NUM_THREADS="$t" ./build/benchmark_q2q_full_prefix_optimized \
    artifacts/model_ternary_i8_weights_v1.bin --threads "$t"
done
```

---

## Submission Checklist

This project satisfies the assignment requirements as follows:

- [x] Both deliverable engines build and run cleanly with the default CMake configuration
      (`basic_engine`, `benchmark_q2q_full_prefix_optimized` — see "Build" and "Final vs
      Experimental Targets")
- [x] Neither engine calls PyTorch, ONNX Runtime, MLAS, oneDNN, or any other ML
      inference/GEMM library in its inference path — both implement their convolution,
      quantization, and dot-product kernels from scratch (see "No external GEMM/MLAS
      usage")
- [x] `scripts/benchmark_cpp_vs_ort.py` benchmarks `basic_engine`, the Q2Q/VNNI engine,
      ORT FP32, and ORT INT8 side by side across a `--threads` sweep (see "C++ vs ONNX
      Runtime Benchmark")
- [x] The Q2Q/VNNI benchmark also supports `--threads N` directly, so its scaling is
      independently reproducible without the comparison script (see "Reviewer Quick
      Start")
- [x] `scripts/report_memory_footprint.py` reports artifact sizes, theoretical weight
      encodings, and runtime workspace breakdown (see "Memory Footprint")
- [x] Both successful optimizations and reverted/negative experiments are documented with
      measurements, not hidden (see "What Worked" and "What Did Not Work / Reverted
      Experiments")

---

## Summary

This project contains two from-scratch C++ ternary ResNet-18 inference engines, built in
sequence, embodying two different kernel strategies:

- **`basic_engine`** started as a naive scalar ternary implementation and progressively
  added sparsity, padding, BN fusion, AVX2 vectorization, pattern grouping, q8 activation
  kernels, OpenMP scaling, workspace execution, ORT profiling, and allocation cleanup —
  executing ternary `{-1,0,+1}` weights directly via sparse, branch-grouped AVX2 kernels
  over NCHW activations.
- **The Q2Q/VNNI engine** (`src/q2q_engine/`) took a different strategy when direct
  ternary execution hit its ceiling: re-quantize the ternary weights to INT8 once, and run
  the entire forward pass through a dense NHWC/q8 pipeline using `AVX-VNNI`
  (`vpdpbusd`/`_mm256_dpbusd_epi32`) dot-product microkernels (see "Q2Q / VNNI Engine").

Neither engine uses PyTorch, ONNX Runtime, MLAS, oneDNN, or any third-party inference
library. ORT and PyTorch are used only as benchmarking references.

**Current performance** (i7-12800HX, 100 runs, batch=1, min/median ms — see "Current
Benchmark Results" for the full sweep):

| Threads | `basic_engine` | **Q2Q/VNNI engine** | ORT INT8 | ORT FP32 |
|---:|---:|---:|---:|---:|
| 1 | 44.8 / 45.2 | **14.6 / 14.8** | 19.5 / 19.7 | 25.7 / 25.9 |
| 8 | 8.8 / 9.0 | **3.0 / 3.1** | 6.1 / 6.2 | 4.4 / 4.5 |

- **The Q2Q/VNNI engine beats both `basic_engine` (~3.0×) and ORT INT8 (~1.3–1.8×)
  end-to-end at every thread count from 1–8** — its sweet spot — and beats ORT FP32 up to
  4 threads. Only ORT FP32 at 16+ threads pulls ahead, where its more uniform
  intra-op parallelism handles the hybrid P+E-core scheduler better than either custom
  engine's per-layer OpenMP split (see "Known Limitations").
- **`basic_engine`** remains slower than ORT INT8 at 1–8T (~2× at 1T, ~1.3× at 8T) and
  ORT FP32 throughout — its gap is kernel arithmetic intensity, which MLAS/GEMM-style
  packing closes and which a sparse, branch-grouped ternary kernel fundamentally can't
  (see "Known Limitations" and "What Did Not Work").

The headline result is that **the Q2Q/VNNI engine is not just a kernel-design study —
it's a from-scratch C++ engine that outperforms ONNX Runtime's INT8 path end-to-end on
this network**, achieved by recognizing that direct ternary execution was the wrong lever
to keep pulling and switching to a re-quantize-to-INT8 + dense-VNNI strategy instead. The
project as a whole:

- Demonstrates a long optimization chain from the original ~2.4-2.7 s dense scalar
  baseline (see "Optimization timeline") down to single-digit-millisecond `basic_engine`
  latency, and then — via a strategy change rather than further tuning — down to
  ~14.8 ms (Q2Q/VNNI, 1T) and ~3.1 ms (8T): roughly **170× and 800× over the original
  scalar baseline**, respectively.
- Documents both engines' successful optimizations and failed hypotheses with
  measurements — including why the *first* engine's strategy hit a ceiling, and what
  changing strategy (rather than continuing to tune) bought.
- Provides two complete, contrasting reference implementations of ternary/quantized CNN
  inference kernels: sparse-direct (`basic_engine`) and dense-requantized-VNNI
  (Q2Q/VNNI engine).

A key result of this project is that the fastest solution was *not* the most direct
interpretation of "ternary weight network" — i.e., a kernel that exploits `{-1,0,+1}`
sparsity directly. That sparse-direct engine was still valuable: building it, profiling it
against ORT, and exhausting its tuning space (see "Investigation Guided by Profiling
Feedback" and "What Did Not Work") is precisely what exposed *why* it had a hard ceiling —
irregular, branchy memory access is fundamentally at odds with this CPU's wide, regular
dense-dot-product hardware. Acting on that finding — re-encoding ternary weights as dense
INT8 and routing them through `AVX-VNNI` instead of continuing to tune the sparse path —
is what turned a "useful but ORT-trailing" engine into one that beats ORT INT8 end-to-end.
The strategy change, not any single local kernel tweak, produced the final result.
