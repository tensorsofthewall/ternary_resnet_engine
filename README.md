# Ternary ResNet-18 Inference Engine

A from-scratch C++ CPU inference engine for a Ternary Weight Network (TWN) variant of
ResNet-18. Convolution weights are constrained to `{-1, 0, +1}`, so multiply-accumulate
can be replaced with add/subtract over non-zero weights. The engine targets x86-64 CPUs
with AVX2/FMA and uses no external ML inference library in the C++ runtime.

This repo is intentionally an engineering study: it builds a working inference engine,
profiles bottlenecks, implements specialized kernels, compares against ONNX Runtime, and
documents both successful and reverted optimization attempts.

The repo contains three C++ engines:

- **`basic_engine`** — the original NCHW/ternary-sparse engine (pattern-grouped AVX2
  kernels, dynamic q8 activation quantization). It improves from a multi-second scalar
  baseline to single-digit-millisecond multi-thread latency, but remains slower than
  ONNX Runtime's MLAS/GEMM kernels (see "Known Limitations").
- **Q2Q / VNNI engine** (`src/q2q_engine/`) — a quantize-once-to-q8 pipeline that
  re-encodes ternary weights as INT8, runs the full forward pass in NHWC/q8 using
  `AVX-VNNI` (`vpdpbusd`) dot-product kernels, and **beats both `basic_engine` and ORT
  INT8 end-to-end** (~14.9 ms vs ~44.8 ms vs ~19.8 ms median at 1 thread on the test
  machine — see "Current Benchmark Results").
- **AVX2 block32 engine** (`src/nhwc_block32_engine/`) — a second, AVX2-only (no
  `AVX-VNNI`) NHWC q8 pipeline using panel-im2col GEMM-style kernels
  (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`). Its final (`opt5`) variant reaches
  **~19.5 ms median at 1 thread, roughly matching ORT INT8** (~19.7 ms) without any
  VNNI instructions, and is now OpenMP-parallel for thread-scaling comparisons (see
  "Current Benchmark Results" and the "AVX2 Block32 Engine" section in
  [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)).

---

## Scope

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

The two primary deliverables that matter for evaluation are:

| Component | Purpose |
|---|---|
| `build/basic_engine` | Original from-scratch AVX2 ternary-sparse inference engine |
| `build/benchmark_q2q_full_prefix_optimized` | Final, fastest engine — NHWC q8 + `AVX-VNNI` pipeline |

A third engine is also built by default and included in the benchmark comparison:

| Component | Purpose |
|---|---|
| `build/benchmark_block32_full_model_opt5` | AVX2-only (no `AVX-VNNI`) NHWC q8 panel-GEMM engine — roughly matches ORT INT8 at 1T without VNNI (see "AVX2 Block32 Engine" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)) |

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

This prints the `C++ engine latency` block (`min_ms`/`median_ms`/`mean_ms`/`p90_ms`) —
expect a median around **~15.2 ms at 1 thread**, vs. `basic_engine`'s ~44.5 ms and ORT
INT8's ~19.7 ms (see "Current Benchmark Results" for the full thread sweep, and "C++ vs
ONNX Runtime Benchmark" for the side-by-side comparison script that produces those
numbers for all five runtimes).

The Q2Q/VNNI path requires the `AVX-VNNI` instruction set (`vpdpbusd`/
`_mm256_dpbusd_epi32`; available on Alder Lake and newer — see "Hardware"). On CPUs
without it, the AVX2 `basic_engine` and AVX2 block32 (`benchmark_block32_full_model_opt5`)
paths are still expected to build and run correctly — the block32 engine reaches the same
"roughly matches ORT INT8" result as Q2Q without requiring `AVX-VNNI` (see "AVX2 Block32
Engine" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)).

## What to Look At

For evaluation, the most relevant code is:

| File / directory | Why it matters |
|---|---|
| `src/basic_engine/` | Original standalone AVX2 ternary-sparse engine |
| `src/q2q_engine/` | Final, fastest engine — NHWC q8 + `AVX-VNNI` pipeline (see "Architecture Overview" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)) |
| `src/q2q_engine/benchmark_q2q_full_prefix_optimized.cpp` | Final benchmark driver — thin orchestration over the `q2q_engine` modules |
| `src/nhwc_block32_engine/` | AVX2-only (no `AVX-VNNI`) NHWC q8 panel-GEMM engine — roughly matches ORT INT8 at 1T (see "AVX2 Block32 Engine" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)) |
| `scripts/benchmark_cpp_vs_ort.py` | Main side-by-side latency comparison: `basic_engine`, Q2Q/VNNI, AVX2 block32, ORT FP32, ORT INT8 |
| `scripts/reproduce_all.sh` | Full, ordered reproduction: artifacts → build → benchmarks → footprint report |
| `scripts/report_memory_footprint.py` | Model/workspace memory footprint report |
| [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) | Kernel-level architecture, ONNX Runtime profiling findings, the investigation that motivated the Q2Q/VNNI rewrite, and the full record of what worked and didn't (with measurements) |

See "Final vs Experimental Targets" (after "Build") for which binaries are the actual
deliverables vs. kept-for-the-record development history. See
[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) for the engineering deep-dive behind these
results.

---

## Hardware

All benchmark numbers in this README were measured on:

- **CPU**: Intel Core i7-12800HX, 12th Gen Alder Lake-HX
  - 8 P-cores + 8 E-cores / 24 hardware threads
  - AVX2 + FMA supported, plus `AVX-VNNI` (`vpdpbusd`/`_mm256_dpbusd_epi32`) — used by the
    Q2Q engine's dot-product microkernels (see "Architecture Overview" in
    [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md))
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
document the optimization process and negative results (see "What Did Not Work /
Reverted Experiments" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) and the
`benchmark_q2q_*` development chain referenced in "Optimization
timeline"), but not required to reproduce the final headline numbers.

The final, authoritative comparison is produced by:

```bash
uv run python scripts/benchmark_cpp_vs_ort.py \
  --threads 1 2 4 8 16 24 \
  --runs 100 \
  --warmup 20
```

which writes a CSV/JSON with one row per `(threads, runtime)` pair, where `runtime` is one
of `cpp_engine` (`basic_engine`), `q2q_engine` (the Q2Q/VNNI engine), `block32_engine`
(the AVX2 block32 opt5 engine), `ort_fp32`, or `ort_int8` — see "C++ vs ONNX Runtime
Benchmark" for details and flags.

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
the engine directly satisfying the project's "inference engine output has to match the
pytorch model's outputs" requirement, as distinct from classification accuracy (which the
scope explicitly excludes, since the model is randomly initialized and never trained —
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
| `--block32-engine PATH` | Path to the AVX2 block32 opt5 full-model binary (default `build/benchmark_block32_full_model_opt5`) |
| `--skip-q2q` | Skip the Q2Q/VNNI engine benchmark |
| `--skip-block32` | Skip the AVX2 block32 opt5 engine benchmark |
| `--skip-ort-fp32` | Skip ORT FP32 baseline |
| `--skip-ort-int8` | Skip ORT INT8 baseline |
| `--force-quantize` | Regenerate cached ORT INT8 ONNX model |
| `--ort-profile` | Run an ORT INT8 profiling pass and print the Chrome trace JSON path |

The script benchmarks five runtimes side by side: `basic_engine`, the Q2Q/VNNI engine,
the AVX2 block32 (opt5) engine, ORT FP32, and ORT INT8. The Q2Q/VNNI run also sweeps the
same `--threads` list via its own `--threads` CLI flag; the block32 engine is threaded
purely via `OMP_NUM_THREADS` (like `basic_engine`, no CLI flag needed) — so all three
C++ engines' thread-scaling behavior is directly comparable.

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
Numbers from `logs/cpp_vs_ort_20260612_025901.csv` and `logs/pytorch_fp32_20260612_030028.csv`.
The exact numbers may vary with CPU governor, thermals, and background load; the reported table was collected from the log files listed above on the stated i7-12800HX system.

| Threads | basic_engine | **Q2Q/VNNI engine** | AVX2 block32 (opt5) | ORT INT8 | ORT FP32 | PyTorch FP32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.1 / 44.5 | **15.0 / 15.2** | 19.5 / 20.0 | 19.6 / 19.7 | 25.6 / 25.8 | 34.7 / 35.2 |
| 2 | 23.3 / 24.0 | **8.1 / 8.4** | 10.4 / 10.6 | 11.9 / 12.0 | 12.9 / 13.1 | 18.7 / 18.9 |
| 4 | 13.1 / 13.4 | **4.6 / 4.7** | 5.9 / 6.0 | 7.6 / 7.8 | 7.3 / 7.5 | 11.0 / 11.3 |
| 8 | 8.6 / 8.7 | **3.1 / 3.2** | 3.8 / 4.0 | 6.1 / 6.3 | 4.3 / 4.5 | 6.9 / 7.0 |
| 16 | 10.9 / 21.0 | **7.2 / 8.4** | 8.3 / 9.7 | 8.5 / 8.8 | 4.2 / 4.3 | 9.0 / 9.1 |
| 24 | 7.7 / 7.8 | **3.9 / 4.3** | 6.6 / 10.5 | 8.3 / 8.6 | 3.6 / 3.9 | 7.7 / 7.8 |

The **Q2Q/VNNI engine beats `basic_engine`, the AVX2 block32 engine, and ORT INT8
end-to-end at every thread count** — by ~2.9× over `basic_engine` and ~1.3–2.0× over ORT
INT8 at 1–8 threads (its sweet spot; see "Architecture Overview" in
[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)). It also beats ORT FP32 up to 4 threads and
stays competitive with it through 8 threads. Only at 16+ threads does ORT FP32 (the
fastest, most stable baseline throughout) pull ahead — the Q2Q engine's per-layer OpenMP
parallelism has less independent work to hand out at high thread counts (e.g., layer4 is
only 7×7 spatial), so it sees the same kind of high-thread-count variance as `basic_engine`,
just at a much lower absolute latency.

The **AVX2 block32 (opt5) engine roughly matches ORT INT8 at 1 thread** (~19.5–20.0 ms for
both — essentially a tie within run-to-run noise) **without using any `AVX-VNNI`
instructions**, using the same panel-im2col GEMM strategy as Q2Q but built on
`_mm256_maddubs_epi16`/`_mm256_madd_epi16` instead of `vpdpbusd` (see "AVX2 Block32
Engine" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)). It scales to ~3.8–4.0 ms at 8
threads (its sweet spot, ~4.9–5.1× over 1T) but, like both other engines, degrades at
16+ threads under this CPU's hybrid P+E-core scheduling — more sharply than Q2Q, since
its per-thread scratch-panel overhead is relatively larger at the smaller per-row work
units available at high thread counts.

`basic_engine` itself still trails ORT INT8 at low-to-mid thread counts and ORT FP32
throughout (see "Known Limitations" for why) — it remains a useful kernel-design study in
its own right, and both the Q2Q and block32 engines' wins come from a fundamentally
different strategy (re-quantizing ternary weights to INT8 and running dense NHWC/q8 GEMM
kernels), not from further tuning the `basic_engine` kernels.

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
| **Q2Q/VNNI engine (new strategy)** | **~14.9 ms at 1T, ~3.0 ms at 8T** | Re-quantized ternary weights to INT8, im2col + `AVX-VNNI` (`vpdpbusd`) microkernels over an NHWC q8 pipeline — see "Architecture Overview" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) |

---

## Architecture, Profiling, and Optimization History

The kernel-level architecture of all three engines (tensor layouts, ternary execution,
the q8, Q2Q/VNNI, and AVX2 block32 pipelines), the ONNX Runtime profiling that motivated
the strategy change, the investigation that connected that profiling to the eventual
rewrite, and the full record of what worked and what didn't — including every reverted
experiment with measurements — now live in **[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)**.

That document covers, in order:

- **Architecture Overview** — tensor layout, ternary weight execution, the q8 activation
  path, workspace execution, the Q2Q/VNNI engine's design (NHWC q8 tensors, weight
  panels, VNNI microkernels, fused epilogue, forward orchestration), and the **AVX2
  Block32 Engine** (the same dense-panel-GEMM strategy built on AVX2-only `maddubs`/`madd`
  instead of `vpdpbusd`, its opt3→opt4→opt5 progression, and its OpenMP thread-scaling
  results) — plus *why* the panel-GEMM engines outperform the sparse engine and
  confirmation that none of the three engines use external GEMM/MLAS libraries
- **basic_engine Kernel Dispatch Strategy** — its runtime kernel-selection fallback chain
- **ONNX Runtime Profiling Findings** — the profiling data that exposed *why* ORT was
  faster (packed GEMM/MLAS-style kernels, not just quantization)
- **Investigation Guided by Profiling Feedback** — how that profiling insight led through
  allocation cleanup and layout/tiling experiments to the Q2Q/VNNI rewrite
- **What Worked** — sparse ternary plans, padding, BN fusion, AVX2 vectorization, pattern
  grouping, q8 kernels, OpenMP, workspace execution, allocation cleanup
- **What Did Not Work / Reverted Experiments** — every negative result with measurements:
  packed sparse activation panels, dense GEMM-style ternary kernels, NCHWc16 blocked
  layouts, naive conv1 threading, and more

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
  re-quantizing to INT8 and using `AVX-VNNI` (see "Architecture Overview" in
  [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) and "Current Benchmark Results" below) —
  it was a change of strategy, not further tuning of these kernels.
- **No GEMM / packed-matrix path in `basic_engine`**: its largest remaining gap is kernel
  arithmetic intensity — ORT's MLAS/GEMM kernels pack data and reuse it more effectively.
  A dense GEMM-style M_TILE=2 experiment was conducted (`benchmark_gemm3x3`) and found to
  be 5× slower than the bucket kernel due to branch misprediction on random ternary
  weights (see "What Did Not Work / Reverted Experiments" in
  [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)). The Q2Q engine sidesteps this entirely by panel-izing
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
  the project uses a randomly initialized, ternarized model, so top-1/top-5
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
├── src/nhwc_block32_engine/
│   ├── nhwc_common.cpp / .hpp          (BLOCK32=32 NHWC q8 helpers, epilogue_oc4_avx2)
│   ├── nhwc_q8_dot_avx2.hpp            (AVX2-only dot-product primitives: maddubs+madd)
│   ├── block32_plan.cpp / .hpp         (per-layer Block32ConvPlan construction)
│   ├── im2col_gemm.cpp / .hpp          (panel-im2col GEMM OC4 kernels, stride-1/stride-2,
│   │                                    OpenMP-parallel over output rows)
│   ├── block32_conv1.cpp / block32_conv1_opt.cpp / .hpp  (conv1 gather + panel-GEMM N16)
│   ├── block32_conv_avx2.cpp / .hpp    (OC4 block32 conv, OpenMP-parallel over OC tiles)
│   ├── block32_layer1_opt.cpp / block32_layer234_opt.cpp / .hpp  (earlier specialized
│   │                                    layer kernels, superseded by panel-im2col in opt4/5)
│   ├── block32_residual.cpp / .hpp     (residual add+ReLU, OpenMP-parallel over rows)
│   ├── block32_resnet18.cpp / .hpp     (forward orchestration)
│   └── benchmark_block32_full_model_opt5.cpp  (the final AVX2-only block32 benchmark;
│                                        see "AVX2 Block32 Engine" in ENGINEERING_NOTES.md)
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
"Architecture Overview" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md) and "Current
Benchmark Results" above: it already beats both `basic_engine` and ORT INT8 end-to-end).
Remaining future work falls into two buckets:

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
   all attempted and closed (see "What Did Not Work / Reverted Experiments" in
   [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)). The sparse branch-free OW-vectorized
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

To reproduce just the five-way comparison table from "Current Benchmark Results"
(`basic_engine` / Q2Q-VNNI / AVX2 block32 opt5 / ORT FP32 / ORT INT8, written to
`logs/cpp_vs_ort_<timestamp>.{json,csv}` with one row per `(threads, runtime)` pair where
`runtime` ∈ `{cpp_engine, q2q_engine, block32_engine, ort_fp32, ort_int8}`):

```bash
uv run python scripts/benchmark_cpp_vs_ort.py \
  --threads 1 2 4 8 16 24 \
  --runs 100 \
  --warmup 20 \
  --build
```

(see "C++ vs ONNX Runtime Benchmark" for the full flag list, including
`--skip-q2q`/`--skip-block32`/`--skip-ort-fp32`/`--skip-ort-int8` if you want to narrow
the comparison).

To reproduce just the Q2Q/VNNI engine's own thread sweep directly — no ORT/PyTorch
involved, just the final binary printing its own `C++ engine latency` block per run (see
"Reviewer Quick Start"):

```bash
for t in 1 2 4 8 16 24; do
  OMP_NUM_THREADS="$t" ./build/benchmark_q2q_full_prefix_optimized \
    artifacts/model_ternary_i8_weights_v1.bin --threads "$t"
done
```

Similarly, for the AVX2 block32 (opt5) engine's own thread sweep (threaded purely via
`OMP_NUM_THREADS`, no `--threads` CLI flag):

```bash
for t in 1 2 4 8 16 24; do
  OMP_NUM_THREADS="$t" OMP_PROC_BIND=close OMP_PLACES=cores \
    ./build/benchmark_block32_full_model_opt5 artifacts/model_ternary_i8_weights_v1.bin
done
```

---

## Submission Checklist

This project satisfies the requirements as follows:

- [x] Both deliverable engines build and run cleanly with the default CMake configuration
      (`basic_engine`, `benchmark_q2q_full_prefix_optimized` — see "Build" and "Final vs
      Experimental Targets")
- [x] Neither engine calls PyTorch, ONNX Runtime, MLAS, oneDNN, or any other ML
      inference/GEMM library in its inference path — both implement their convolution,
      quantization, and dot-product kernels from scratch (see "No external GEMM/MLAS
      usage" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md))
- [x] `scripts/benchmark_cpp_vs_ort.py` benchmarks `basic_engine`, the Q2Q/VNNI engine,
      the AVX2 block32 opt5 engine, ORT FP32, and ORT INT8 side by side across a
      `--threads` sweep (see "C++ vs ONNX Runtime Benchmark")
- [x] The Q2Q/VNNI benchmark also supports `--threads N` directly, so its scaling is
      independently reproducible without the comparison script (see "Reviewer Quick
      Start")
- [x] `scripts/report_memory_footprint.py` reports artifact sizes, theoretical weight
      encodings, and runtime workspace breakdown (see "Memory Footprint")
- [x] Both successful optimizations and reverted/negative experiments are documented with
      measurements, not hidden (see "What Worked" and "What Did Not Work / Reverted
      Experiments" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md))

---

## Summary

This project contains three from-scratch C++ ternary ResNet-18 inference engines, built
in sequence, embodying three different kernel strategies:

- **`basic_engine`** started as a naive scalar ternary implementation and progressively
  added sparsity, padding, BN fusion, AVX2 vectorization, pattern grouping, q8 activation
  kernels, OpenMP scaling, workspace execution, ORT profiling, and allocation cleanup —
  executing ternary `{-1,0,+1}` weights directly via sparse, branch-grouped AVX2 kernels
  over NCHW activations.
- **The Q2Q/VNNI engine** (`src/q2q_engine/`) took a different strategy when direct
  ternary execution hit its ceiling: re-quantize the ternary weights to INT8 once, and run
  the entire forward pass through a dense NHWC/q8 pipeline using `AVX-VNNI`
  (`vpdpbusd`/`_mm256_dpbusd_epi32`) dot-product microkernels (see "Architecture
Overview" in [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)).
- **The AVX2 block32 engine** (`src/nhwc_block32_engine/`) asks whether the Q2Q strategy's
  win requires `AVX-VNNI` at all: it applies the same re-quantize-to-INT8,
  dense-NHWC/q8-panel-GEMM strategy, but built entirely from `AVX2`/`FMA`
  (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`) — no VNNI instructions — and its final
  `opt5` variant still roughly matches ORT INT8 at 1 thread (see "AVX2 Block32 Engine" in
  [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)).

None of the engines use PyTorch, ONNX Runtime, MLAS, oneDNN, or any third-party inference
library. ORT and PyTorch are used only as benchmarking references.

**Current performance** (i7-12800HX, 100 runs, batch=1, min/median ms — see "Current
Benchmark Results" for the full sweep):

| Threads | `basic_engine` | **Q2Q/VNNI engine** | AVX2 block32 (opt5) | ORT INT8 | ORT FP32 |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.1 / 44.5 | **15.0 / 15.2** | 19.5 / 20.0 | 19.6 / 19.7 | 25.6 / 25.8 |
| 8 | 8.6 / 8.7 | **3.1 / 3.2** | 3.8 / 4.0 | 6.1 / 6.3 | 4.3 / 4.5 |

- **The Q2Q/VNNI engine beats `basic_engine` (~2.9×), the AVX2 block32 engine
  (~1.3–1.2×), and ORT INT8 (~1.3–2.0×) end-to-end at every thread count from 1–8** — its
  sweet spot — and beats ORT FP32 up to 4 threads. Only ORT FP32 at 16+ threads pulls
  ahead, where its more uniform intra-op parallelism handles the hybrid P+E-core
  scheduler better than any of the three custom engines' per-layer/per-row OpenMP splits
  (see "Known Limitations").
- **The AVX2 block32 engine roughly ties ORT INT8 at 1T** (~19.5–20.0 ms for both) using
  no `AVX-VNNI` — proof that the dense-panel-GEMM strategy itself, not `vpdpbusd`
  specifically, is what closes most of the gap to ORT; `AVX-VNNI` is what then pushes Q2Q
  decisively past ORT INT8.
- **`basic_engine`** remains slower than ORT INT8 at 1–8T (~2.2× at 1T, ~1.3× at 8T) and
  ORT FP32 throughout — its gap is kernel arithmetic intensity, which MLAS/GEMM-style
  packing closes and which a sparse, branch-grouped ternary kernel fundamentally can't
  (see "Known Limitations" above and "What Did Not Work / Reverted Experiments" in
  [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)).

The headline result is that **the Q2Q/VNNI engine is not just a kernel-design study —
it's a from-scratch C++ engine that outperforms ONNX Runtime's INT8 path end-to-end on
this network**, achieved by recognizing that direct ternary execution was the wrong lever
to keep pulling and switching to a re-quantize-to-INT8 + dense-panel-GEMM strategy
instead — and the AVX2 block32 engine shows that strategy alone (even without
`AVX-VNNI`) already gets to within noise of ORT INT8. The project as a whole:

- Demonstrates a long optimization chain from the original ~2.4-2.7 s dense scalar
  baseline (see "Optimization timeline") down to single-digit-millisecond `basic_engine`
  latency, and then — via a strategy change rather than further tuning — down to
  ~15.2 ms (Q2Q/VNNI, 1T) / ~3.2 ms (8T) and ~20.0 ms (AVX2 block32, 1T) / ~4.0 ms (8T):
  roughly **170× and 800× over the original scalar baseline** for Q2Q/VNNI, respectively.
- Documents all three engines' successful optimizations and failed hypotheses with
  measurements — including why the *first* engine's strategy hit a ceiling, and what
  changing strategy (rather than continuing to tune) bought.
- Provides three complete, contrasting reference implementations of ternary/quantized CNN
  inference kernels: sparse-direct (`basic_engine`), dense-requantized-VNNI (Q2Q/VNNI
  engine), and dense-requantized-AVX2-only (block32 engine).

A key result of this project is that the fastest solution was *not* the most direct
interpretation of "ternary weight network" — i.e., a kernel that exploits `{-1,0,+1}`
sparsity directly. That sparse-direct engine was still valuable: building it, profiling it
against ORT, and exhausting its tuning space (see "Investigation Guided by Profiling
Feedback" and "What Did Not Work / Reverted Experiments" in
[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)) is precisely what exposed *why* it had a hard ceiling —
irregular, branchy memory access is fundamentally at odds with this CPU's wide, regular
dense-dot-product hardware. Acting on that finding — re-encoding ternary weights as dense
INT8 and routing them through `AVX-VNNI` instead of continuing to tune the sparse path —
is what turned a "useful but ORT-trailing" engine into one that beats ORT INT8 end-to-end.
The strategy change, not any single local kernel tweak, produced the final result.
