# Engineering Notes: Architecture, Profiling, and Optimization History

This document is the engineering deep-dive that backs the headline results in
[README.md](README.md) — the kernel-level architecture of all three engines, the ONNX Runtime
profiling that motivated the strategy change, the reasoning that connects profiling
findings to the investigation that followed, and the full record of what worked and what
didn't (including negative results with measurements).

It's split out from `README.md` so that the reviewer-facing document (quick start,
build/run/reproduce instructions, results tables, known limitations, summary) stays short
and scannable, while this document preserves the full narrative for anyone who wants to
go deeper into *why* the engines are built the way they are. See "What to Look At" in
`README.md` for the suggested reading order.

---

## Architecture Overview

### `basic_engine` tensor layout

This subsection describes the original `basic_engine`. The Q2Q/VNNI and AVX2 block32 engines use NHWC/q8 tensor layouts described in their own sections below.

`basic_engine` uses NCHW tensors:

```text
data[(c * h + y) * w + x]
```

This layout makes per-output-channel writes contiguous, which is important for the current
OC-block parallel strategy. An attempted output-height tile parallel schedule regressed
because it scattered output writes across channel planes.

### `basic_engine` ternary weight execution

In `basic_engine`, weights are stored as `int8_t` values in `{-1, 0, +1}` and executed directly as ternary sparse/pattern actions.

The engine precomputes sparse plans so inner loops can iterate only over non-zero weight
positions. For multiple output channels, it groups actions by ternary pattern:

- OC-Block2: weight patterns across 2 output channels
- OC-Block4: weight patterns across 4 output channels

This reduces branching and allows AVX2 kernels to update multiple output channels or
spatial lanes efficiently.

### `basic_engine` q8 activation path

The `basic_engine` q8 kernels dynamically quantize FP32 activations per layer using symmetric int8
quantization, perform INT32 accumulation, and apply fused BN/output scaling back to FP32.

The current q8 family includes:

- q8 OC-Block2 spatial4 kernels for layer1/layer2/layer3/layer4 3×3 convs
- q8 OC-Block4 kernels for selected deeper layers
- q8 stride-2 3×3 kernels
- q8 1×1 stride-2 downsample kernel
- fused quantize+pad helpers with reusable workspace buffers

### `basic_engine` workspace execution

The `basic_engine` fast path uses reusable workspace buffers:

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
scratch for this project (see "Scope" in README.md). ORT and PyTorch remain purely
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
`basic_engine`'s module structure. This design is the fastest path in the repo and beats
both `basic_engine` and ORT INT8 end-to-end (see "Current Benchmark Results" in
README.md). The AVX2 block32 engine below asks a related but distinct question: how much of
that win comes from the dense-panel strategy itself rather than from `AVX-VNNI` specifically.

### AVX2 Block32 Engine (opt5, AVX2-only)

The AVX2 block32 engine (`src/nhwc_block32_engine/`) is a third engine that asks a
narrower question than the Q2Q/VNNI rewrite: **does the win over ORT INT8 require
`AVX-VNNI`, or does the dense-NHWC/q8-panel-GEMM *strategy* alone get most of the way
there on plain `AVX2`/`FMA`?**

It adopts the same high-level strategy as Q2Q — re-quantize ternary weights to INT8 once,
run the forward pass through a quantize-once-to-q8 NHWC pipeline, and use panel-style
weight layouts for high arithmetic intensity — but its dot-product microkernels are built
entirely from two-instruction AVX2 sequences instead of `vpdpbusd`:

```cpp
// AVX2-only INT8 dot product (nhwc_q8_dot_avx2.hpp): the non-VNNI equivalent
// of a single vpdpbusd. Two instructions instead of one, no AVX-512/VNNI required.
__m256i prod16 = _mm256_maddubs_epi16(activations_u8, weights_i8); // u8*i8 -> i16, pairwise add
acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod16, ones16)); // i16 -> i32, pairwise add
```

Activations are stored in `BLOCK32`-channel NHWC q8 tensors (`q2q::QTensorNHWC`, reused
from the Q2Q engine — `nhwc_common.hpp` adds the AVX2-only epilogue
`epilogue_oc4_avx2`), and weights are packed into dense `OC4`-output-channel ×
32-input-channel blocks (`Block32ConvPlan`, `block32_plan.cpp`) — directly analogous to
Q2Q's `OC_TILE=4`/`K_CHUNK=32` panels, just consumed by the `maddubs`/`madd` pair above
instead of `vpdpbusd`.

#### Optimization progression (opt3 -> opt4 -> opt5)

| Variant | Full-model median (1T) | What changed |
|---|---:|---|
| opt3 (Phase 14 final) | ~29.19 ms | Specialized per-layer OC4 block32 kernels (`block32_conv_avx2`, `block32_layer1_opt`/`block32_layer234_opt`), gather-based conv1 |
| opt4 (Phase 15 final) | ~24.01 ms | Replaced the repeated 3x3/s1/p1 convs in layer2/3/4 with **panel-im2col GEMM OC4** kernels (`im2col_gemm.cpp`, `panel_im2col_gemm_oc4_n2_avx2`) — gathers each output pixel's receptive field into a panel once, then runs an OC-tile-major GEMM over it, amortizing the gather cost across all 4 output channels |
| opt5 (Phase 16 final) | **~19.4-20.0 ms** | Extended panel-im2col GEMM to layer1's 3x3/s1/p1 convs and to the stride-2 transition convs (`panel_im2col_gemm_oc4_stride2_avx2`), and added a panel-GEMM N16 variant for conv1's 7x7/s2 stem (`conv1_panel_gemm_oc4_n16_avx2`) |

Every step preserves **exact q8 correctness** (0/0 mismatches against the scalar
reference at every stage of every variant) — these are pure kernel-strategy changes, not
approximations.

opt5's ~19.4-20.0 ms median at 1 thread lands within run-to-run noise of ORT INT8's
~19.6-19.7 ms on this machine (see "Current Benchmark Results" in README.md) — i.e. the
panel-GEMM strategy alone, without `AVX-VNNI`, already closes essentially the entire gap
to ORT INT8 that `basic_engine`'s sparse-direct kernels could not. `AVX-VNNI` (Q2Q) is
what then pushes decisively *past* ORT INT8 (~15 ms), by halving the two-instruction
`maddubs`+`madd` sequence above into a single `vpdpbusd`.

#### Multi-threading and thread scaling

opt5's hot kernels are OpenMP-parallel (`#pragma omp parallel for schedule(static)`,
following the same precedent as `q2q::q8_maxpool_3x3_s2_p1`):

- `panel_im2col_gemm_oc4_avx2_impl`/`_stride2_avx2_impl` (`im2col_gemm.cpp`) — parallel
  over output rows, with a **per-thread** `AlignedBuffer32<uint8_t> panel` scratch buffer
  (each thread reads an overlapping, read-only band of the input and writes a disjoint
  output row — no races)
- `conv1_panel_gemm_oc4_avx2_impl` (`block32_conv1_opt.cpp`) — same per-thread-panel
  pattern for the conv1 stem
- `conv_block32_oc4_avx2` (`block32_conv_avx2.cpp`) — parallel over output-channel tiles
  (used for downsample convs)
- `block32_residual_add_relu_avx2` (`block32_residual.cpp`) — parallel over output rows,
  no scratch buffers (exact analog of `q8_maxpool_3x3_s2_p1`)

Threaded purely via `OMP_NUM_THREADS` (no `--threads` CLI flag), like `basic_engine`.
Correctness is **0/0 mismatches at every thread count** (1/2/4/8/16/24 verified) — the
per-thread scratch buffers eliminate the only potential race.

Thread-scaling (`total_profiled_ms` median, 100 runs, `OMP_PROC_BIND=close
OMP_PLACES=cores`):

| Threads | median (ms) | speedup vs 1T |
|---:|---:|---:|
| 1  | 19.66 | 1.00x |
| 2  | 11.16 | 1.76x |
| 4  | 6.37  | 3.09x |
| 8  | 4.09  | **4.81x** |
| 16 | 13.89 | 1.42x |
| 24 | 16.26 | 1.21x |

Scaling peaks at **8 threads** (this CPU's 8 P-cores) — beyond that, threads land on
hyperthreads/E-cores and AVX2 throughput degrades sharply, the same hybrid P+E-core
effect documented for `basic_engine` and Q2Q in "Known Limitations" in README.md, but
more pronounced here: block32's per-thread scratch-panel allocation and smaller per-row
work units amortize less well once each thread's share of a row shrinks at high thread
counts.

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

Representative comparison for the tuned `basic_engine` sparse/q8 path before the Q2Q and block32 rewrites:

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

After `basic_engine` reached its tuned state (see "Optimization timeline" in
README.md), the question
was *why* ONNX Runtime remained faster despite the engine's hand-written, ternary-aware
kernels. The profiling above gave the answer: ORT's advantage came from packed,
GEMM/MLAS-style convolution with high arithmetic intensity and cache-friendly data
layout — not merely from being a more mature codebase, and not primarily from
quantization itself (recall ORT INT8 isn't even fully INT8 end-to-end; see the table
above).

That reframed the problem: the sparse, branch-grouped ternary strategy and ORT's dense
packed strategy aren't two tunings of the same idea — they're different ideas, and
closing the gap meant picking one. Four lines of investigation followed directly from
that profiling insight:

1. **Allocation/dispatch cleanup** — heap-allocation profiling
   (`ENABLE_ALLOC_PROFILE`) found and removed `BNFusedParams` value copies and other
   per-forward allocation artifacts (see "Allocation findings" in README.md / "What
   Worked" / "Allocation cleanup" below). This closed real but small overheads — it could not, by itself,
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
   Engine" and "Why Q2Q/VNNI outperforms the sparse engine"). This produced the fastest
   engine in the repo and beats ORT INT8 decisively on this machine.
4. **AVX2 block32 validation of the same strategy** — to separate *strategy* from
   *instruction-set advantage*, the block32 engine repeats the dense NHWC/q8 panel-GEMM
   design using only AVX2 `maddubs`+`madd` dot products. It does not reach Q2Q/VNNI's
   latency, but it closes essentially the whole ORT INT8 gap without `AVX-VNNI`, showing
   that dense panel-GEMM layout was the decisive architectural shift while `vpdpbusd`
   provides the final speedup.
  
I also treated `bitnet.cpp` as a strategy reference for low-bit kernel design — primarily the idea that low-bit models benefit from explicit packing/layout and custom kernels — but did not reuse or depend on any of its code.

In short: the first two investigation lines used the profiling feedback to *characterize*
the ceiling precisely (and are documented in full, including the negative results, because
that characterization is itself a useful engineering artifact); the Q2Q/VNNI and AVX2
block32 lines used that characterization to *change strategy* rather than keep tuning
against it.

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


### Dense NHWC/q8 panel strategy

The Q2Q/VNNI and AVX2 block32 engines both switch away from direct sparse ternary
execution and instead run dense INT8-style dot products over NHWC/q8 activations. This
regularizes memory access, increases arithmetic intensity, and makes the hot loops look
like small packed GEMM kernels rather than sparse bucket traversals.

### AVX-VNNI dot-product kernels

In the Q2Q engine, `_mm256_dpbusd_epi32` collapses the unsigned-activation × signed-weight
INT8 dot product into one dense instruction over 32 lanes. This is the fastest path in the
repo and is what pushes the dense-panel strategy decisively past ORT INT8.

### AVX2 block32 dot-product kernels

The AVX2 block32 engine implements the same dense-panel strategy without `AVX-VNNI`,
using `_mm256_maddubs_epi16` followed by `_mm256_madd_epi16`. It is slower than Q2Q/VNNI
but still roughly matches ORT INT8 at 1 thread, proving that the layout/kernel strategy
alone closes most of the gap.

### Panel-im2col GEMM for dominant conv shapes

Panel-im2col GEMM was the key AVX2 block32 breakthrough. Instead of revisiting the same
spatial patch once per output-channel tile, the kernel gathers an output-pixel panel once
and reuses it across OC tiles. This moved block32 from the opt3/opt4 range into opt5,
where it roughly matches ORT INT8 without VNNI.

### Vectorized epilogues and fast q8 requantization

Replacing scalar/libm rounding and scalar per-channel epilogues with inline q8 rounding
and 4-wide OC epilogues removed avoidable per-output overhead across every dense q8
kernel. This helped both the intermediate block32 kernels and the final panel-GEMM path.

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

### Dense GEMM-style M_TILE=2 ternary kernel inside `basic_engine`

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

**Conclusion**: Dense ternary GEMM with per-k weight branching does not work inside the
`sparse-direct basic_engine` strategy. This negative result does **not** contradict the
later Q2Q/block32 panel-GEMM wins, because those engines first re-encode weights as dense
INT8 panels and avoid per-k ternary sign branches entirely. Not integrated. Files retained
in `benchmark_gemm3x3.cpp` for reference.

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

