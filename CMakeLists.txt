cmake_minimum_required(VERSION 3.20)

project(basic_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(basic_engine
    src/basic_engine/main.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/ops.cpp
    src/basic_engine/resnet18_engine.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_avx2.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/conv3x3_sparse_ocblock_avx2.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock2_avx2.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock4_avx2.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_avx2.cpp
    src/basic_engine/quantized/quant_utils.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock4_avx2.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock2_avx2.cpp
    src/basic_engine/quantized/conv1x1_q8_s2_avx2.cpp
)

target_include_directories(basic_engine PRIVATE
    src/basic_engine
)

target_compile_definitions(basic_engine PRIVATE NDEBUG)

option(ENABLE_ALLOC_PROFILE "Track heap allocations per forward pass (basic_engine only)" OFF)
if(ENABLE_ALLOC_PROFILE)
    target_compile_definitions(basic_engine PRIVATE ENABLE_ALLOC_PROFILE)
endif()

target_compile_options(basic_engine PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)


add_executable(benchmark_layer4_q8
    src/basic_engine/benchmark_layer4_q8.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/ops.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_avx2.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/conv3x3_sparse_ocblock_avx2.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock2_avx2.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock4_avx2.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_avx2.cpp
    src/basic_engine/quantized/quant_utils.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock4_avx2.cpp
)

target_include_directories(benchmark_layer4_q8 PRIVATE
    src/basic_engine
)

target_compile_definitions(benchmark_layer4_q8 PRIVATE NDEBUG)

target_compile_options(benchmark_layer4_q8 PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)

add_executable(benchmark_packed_3x3
    src/basic_engine/benchmark_packed_3x3.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/ops.cpp
    src/basic_engine/resnet18_engine.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_avx2.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/conv3x3_sparse_ocblock_avx2.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock2_avx2.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock4_avx2.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_avx2.cpp
    src/basic_engine/quantized/quant_utils.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock4_avx2.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock2_avx2.cpp
    src/basic_engine/quantized/conv1x1_q8_s2_avx2.cpp
    src/basic_engine/quantized/packed_q8_3x3_kernel.cpp
)

target_include_directories(benchmark_packed_3x3 PRIVATE
    src/basic_engine
)

target_compile_definitions(benchmark_packed_3x3 PRIVATE NDEBUG)

target_compile_options(benchmark_packed_3x3 PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)

add_executable(benchmark_gemm3x3
    src/basic_engine/benchmark_gemm3x3.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/ops.cpp
    src/basic_engine/resnet18_engine.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_avx2.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/conv3x3_sparse_ocblock_avx2.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock2_avx2.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock4_avx2.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_avx2.cpp
    src/basic_engine/quantized/quant_utils.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock4_avx2.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock2_avx2.cpp
    src/basic_engine/quantized/conv1x1_q8_s2_avx2.cpp
    src/basic_engine/quantized/packed_q8_3x3_kernel.cpp
)

target_include_directories(benchmark_gemm3x3 PRIVATE src/basic_engine)
target_compile_definitions(benchmark_gemm3x3 PRIVATE NDEBUG)
target_compile_options(benchmark_gemm3x3 PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)

find_package(OpenMP REQUIRED)

add_executable(benchmark_nchwc
    src/basic_engine/benchmark_nchwc.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/ops.cpp
    src/basic_engine/resnet18_engine.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_avx2.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/conv3x3_sparse_ocblock_avx2.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock2_avx2.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv3x3_pattern_ocblock4_avx2.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_avx2.cpp
    src/basic_engine/quantized/quant_utils.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock4_avx2.cpp
    src/basic_engine/quantized/conv3x3_q8_pattern_ocblock2_avx2.cpp
    src/basic_engine/quantized/conv1x1_q8_s2_avx2.cpp
    src/basic_engine/quantized/packed_q8_3x3_kernel.cpp
)

target_include_directories(benchmark_nchwc PRIVATE src/basic_engine)
target_compile_definitions(benchmark_nchwc PRIVATE NDEBUG)
target_compile_options(benchmark_nchwc PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)

target_link_libraries(basic_engine PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_layer4_q8 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_packed_3x3 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_gemm3x3 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(benchmark_nchwc PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Blocked engine: NCHWc16 activations + mullo_epi32 ternary conv prototype.
# Separate from basic_engine — shares model loader and plan builders only.
# Phase 1: types + weight reorganization (compile test).
# ---------------------------------------------------------------------------
add_executable(blocked_engine
    src/blocked_engine/blocked_engine_main.cpp
    src/blocked_engine/blocked_model.cpp
    src/blocked_engine/q8_runtime.cpp
    src/blocked_engine/blocked_conv3x3.cpp
    src/blocked_engine/blocked_ops.cpp
    src/blocked_engine/blocked_conv1.cpp
    src/blocked_engine/blocked_resnet18.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)

target_include_directories(blocked_engine PRIVATE
    src/blocked_engine
    src/basic_engine
)

target_compile_definitions(blocked_engine PRIVATE NDEBUG)

target_compile_options(blocked_engine PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)

target_link_libraries(blocked_engine PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# benchmark_cbhwb: CB_H_WB_CI_WI_EXT sparse OW8 kernel (runtime-only).
# Layer2.1.conv2 only. No correctness check.
# ---------------------------------------------------------------------------
add_executable(benchmark_cbhwb
    src/basic_engine/benchmark_cbhwb.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_cbhwb PRIVATE src/basic_engine)
target_compile_definitions(benchmark_cbhwb PRIVATE NDEBUG)
target_compile_options(benchmark_cbhwb PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma)
target_link_libraries(benchmark_cbhwb PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# benchmark_w16w32: wider-tile blocked sparse OC2 kernels (runtime-only).
# Variant A: W_BLOCK=16 OW_TILE=16. Variant B: W_BLOCK=32 OW_TILE=32.
# Layer2.1.conv2 only. No correctness check. Lives alongside blocked_engine.
# ---------------------------------------------------------------------------
add_executable(benchmark_w16w32
    src/blocked_engine/benchmark_w16w32.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_w16w32 PRIVATE src/basic_engine src/blocked_engine)
target_compile_definitions(benchmark_w16w32 PRIVATE NDEBUG)
target_compile_options(benchmark_w16w32 PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma)
target_link_libraries(benchmark_w16w32 PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q experimental engine: NHWC uint8 activations + dense ternary int8 weights
# + AVX-VNNI dpbusd_epi32. Separate from basic_engine/blocked_engine — shares
# only the model loader and BN fusion. Phase 1: standalone single-layer
# benchmark for layer2.1.conv2 (gate: total<=2.13ms continue, <=1.5ms strong).
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_vnni
    src/q2q_engine/benchmark_q2q_vnni.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_vnni PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_vnni PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_vnni PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_vnni PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q Phase 2: layer2.1 q8-to-q8 residual block (runtime-only, no correctness
# check). Chains the validated VNNI dense kernel through conv1 -> conv2 ->
# residual entirely in QTensorNHWC (uint8 NHWC) — no FP32 activation tensor
# crosses any op boundary. Gate: q2q_only must beat the measured basic_engine
# layer2.1 block (~5.42 ms, q8 OCBlock2 bucket, 1T) to continue.
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_block
    src/q2q_engine/benchmark_q2q_block.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_block PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_block PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_block PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_block PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q Phase 3: full layer2 stack (layer2.0 + layer2.1), q8-to-q8 end-to-end.
# layer2.0 adds conv1 stride-2, conv2 stride-1, a 1x1 stride-2 downsample
# projection, and its own q8 residual add+ReLU, chained straight into the
# validated layer2.1 block. One fused conv kernel (generalized for stride and
# pad) serves all five convs. Gate: q2q_only must beat the measured
# basic_engine layer2 stack (~10.28 ms, q8 OCBlock2 bucket, 1T) to proceed
# toward layer3 / full ResNet.
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_layer2_stack
    src/q2q_engine/benchmark_q2q_layer2_stack.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_layer2_stack PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_layer2_stack PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_layer2_stack PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_layer2_stack PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q Phase 4: full layer3 stack (layer3.0 + layer3.1), q8-to-q8 end-to-end.
# Structurally identical to the layer2 stack (conv1 s2, conv2 s1, 1x1 s2
# downsample, q8 residual add+ReLU) but with K doubled (128->256, 256->256) —
# tests whether VNNI's per-MAC throughput edge grows with K. Reuses the same
# fused conv kernel / vnni_oc4_dots / residual-add byte-for-byte; only model
# tensor names, channel counts, and spatial dims differ. Gate: q2q_only must
# beat the measured basic_engine layer3 stack (~12.50 ms, q8 OCBlock2 bucket,
# 1T) to proceed toward full ResNet.
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_layer3_stack
    src/q2q_engine/benchmark_q2q_layer3_stack.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_layer3_stack PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_layer3_stack PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_layer3_stack PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_layer3_stack PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q Phase 5: combined layer2+layer3 stack, q8-to-q8 end-to-end with NO FP32
# materialization at the layer2/layer3 boundary (layer2's q8 block output
# feeds directly into layer3.0). Tests whether each stack's individually
# validated speedup (1.48x / ~2.05x) survives chaining. Reuses every kernel
# byte-for-byte from the layer2/layer3 stack benchmarks — pure orchestration.
# Gate: layer23_q2q_only <= 15.0 ms continue (<= 13.5 ms strong) vs measured
# basic_engine combined cost (~22.78 ms, q8 OCBlock2 bucket, 1T).
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_layer23_stack
    src/q2q_engine/benchmark_q2q_layer23_stack.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_layer23_stack PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_layer23_stack PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_layer23_stack PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_layer23_stack PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q Phase 6: full layer4 stack (layer4.0 + layer4.1), q8-to-q8 end-to-end.
# Structurally identical to layer2/layer3 but at the network's largest K
# (256->512, 512->512) and smallest spatial extent (7x7 = 49 output pixels).
# Tests whether dense VNNI's per-MAC throughput edge (favored by large K)
# outweighs fixed per-pixel pack/loop overhead (amortized over far fewer
# pixels than layer2/layer3) -- empirical, not assumed. Reuses every kernel
# byte-for-byte from the layer2/layer3 stack benchmarks. Gate: q2q_only must
# beat the measured basic_engine layer4 stack (~9.59 ms, q8 OCBlock2 bucket,
# 1T).
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_layer4_stack
    src/q2q_engine/benchmark_q2q_layer4_stack.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_layer4_stack PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_layer4_stack PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_layer4_stack PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_layer4_stack PRIVATE OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# Q2Q Phase 7: combined layer2+3+4 stack, q8-to-q8 end-to-end with NO FP32
# materialization at ANY stack boundary -- each stack's q8 block output feeds
# the next stack's conv1/downsample directly. Verifies the validated per-stack
# speedups (1.48x / ~2.05x / ~2.1x) survive across the entire deep ResNet
# trunk. Reuses every kernel byte-for-byte from the per-stack benchmarks --
# pure orchestration. Gate: layer234_q2q_only must beat the measured
# basic_engine combined cost (~32.37 ms, q8 OCBlock2 bucket, 1T).
# ---------------------------------------------------------------------------
add_executable(benchmark_q2q_layer234_stack
    src/q2q_engine/benchmark_q2q_layer234_stack.cpp
    src/basic_engine/model_loader.cpp
    src/basic_engine/bn_fusion.cpp
    src/basic_engine/conv_plan.cpp
    src/basic_engine/sparse_ocblock_plan.cpp
    src/basic_engine/pattern_ocblock2_plan.cpp
    src/basic_engine/pattern_ocblock4_plan.cpp
    src/basic_engine/conv1_ocblock4_plan.cpp
)
target_include_directories(benchmark_q2q_layer234_stack PRIVATE src/basic_engine src/q2q_engine)
target_compile_definitions(benchmark_q2q_layer234_stack PRIVATE NDEBUG)
target_compile_options(benchmark_q2q_layer234_stack PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic -mavx2 -mfma -mavxvnni)
target_link_libraries(benchmark_q2q_layer234_stack PRIVATE OpenMP::OpenMP_CXX)