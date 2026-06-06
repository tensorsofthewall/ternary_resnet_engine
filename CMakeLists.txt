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