cmake_minimum_required(VERSION 3.20)

project(basic_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
)

target_include_directories(basic_engine PRIVATE
    src/basic_engine
)

target_compile_definitions(basic_engine PRIVATE NDEBUG)

target_compile_options(basic_engine PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
    -mavx2
    -mfma
)