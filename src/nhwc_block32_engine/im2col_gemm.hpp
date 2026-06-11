#pragma once

// im2col_gemm.hpp -- experimental im2col/GEMM-style AVX2 block32 kernels.
//
// Tests whether materializing a dense im2col activation matrix A[N][K_pad]
// (uint8) and a flat, 32-byte-aligned, oc_tile-major weight panel
// B[oc_tile][k_block][oc_lane][32] (int8 ternary), then running a plain
// GEMM-style OC4 microkernel over them, can beat the specialized
// conv3x3s1p1_*_oc4_n2_avx2 kernels for the repeated 3x3/stride1/pad1 shapes.
//
// AVX2-only (no AVX-VNNI), q8 NHWC activations, int8 ternary {-1,0,+1}
// weights, exact agreement with conv_block32_scalar required. Benchmark-only
// for now -- not wired into any production forward pass.

#include "block32_layer234_opt.hpp"  // Conv3x3S1P1PanelPlan
#include "block32_plan.hpp"
#include "model.hpp"
#include "q2q_tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

namespace nhwc_block32 {

// Move-only, 32-byte-aligned heap buffer of `count` T's (uninitialized).
template <typename T>
class AlignedBuffer32 {
public:
    AlignedBuffer32() = default;
    explicit AlignedBuffer32(size_t count)
        : ptr_(count != 0
                   ? static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t(32)))
                   : nullptr),
          size_(count) {}
    ~AlignedBuffer32() { reset(); }

    AlignedBuffer32(const AlignedBuffer32&) = delete;
    AlignedBuffer32& operator=(const AlignedBuffer32&) = delete;

    AlignedBuffer32(AlignedBuffer32&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    AlignedBuffer32& operator=(AlignedBuffer32&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }

private:
    void reset() {
        if (ptr_ != nullptr) {
            ::operator delete(ptr_, std::align_val_t(32));
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// Flat, oc_tile-major, 32-byte-aligned ternary weight panel for an OC4/K32
// GEMM: logical layout B[oc_tile][k_block][oc_lane][32], oc_tile in
// [0, out_channels/4), k_block in [0, K_pad/32), oc_lane in [0,4). K_pad is
// always a multiple of 32 (K_pad = 9*ic_blocks*32 for a 3x3/s1/p1 conv).
struct PackedBPanel {
    int out_channels = 0;
    int K_pad = 0;
    AlignedBuffer32<int8_t> data;

    // Pointer to the 32 ternary weights for (oc_tile, k_block, oc_lane).
    const int8_t* block(int oc_tile, int k_block, int lane) const {
        const int k_blocks = K_pad / 32;
        const size_t idx = ((static_cast<size_t>(oc_tile) * static_cast<size_t>(k_blocks) + static_cast<size_t>(k_block)) * 4
                             + static_cast<size_t>(lane)) * 32;
        return data.data() + idx;
    }
};

// Builds a PackedBPanel from an existing Conv3x3S1P1PanelPlan (same
// (ky,kx,ic_block) k_block ordering as build_conv3x3s1p1_panel_plan), copied
// into flat, contiguous, 32-byte-aligned storage.
PackedBPanel build_packed_b_panel(const Conv3x3S1P1PanelPlan& plan);

// Materialized im2col activation matrix: A[N][K_pad], uint8, row-major,
// 32-byte-aligned base (so every row, being a multiple of 32 bytes, is also
// 32-byte aligned).
struct Im2colMatrix {
    int N = 0;
    int K_pad = 0;
    AlignedBuffer32<uint8_t> data;

    const uint8_t* row(int n) const { return data.data() + static_cast<size_t>(n) * static_cast<size_t>(K_pad); }
};

// Builds the materialized im2col matrix for a 3x3/stride1/pad1 block32 conv
// from a q8 NHWC input tensor (in.border zero-halo provides the padding
// positions, exactly as conv3x3s1p1_oc4_avx2_impl relies on). K_pad =
// 9*ic_blocks*32; k_block index = ky*3*ic_blocks + kx*ic_blocks + ic_block,
// matching build_conv3x3s1p1_panel_plan's ordering. N = OH*OW.
Im2colMatrix build_im2col_3x3s1p1(const q2q::QTensorNHWC& in, int ic_blocks, int OH, int OW);

// GEMM-style OC4 microkernels: C[N][out_channels] (q8 NHWC, written into
// `out`) = requantize(A[N][K_pad] . B[oc_tile][:][lane][32] + bias), with
// vector accumulators kept across all K_pad/32 blocks and hsum'd once at the
// end. N_TILE adjacent output pixels (row-major n = oh*OW+ow) are processed
// per inner iteration; A.N must be a multiple of N_TILE.
void gemm_oc4_n1_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void gemm_oc4_n2_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void gemm_oc4_n4_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void gemm_oc4_n8_avx2(
    const PackedBPanel& B, const Im2colMatrix& A, const BNFusedParams& bn,
    float in_scale, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Step 6 (panel im2col): builds a small A_panel[N_TILE][K_pad] for each tile
// of N_TILE adjacent output pixels directly from the q8 NHWC input (no full
// im2col materialization), then runs the OC4 GEMM microkernel against that
// panel for every oc_tile. Handles OW not a multiple of N_TILE via an N_TILE=1
// tail for the remaining columns of each row.
void panel_im2col_gemm_oc4_n1_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void panel_im2col_gemm_oc4_n2_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void panel_im2col_gemm_oc4_n4_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void panel_im2col_gemm_oc4_n8_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

// Phase 16 Step 3: like panel_im2col_gemm_oc4_n*_avx2 but for 3x3/stride2/pad1
// transition convs (layer2.0.conv1, layer3.0.conv1, layer4.0.conv1). B must
// come from build_conv3x3_panel_plan_strided. Handles OW not a multiple of
// N_TILE via an N_TILE=1 tail.
void panel_im2col_gemm_oc4_n2_stride2_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW);
void panel_im2col_gemm_oc4_n4_stride2_avx2(
    const PackedBPanel& B, const q2q::QTensorNHWC& in, int ic_blocks,
    const BNFusedParams& bn, bool relu, q2q::QTensorNHWC& out, int OH, int OW);

} // namespace nhwc_block32
