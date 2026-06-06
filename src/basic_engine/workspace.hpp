#pragma once

#include "tensor.hpp"
#include "quantized/quant_tensor.hpp"

struct EngineWorkspace {
    Tensor pad_buffer;

    QuantTensorI8 q8_padded_input;
    QuantTensorI8 q8_padded_input_s2;

    // Pre-allocated activation buffers for forward_resnet18_scalar.
    // Reused across inference calls via Tensor::reuse() — no malloc/zero-init after first run.
    Tensor buf_x;         // current input "x" in the forward pass
    Tensor buf_a;         // conv1 output within a block (largest: 64x112x112)
    Tensor buf_b;         // conv2 output / new x candidate
    Tensor buf_identity;  // identity branch for downsampling blocks

    Tensor buf_avgpool;   // adaptive_avgpool_1x1 output (512x1x1)
    Tensor buf_fc;        // linear (fc) output (1000x1x1)
    Tensor buf_softmax;   // softmax output (1000x1x1)

    EngineWorkspace() { reserve_max(); }

    void reserve_max() {
        // buf_a holds conv1 output: 64x112x112 = 802816 elements (largest activation)
        buf_a.reserve(64, 112, 112);
        // buf_x holds post-maxpool state: 64x56x56 = 200704 elements (largest)
        buf_x.reserve(64, 56, 56);
        // buf_b is swapped with buf_x; same maximum needed
        buf_b.reserve(64, 56, 56);
        // buf_identity holds downsample output: 128x28x28 = 100352 elements (layer2.0 is largest)
        buf_identity.reserve(128, 28, 28);
        // pad_buffer: precautionary for any FP32 fallback kernels
        pad_buffer.reserve(64, 58, 58);

        // q8_padded_input: largest is layer2.0 1x1-s2 downsample (64x56x64 = 229376 bytes)
        q8_padded_input.data.reserve(
            static_cast<size_t>(64) * static_cast<size_t>(58) * static_cast<size_t>(64)
        );
        // q8_padded_input_s2: largest is layer2.0 stride-2 conv (64x58x66 = 245376 bytes)
        q8_padded_input_s2.data.reserve(
            static_cast<size_t>(64) * static_cast<size_t>(58) * static_cast<size_t>(70)
        );

        buf_avgpool.reserve(512, 1, 1);
        buf_fc.reserve(1000, 1, 1);
        buf_softmax.reserve(1000, 1, 1);
    }
};