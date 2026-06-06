#!/usr/bin/env bash
# Autotune existing dispatch flags — no new kernels.
# Toggles q8 OCBlock2 layer1/2/3 flags one at a time and in combination.
# Reports 1T and 8T min_ms for each configuration.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/src/basic_engine/resnet18_engine.cpp"
MODEL="$REPO/artifacts/model_ternary_i8_weights_v1.bin"
INPUT="$REPO/artifacts/golden_input_fp32.bin"
LOGITS="$REPO/artifacts/golden_logits_fp32.bin"
ENGINE="$REPO/build/basic_engine"
parse_min() {
    grep "min_ms:" | awk '{print $2}' | head -1
}

benchmark_threads() {
    local threads=$1
    OMP_NUM_THREADS=$threads OMP_PROC_BIND=close OMP_PLACES=cores \
        "$ENGINE" "$MODEL" "$INPUT" "$LOGITS" 2>/dev/null | parse_min
}

set_flag() {
    local flag=$1 val=$2
    # Replace in both conv_bn and conv_bn_relu (two occurrences)
    sed -i "s/constexpr bool ${flag} = \(true\|false\);/constexpr bool ${flag} = ${val};/g" "$SRC"
}

rebuild() {
    cmake --build "$REPO/build" -j"$(nproc)" --target basic_engine > /dev/null 2>&1
}

run_config() {
    local label=$1
    rebuild
    local min1t min8t
    min1t=$(benchmark_threads 1)
    min8t=$(benchmark_threads 8)
    printf "%-55s  1T: %6s ms   8T: %6s ms\n" "$label" "$min1t" "$min8t"
}

echo "=== Dispatch Flag Autotune ==="
echo "Source: $SRC"
echo "Runs: 100  Warmup: 20 (hardcoded in main.cpp)"
echo

# ── Baseline ────────────────────────────────────────────────────────────────
run_config "baseline (all current flags)"

# ── Layer1 OCBlock2 OFF ──────────────────────────────────────────────────────
set_flag enable_q8_ocblock2_layer1_s1_kernel false
run_config "layer1_ocblock2=OFF"
set_flag enable_q8_ocblock2_layer1_s1_kernel true

# ── Layer2 OCBlock2 OFF ──────────────────────────────────────────────────────
set_flag enable_q8_ocblock2_layer2_s1_kernel false
run_config "layer2_ocblock2=OFF"
set_flag enable_q8_ocblock2_layer2_s1_kernel true

# ── Layer3 OCBlock2 OFF (falls back to OCBlock4 q8) ─────────────────────────
set_flag enable_q8_ocblock2_layer3_s1_kernel false
run_config "layer3_ocblock2=OFF (OCBlock4 q8 fallback)"
set_flag enable_q8_ocblock2_layer3_s1_kernel true

# ── Layer1+2 OCBlock2 OFF ────────────────────────────────────────────────────
set_flag enable_q8_ocblock2_layer1_s1_kernel false
set_flag enable_q8_ocblock2_layer2_s1_kernel false
run_config "layer1+2_ocblock2=OFF"
set_flag enable_q8_ocblock2_layer1_s1_kernel true
set_flag enable_q8_ocblock2_layer2_s1_kernel true

# ── Layer1+2 OFF, Layer3 OCBlock4 ───────────────────────────────────────────
set_flag enable_q8_ocblock2_layer1_s1_kernel false
set_flag enable_q8_ocblock2_layer2_s1_kernel false
set_flag enable_q8_ocblock2_layer3_s1_kernel false
run_config "layer1+2+3_ocblock2=OFF (layer3 OCBlock4 q8)"
set_flag enable_q8_ocblock2_layer1_s1_kernel true
set_flag enable_q8_ocblock2_layer2_s1_kernel true
set_flag enable_q8_ocblock2_layer3_s1_kernel true

# ── Restore and confirm baseline ─────────────────────────────────────────────
run_config "baseline restored"
