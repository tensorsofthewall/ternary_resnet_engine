#!/usr/bin/env bash
set -euo pipefail

MODEL="artifacts/model_ternary_i8_weights_v1.bin"
INPUT="artifacts/golden_input_fp32.bin"
GOLDEN="artifacts/golden_logits_fp32.bin"

THREAD_COUNTS=(1 2 4 8)

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/thread_scaling_${TIMESTAMP}.log"

{
    echo "Thread-scaling benchmark"
    echo "Timestamp: ${TIMESTAMP}"
    echo "Host: $(hostname)"
    echo "OMP_PROC_BIND=${OMP_PROC_BIND}"
    echo "OMP_PLACES=${OMP_PLACES}"
    echo "OMP_MAX_ACTIVE_LEVELS=${OMP_MAX_ACTIVE_LEVELS}"
    echo "OMP_NESTED=${OMP_NESTED}"
    echo

    echo "Building..."
    cmake --build build -j

    echo
    echo "Running thread-scaling benchmark..."
    echo

    for t in "${THREAD_COUNTS[@]}"; do
        echo "========================================"
        echo "threads=${t}"
        echo "========================================"

        OMP_NUM_THREADS="${t}" \
        ./build/basic_engine "$MODEL" "$INPUT" "$GOLDEN"

        echo
    done
} 2>&1 | tee "$LOG_FILE"

echo
echo "Saved log to: $LOG_FILE"