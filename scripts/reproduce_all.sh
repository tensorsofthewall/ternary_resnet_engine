#!/usr/bin/env bash
# Reproduces all benchmark results from a clean checkout.
# Requires: cmake, gcc, python 3.11+ with torch/onnxruntime/numpy installed.
# Usage: bash scripts/reproduce_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
RUNS=100
WARMUP=20
THREADS="1 2 4 8 16 24"

# Prefer .venv if present
if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
fi

echo "==> Python: $($PYTHON --version)"
echo "==> Repo: $REPO_ROOT"
echo

# ── Step 1: Generate model artifacts ────────────────────────────────────────
echo "==> [1/6] Generating model artifacts..."
$PYTHON tools/export_model.py
echo

# ── Step 2: Generate 2-bit packed weights ───────────────────────────────────
echo "==> [2/6] Packing ternary weights (2-bit)..."
$PYTHON tools/pack_ternary_weights.py
echo

# ── Step 3: Build C++ engine ─────────────────────────────────────────────────
echo "==> [3/6] Building C++ engine..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j"$(nproc)"
echo

# ── Step 4: Single-thread smoke test ─────────────────────────────────────────
echo "==> [4/6] Single-thread smoke test..."
OMP_NUM_THREADS=1 ./build/basic_engine \
    artifacts/model_ternary_i8_weights_v1.bin \
    artifacts/golden_input_fp32.bin \
    artifacts/golden_logits_fp32.bin
echo

# ── Step 5: C++ vs ORT benchmark ─────────────────────────────────────────────
echo "==> [5/6] C++ vs ORT benchmark (threads: $THREADS)..."
$PYTHON scripts/benchmark_cpp_vs_ort.py \
    --threads $THREADS \
    --runs "$RUNS" \
    --warmup "$WARMUP"
echo

# ── Step 6: PyTorch FP32 benchmark ───────────────────────────────────────────
echo "==> [6/6] PyTorch FP32 benchmark (threads: $THREADS)..."
$PYTHON scripts/benchmark_pytorch_fp32.py \
    --threads $THREADS \
    --runs "$RUNS" \
    --warmup "$WARMUP"
echo

# ── Memory footprint report ───────────────────────────────────────────────────
echo "==> Memory footprint report..."
$PYTHON scripts/report_memory_footprint.py
echo

echo "==> Done. Results written to logs/."
echo
echo "Optional: standalone packed-panel benchmark (experimental, not wired into engine):"
echo "  ./build/benchmark_packed_3x3 artifacts/model_ternary_i8_weights_v1.bin"
