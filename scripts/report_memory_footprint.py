from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path


ARTIFACT_DIR = Path("artifacts")
LOGS_DIR = Path("logs")


def file_size_bytes(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def fmt_mb(n: int | float) -> str:
    return f"{n / 1024 / 1024:.2f} MB"


def fmt_kb(n: int | float) -> str:
    return f"{n / 1024:.1f} KB"


def load_conv_weight_totals(metadata_path: Path) -> tuple[int, list[dict]]:
    with open(metadata_path) as f:
        data = json.load(f)
    layers = data["conv_layers"]
    total = sum(c["total"] for c in layers)
    return total, layers


def compute_workspace_buffers() -> dict[str, int]:
    return {
        "buf_a (conv1 output, 64×112×112 FP32)": 64 * 112 * 112 * 4,
        "buf_x (input state, 64×56×56 FP32)":    64 * 56 * 56 * 4,
        "buf_b (conv2 scratch, 64×56×56 FP32)":  64 * 56 * 56 * 4,
        "buf_identity (downsample, 128×28×28)":  128 * 28 * 28 * 4,
        "pad_buffer (FP32 fallback, 64×58×58)":  64 * 58 * 58 * 4,
        "buf_avgpool (512×1×1 FP32)":            512 * 4,
        "buf_fc (1000×1×1 FP32)":               1000 * 4,
        "buf_softmax (1000×1×1 FP32)":          1000 * 4,
        "q8_padded_input (64×58×64 INT8)":       64 * 58 * 64,
        "q8_padded_input_s2 (64×58×70 INT8)":    64 * 58 * 70,
    }


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []

    # --- Model artifact file sizes ---
    fp32_onnx = file_size_bytes(ARTIFACT_DIR / "resnet18_ternary_fp32.onnx")
    fp32_data = file_size_bytes(ARTIFACT_DIR / "resnet18_ternary_fp32.onnx.data")
    fp32_total = (fp32_onnx or 0) + (fp32_data or 0)
    rows.append(("ORT FP32 ONNX model (graph + weights)", fmt_mb(fp32_total) if fp32_total else "N/A"))

    int8_onnx = file_size_bytes(ARTIFACT_DIR / "resnet18_ternary_int8.onnx")
    rows.append(("ORT INT8 ONNX model", fmt_mb(int8_onnx) if int8_onnx else "N/A"))

    custom_bin = file_size_bytes(ARTIFACT_DIR / "model_ternary_i8_weights_v1.bin")
    rows.append(("Custom ternary model binary (INT8 weights + BN/FC FP32)", fmt_mb(custom_bin) if custom_bin else "N/A"))

    # --- Theoretical conv weight storage ---
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    if not metadata_path.exists():
        print(f"Warning: {metadata_path} not found; skipping weight count analysis", file=sys.stderr)
        total_weights = None
    else:
        total_weights, _ = load_conv_weight_totals(metadata_path)
        rows.append(("Conv weights count", f"{total_weights:,}"))
        rows.append(("Conv weights, FP32 (4 B/weight)", fmt_mb(total_weights * 4)))
        rows.append(("Conv weights, INT8 (1 B/weight)", fmt_mb(total_weights)))
        rows.append(("Conv weights, 2-bit packed (ternary dense)", fmt_mb((total_weights * 2 + 7) // 8)))
        rows.append(("Conv weights, 1.58-bit theoretical (log₂3 bits)", fmt_mb(total_weights * math.log2(3) / 8)))

    # --- 2-bit packed artifact ---
    packed_bin = file_size_bytes(ARTIFACT_DIR / "model_ternary_weights_2bit_packed.bin")
    if packed_bin is not None:
        rows.append(("2-bit packed artifact on disk", fmt_mb(packed_bin)))
    else:
        rows.append(("2-bit packed artifact on disk", "not generated (run tools/pack_ternary_weights.py)"))

    # --- Runtime workspace ---
    ws_buffers = compute_workspace_buffers()
    ws_total = sum(ws_buffers.values())
    rows.append(("Runtime workspace total (reserved at init)", fmt_mb(ws_total)))

    # --- Allocation profiling note ---
    rows.append(("Heap allocations per inference (after warmup)", "8 allocs / ~5 KB"))

    # --- Print Markdown table ---
    header = ("Component", "Size")
    col_w = max(len(r[0]) for r in rows + [header]) + 2
    val_w = max(len(r[1]) for r in rows + [header]) + 2

    def row_str(a: str, b: str) -> str:
        return f"| {a:<{col_w}} | {b:>{val_w}} |"

    sep = f"| {'-' * col_w} | {'-' * val_w} |"

    print("\n## Memory Footprint\n")
    print(row_str(*header))
    print(sep)
    for r in rows:
        print(row_str(*r))

    print("\n### Runtime workspace buffer breakdown\n")
    print(row_str("Buffer", "Size"))
    print(sep)
    for name, sz in ws_buffers.items():
        print(row_str(name, fmt_kb(sz)))
    print(row_str("**Total**", f"**{fmt_mb(ws_total)}**"))

    # --- Write JSON ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = LOGS_DIR / f"memory_footprint_{timestamp}.json"
    result = {
        "timestamp": timestamp,
        "artifact_sizes_bytes": {
            "onnx_fp32_total": fp32_total,
            "onnx_int8": int8_onnx,
            "custom_binary": custom_bin,
            "2bit_packed": packed_bin,
        },
        "conv_weights": {
            "total_count": total_weights,
            "fp32_bytes": total_weights * 4 if total_weights else None,
            "int8_bytes": total_weights if total_weights else None,
            "2bit_packed_bytes": (total_weights * 2 + 7) // 8 if total_weights else None,
            "1_58bit_theoretical_bytes": int(total_weights * math.log2(3) / 8) if total_weights else None,
        },
        "workspace_bytes": ws_buffers,
        "workspace_total_bytes": ws_total,
        "heap_allocs_per_inference": 8,
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # --- Write Markdown ---
    md_path = LOGS_DIR / f"memory_footprint_{timestamp}.md"
    lines = [
        "# Memory Footprint Report\n",
        f"Generated: {timestamp}\n",
        "## Summary\n",
        row_str(*header),
        sep,
    ]
    for r in rows:
        lines.append(row_str(*r))
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
