from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def compute_stats(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "min_ms": float(arr.min()),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def load_model(artifact_dir: Path) -> nn.Module:
    # Add repo root to path so models/ is importable
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.resnet18 import build_ternary_resnet18

    state_path = artifact_dir / "resnet18_ternary_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"State dict not found: {state_path}\n"
            "Run: uv run python tools/export_model.py"
        )

    # Build model with same defaults used by export_model.py
    model = build_ternary_resnet18(num_classes=1000, seed=42, threshold_factor=0.7)
    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def benchmark_pytorch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    runs: int,
    warmup: int,
    threads: int,
) -> dict[str, float]:
    torch.set_num_threads(threads)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

        times_ms: list[float] = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    return compute_stats(times_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch FP32 ternary ResNet-18 inference on CPU"
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 24],
        metavar="T",
        help="Thread counts to benchmark (default: 1 2 4 8 16 24)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of timed runs per thread count (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup runs (default: 20)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing golden_input.npy and model state dict (default: artifacts)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact_dir: Path = args.artifact_dir
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Set interop threads once before any parallel work starts
    torch.set_num_interop_threads(1)

    print("Loading model...", flush=True)
    model = load_model(artifact_dir)

    input_path = artifact_dir / "golden_input.npy"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}\n"
            "Run: uv run python tools/export_model.py"
        )
    x_np = np.load(input_path).astype(np.float32)
    x = torch.from_numpy(x_np)
    print(f"Input shape: {tuple(x.shape)}", flush=True)

    results: list[dict] = []

    for threads in args.threads:
        print(f"  threads={threads:2d}  ", end="", flush=True)
        stats = benchmark_pytorch(model, x, runs=args.runs, warmup=args.warmup, threads=threads)
        row = {"threads": threads, "runtime": "pytorch_fp32", **stats}
        results.append(row)
        print(
            f"min={stats['min_ms']:.3f} ms  "
            f"median={stats['median_ms']:.3f} ms  "
            f"mean={stats['mean_ms']:.3f} ms  "
            f"p90={stats['p90_ms']:.3f} ms"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"pytorch_fp32_{timestamp}.json"
    csv_path = logs_dir / f"pytorch_fp32_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    fieldnames = ["threads", "runtime", "min_ms", "median_ms", "mean_ms", "p90_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
