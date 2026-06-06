from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


class RandomCalibrationDataReader(CalibrationDataReader):
    def __init__(self, num_samples: int = 32, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._data: list[dict[str, np.ndarray]] = [
            {"input": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}
            for _ in range(num_samples)
        ]
        self._iterator = iter(self._data)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iterator, None)


def compute_stats(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "min_ms": float(arr.min()),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
    }


def configure_thread_env(threads: int) -> None:
    # Keep non-ORT threadpools from oversubscribing.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["OMP_NESTED"] = "FALSE"

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def benchmark_ort(
    model_path: Path,
    input_array: np.ndarray,
    runs: int,
    warmup: int,
    threads: int,
) -> dict[str, float]:
    configure_thread_env(threads)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    for _ in range(warmup):
        session.run(None, {input_name: input_array})

    times_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_array})
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    return compute_stats(times_ms)


def parse_cpp_engine_stats(output: str) -> dict[str, float]:
    # Parse only the top latency block, not per-layer profile entries.
    block_match = re.search(
        r"C\+\+ engine latency\s+"
        r"runs:\s+\d+\s+"
        r"min_ms:\s+([0-9.]+)\s+"
        r"median_ms:\s+([0-9.]+)\s+"
        r"mean_ms:\s+([0-9.]+)\s+"
        r"p90_ms:\s+([0-9.]+)",
        output,
        flags=re.MULTILINE,
    )

    if block_match is None:
        raise RuntimeError("Could not parse C++ engine latency block")

    return {
        "min_ms": float(block_match.group(1)),
        "median_ms": float(block_match.group(2)),
        "mean_ms": float(block_match.group(3)),
        "p90_ms": float(block_match.group(4)),
    }


def benchmark_cpp_engine(
    engine_path: Path,
    model_bin: Path,
    input_bin: Path,
    golden_logits_bin: Path,
    threads: int,
) -> dict[str, float]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"
    env["OMP_NESTED"] = "FALSE"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        str(engine_path),
        str(model_bin),
        str(input_bin),
        str(golden_logits_bin),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    return parse_cpp_engine_stats(result.stdout)


def quantize_onnx_if_needed(
    fp32_onnx: Path,
    int8_onnx: Path,
    calib_samples: int,
    force: bool,
) -> None:
    if int8_onnx.exists() and not force:
        return

    print("Quantizing ONNX model to INT8...")

    calibration_reader = RandomCalibrationDataReader(
        num_samples=calib_samples,
        seed=0,
    )

    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def print_stats(label: str, stats: dict[str, float]) -> None:
    print(label)
    print(f"  min_ms:    {stats['min_ms']:.4f}")
    print(f"  median_ms: {stats['median_ms']:.4f}")
    print(f"  mean_ms:   {stats['mean_ms']:.4f}")
    print(f"  p90_ms:    {stats['p90_ms']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark C++ ternary engine vs ONNX Runtime FP32/INT8."
    )

    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--engine", type=Path, default=Path("build/basic_engine"))
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs"))
    parser.add_argument("--force-quantize", action="store_true")
    parser.add_argument(
        "--skip-ort-int8",
        action="store_true",
        help="Only benchmark C++ and ORT FP32.",
    )
    parser.add_argument(
        "--skip-ort-fp32",
        action="store_true",
        help="Only benchmark C++ and ORT INT8.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run cmake --build before benchmarking.",
    )
    parser.add_argument(
        "--ort-profile",
        action="store_true",
        help="Run one ORT INT8 profiling pass after benchmarks and save Chrome trace JSON.",
    )

    args = parser.parse_args()

    if args.build:
        import multiprocessing
        result = subprocess.run(
            ["cmake", "--build", "build", f"-j{multiprocessing.cpu_count()}"],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit("cmake build failed")

    artifact_dir: Path = args.artifact_dir

    fp32_onnx = artifact_dir / "resnet18_ternary_fp32.onnx"
    int8_onnx = artifact_dir / "resnet18_ternary_int8.onnx"

    model_bin = artifact_dir / "model_ternary_i8_weights_v1.bin"
    input_bin = artifact_dir / "golden_input_fp32.bin"
    golden_logits_bin = artifact_dir / "golden_logits_fp32.bin"
    input_npy = artifact_dir / "golden_input.npy"

    required_paths = [
        args.engine,
        fp32_onnx,
        model_bin,
        input_bin,
        golden_logits_bin,
        input_npy,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    if not args.skip_ort_int8:
        quantize_onnx_if_needed(
            fp32_onnx=fp32_onnx,
            int8_onnx=int8_onnx,
            calib_samples=args.calib_samples,
            force=args.force_quantize,
        )

    x = np.load(input_npy).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = args.out_dir / f"cpp_vs_ort_{timestamp}.json"
    csv_path = args.out_dir / f"cpp_vs_ort_{timestamp}.csv"

    rows: list[dict[str, Any]] = []

    print("Benchmark config")
    print(f"  runs:      {args.runs}")
    print(f"  warmup:    {args.warmup}")
    print(f"  threads:   {args.threads}")
    print()

    for threads in args.threads:
        print("=" * 48)
        print(f"threads={threads}")
        print("=" * 48)

        cpp_stats = benchmark_cpp_engine(
            engine_path=args.engine,
            model_bin=model_bin,
            input_bin=input_bin,
            golden_logits_bin=golden_logits_bin,
            threads=threads,
        )

        print_stats("C++ engine", cpp_stats)
        print()

        rows.append({
            "threads": threads,
            "runtime": "cpp_engine",
            **cpp_stats,
        })

        if not args.skip_ort_fp32:
            fp32_stats = benchmark_ort(
                model_path=fp32_onnx,
                input_array=x,
                runs=args.runs,
                warmup=args.warmup,
                threads=threads,
            )

            print_stats("ORT FP32", fp32_stats)
            print()

            rows.append({
                "threads": threads,
                "runtime": "ort_fp32",
                **fp32_stats,
            })

        if not args.skip_ort_int8:
            int8_stats = benchmark_ort(
                model_path=int8_onnx,
                input_array=x,
                runs=args.runs,
                warmup=args.warmup,
                threads=threads,
            )

            print_stats("ORT INT8", int8_stats)
            print()

            rows.append({
                "threads": threads,
                "runtime": "ort_int8",
                **int8_stats,
            })

    if args.ort_profile and not args.skip_ort_int8:
        profile_threads = args.threads[0]
        print(f"Running ORT INT8 profiling pass (threads={profile_threads})...")
        prof_opts = ort.SessionOptions()
        prof_opts.enable_profiling = True
        prof_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prof_opts.intra_op_num_threads = profile_threads
        prof_opts.inter_op_num_threads = 1
        prof_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        prof_session = ort.InferenceSession(
            str(int8_onnx),
            sess_options=prof_opts,
            providers=["CPUExecutionProvider"],
        )
        input_name = prof_session.get_inputs()[0].name
        prof_session.run(None, {input_name: x})  # warmup
        prof_session.run(None, {input_name: x})  # profiled pass
        trace_path = prof_session.end_profiling()
        print(f"ORT profile trace: {trace_path}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threads",
                "runtime",
                "min_ms",
                "median_ms",
                "mean_ms",
                "p90_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()