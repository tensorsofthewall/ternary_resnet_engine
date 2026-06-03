from __future__ import annotations

import argparse
import time
from pathlib import Path

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
    

def benchmark_ort(
    model_path: Path,
    input_array: np.ndarray,
    runs: int,
    warmup: int,
    threads: int,
) -> dict[str, float]:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    for _ in range(warmup):
        session.run(None, {"input": input_array})

    times_ms: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"input": input_array})
        end = time.perf_counter()

        times_ms.append((end - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)

    return {
        "min_ms": float(arr.min()),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p90_ms": float(np.percentile(arr, 90)),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime FP32 and INT8.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--calib-samples", type=int, default=32)

    args = parser.parse_args()

    artifact_dir: Path = args.artifact_dir

    fp32_onnx = artifact_dir / "resnet18_ternary_fp32.onnx"
    int8_onnx = artifact_dir / "resnet18_ternary_int8.onnx"
    input_path = artifact_dir / "golden_input.npy"

    x = np.load(input_path).astype(np.float32)

    if not int8_onnx.exists():
        print("Quantizing ONNX model to INT8...")

        calibration_reader = RandomCalibrationDataReader(
            num_samples=args.calib_samples,
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

    print(f"Threads: {args.threads}")
    print()

    print("Benchmarking ORT FP32...")
    fp32_stats = benchmark_ort(
        model_path=fp32_onnx,
        input_array=x,
        runs=args.runs,
        warmup=args.warmup,
        threads=args.threads,
    )

    print("Benchmarking ORT INT8...")
    int8_stats = benchmark_ort(
        model_path=int8_onnx,
        input_array=x,
        runs=args.runs,
        warmup=args.warmup,
        threads=args.threads,
    )

    print()
    print("ORT FP32")
    for k, v in fp32_stats.items():
        print(f"  {k}: {v:.4f}")

    print()
    print("ORT INT8")
    for k, v in int8_stats.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()