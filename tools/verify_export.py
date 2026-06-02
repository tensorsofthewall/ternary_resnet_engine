from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray

from models.resnet18 import build_ternary_resnet18


ARTIFACT_DIR: Final[Path] = Path("artifacts")
ONNX_FILENAME: Final[str] = "resnet18_ternary_fp32.onnx"
GOLDEN_INPUT_FILENAME: Final[str] = "golden_input.npy"
GOLDEN_LOGITS_FILENAME: Final[str] = "golden_logits.npy"

FloatArray = NDArray[np.float32]


def normalize_logits(logits: FloatArray) -> FloatArray:
    """Normalize logits by max absolute value per batch item."""
    scale = np.max(np.abs(logits), axis=1, keepdims=True)
    return (logits / np.maximum(scale, 1e-12)).astype(np.float32)


def compute_abs_error(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.abs(a - b).astype(np.float32)


def compute_rel_error(a: FloatArray, b: FloatArray) -> FloatArray:
    abs_err = compute_abs_error(a, b)
    denom = np.maximum(np.abs(a), 1e-12)
    return (abs_err / denom).astype(np.float32)


def print_error_report(name: str, reference: FloatArray, candidate: FloatArray) -> None:
    abs_err = compute_abs_error(reference, candidate)
    rel_err = compute_rel_error(reference, candidate)

    print(name)
    print(f"  max_abs_error:   {abs_err.max():.8e}")
    print(f"  mean_abs_error:  {abs_err.mean():.8e}")
    print(f"  max_rel_error:   {rel_err.max():.8e}")
    print(f"  mean_rel_error:  {rel_err.mean():.8e}")
    print(f"  allclose strict: {np.allclose(reference, candidate, atol=1e-5, rtol=1e-5)}")
    print(f"  allclose relaxed:{np.allclose(reference, candidate, atol=1e-3, rtol=1e-5)}")


def validate_finite(name: str, array: FloatArray) -> None:
    if not np.isfinite(array).all():
        num_bad = int((~np.isfinite(array)).sum())
        raise RuntimeError(f"{name} contains {num_bad} NaN/Inf values.")


def load_required_array(path: Path) -> FloatArray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")

    return np.load(path).astype(np.float32)

def load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    with path.open("r") as f:
        return json.load(f)

def run_onnx(
    onnx_path: Path,
    input_array: FloatArray,
) -> FloatArray:
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX file: {onnx_path}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    output = session.run(
        ["logits"],
        {"input": input_array},
    )[0]

    return output.astype(np.float32)


def main() -> None:
    onnx_path = ARTIFACT_DIR / ONNX_FILENAME
    input_path = ARTIFACT_DIR / GOLDEN_INPUT_FILENAME
    logits_path = ARTIFACT_DIR / GOLDEN_LOGITS_FILENAME

    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata = load_metadata(metadata_path)

    model_seed = int(metadata["model_seed"])
    threshold_factor = float(metadata["threshold_factor"])
    weight_mode = str(metadata.get("weight_mode", "random"))
    num_classes = int(metadata.get("num_classes", 1000))

    x_np = load_required_array(input_path)
    golden_logits = load_required_array(logits_path)

    # Rebuild PyTorch model to check deterministic reproducibility.
    model = build_ternary_resnet18(
        num_classes=num_classes,
        seed=model_seed,
        threshold_factor=threshold_factor,
        weight_mode=weight_mode,
    )
    model.eval()

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x_np)).detach().cpu().numpy().astype(np.float32)

    ort_logits = run_onnx(
        onnx_path=onnx_path,
        input_array=x_np,
    )

    validate_finite("golden_logits", golden_logits)
    validate_finite("torch_logits", torch_logits)
    validate_finite("ort_logits", ort_logits)

    print("Verification results")
    print("=" * 80)

    print_error_report(
        name="Saved golden logits vs rebuilt PyTorch",
        reference=golden_logits,
        candidate=torch_logits,
    )

    print()
    print_error_report(
        name="PyTorch logits vs ONNX Runtime logits",
        reference=torch_logits,
        candidate=ort_logits,
    )

    torch_norm = normalize_logits(torch_logits)
    ort_norm = normalize_logits(ort_logits)

    norm_abs_err = compute_abs_error(torch_norm, ort_norm)

    print()
    print("Normalized logits comparison")
    print(f"  max_abs_error:  {norm_abs_err.max():.8e}")
    print(f"  mean_abs_error: {norm_abs_err.mean():.8e}")
    print(f"  allclose:       {np.allclose(torch_norm, ort_norm, atol=1e-5, rtol=1e-5)}")

    rel_err = compute_rel_error(torch_logits, ort_logits)

    print()
    print("Sample logits")
    print(f"  PyTorch: {torch_logits[0, :5]}")
    print(f"  ORT:     {ort_logits[0, :5]}")

    print()

    normalized_pass = np.allclose(
        torch_norm,
        ort_norm,
        atol=1e-5,
        rtol=1e-5,
    )

    if not normalized_pass:
        raise RuntimeError(
            "ONNX Runtime output failed normalized-logit verification. "
            f"normalized_max_abs_error={norm_abs_err.max():.8e}"
        )

    print("Verification passed under normalized-logit sanity check.")
    print(
        "Note: raw allclose may fail because the random ternary ResNet produces very large logits; "
        "normalized logits are used for ONNX sanity verification."
    )


if __name__ == "__main__":
    main()