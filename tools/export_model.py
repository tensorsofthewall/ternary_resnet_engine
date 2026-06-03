from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from models.resnet18 import (
    assert_conv_weights_are_ternary,
    build_ternary_resnet18,
    collect_conv_ternary_stats,
    register_activation_hooks,
)


ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
WeightMode = Literal["random", "pretrained"]


def write_string(f, s: str) -> None:
    data = s.encode("utf-8")
    f.write(struct.pack("<I", len(data)))
    f.write(data)


def write_array_f32(f, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    f.write(struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        f.write(struct.pack("<I", dim))
    f.write(arr.tobytes(order="C"))


def write_array_i8(f, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.int8)
    f.write(struct.pack("<I", arr.ndim))
    for dim in arr.shape:
        f.write(struct.pack("<I", dim))
    f.write(arr.tobytes(order="C"))


def export_custom_binary_v1(
    model: nn.Module,
    output_path: Path,
) -> None:
    """
    Simple custom binary export.

    This is intentionally simple for Part A:
      - Conv weights are stored as int8 {-1, 0, +1}
      - BN and FC parameters are stored as float32

    In Part B, we can add a packed 2-bit ternary format while keeping this
    v1 format for correctness debugging.
    """
    records = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            records.append(("conv2d", name, module))
        elif isinstance(module, nn.BatchNorm2d):
            records.append(("batchnorm2d", name, module))
        elif isinstance(module, nn.Linear):
            records.append(("linear", name, module))

    with output_path.open("wb") as f:
        # Magic + version
        f.write(b"TRNRES18")
        f.write(struct.pack("<I", 1))

        # Number of records
        f.write(struct.pack("<I", len(records)))

        for record_type, name, module in records:
            write_string(f, record_type)
            write_string(f, name)

            if record_type == "conv2d":
                assert isinstance(module, nn.Conv2d)

                f.write(struct.pack("<I", module.in_channels))
                f.write(struct.pack("<I", module.out_channels))

                kh, kw = module.kernel_size
                sh, sw = module.stride
                ph, pw = module.padding
                dh, dw = module.dilation

                f.write(struct.pack("<I", kh))
                f.write(struct.pack("<I", kw))
                f.write(struct.pack("<I", sh))
                f.write(struct.pack("<I", sw))
                f.write(struct.pack("<I", ph))
                f.write(struct.pack("<I", pw))
                f.write(struct.pack("<I", dh))
                f.write(struct.pack("<I", dw))
                f.write(struct.pack("<?", module.bias is not None))

                weight = module.weight.detach().cpu().numpy().astype(np.int8)
                write_array_i8(f, weight)

                if module.bias is not None:
                    bias = module.bias.detach().cpu().numpy().astype(np.float32)
                    write_array_f32(f, bias)

            elif record_type == "batchnorm2d":
                assert isinstance(module, nn.BatchNorm2d)

                f.write(struct.pack("<I", module.num_features))
                f.write(struct.pack("<f", float(module.eps)))

                write_array_f32(f, module.weight.detach().cpu().numpy())
                write_array_f32(f, module.bias.detach().cpu().numpy())
                write_array_f32(f, module.running_mean.detach().cpu().numpy())
                write_array_f32(f, module.running_var.detach().cpu().numpy())

            elif record_type == "linear":
                assert isinstance(module, nn.Linear)

                f.write(struct.pack("<I", module.in_features))
                f.write(struct.pack("<I", module.out_features))
                f.write(struct.pack("<?", module.bias is not None))

                write_array_f32(f, module.weight.detach().cpu().numpy())

                if module.bias is not None:
                    write_array_f32(f, module.bias.detach().cpu().numpy())

            else:
                raise ValueError(f"Unknown record type: {record_type}")


def export_metadata(
    model: nn.Module,
    output_path: Path,
    model_seed: int,
    input_seed: int,
    threshold_factor: float,
    weight_mode: WeightMode,
) -> None:
    conv_stats = collect_conv_ternary_stats(model)

    metadata = {
        "architecture": "resnet18",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000,
        "model_seed": model_seed,
        "input_seed": input_seed,
        "ternary_method": "TWN-style per-output-channel thresholding",
        "threshold_factor": threshold_factor,
        "conv_weight_values": [-1, 0, 1],
        "custom_binary_version": 1,
        "weight_mode": weight_mode,
        "conv_layers": [
            {
                "name": s.name,
                "shape": list(s.shape),
                "num_neg": s.num_neg,
                "num_zero": s.num_zero,
                "num_pos": s.num_pos,
                "total": s.total,
                "sparsity": s.sparsity,
            }
            for s in conv_stats
        ],
    }

    with output_path.open("w") as f:
        json.dump(metadata, f, indent=2)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ternary ResNet-18 artifacts."
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where exported artifacts are written.",
    )

    parser.add_argument(
        "--model-seed",
        type=int,
        default=0,
        help="Seed for deterministic random model initialization.",
    )

    parser.add_argument(
        "--input-seed",
        type=int,
        default=123,
        help="Seed for deterministic golden input generation.",
    )

    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=0.7,
        help="TWN-style ternarization threshold factor.",
    )

    parser.add_argument(
        "--weights",
        choices=["random", "pretrained"],
        default="random",
        help="Model initialization mode before ternarization.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    artifact_dir: Path = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_seed: int = args.model_seed
    input_seed: int = args.input_seed
    threshold_factor: float = args.threshold_factor
    weight_mode: WeightMode = args.weights

    model = build_ternary_resnet18(
        num_classes=1000,
        seed=model_seed,
        threshold_factor=threshold_factor,
        weight_mode=weight_mode,
    )
    model.eval()

    assert_conv_weights_are_ternary(model)

    torch.manual_seed(input_seed)
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    activations: dict[str, torch.Tensor] = {}
    handles = register_activation_hooks(model, activations)

    with torch.no_grad():
        logits = model(x)

    for handle in handles:
        handle.remove()

    # Save PyTorch artifacts
    torch.save(
        model.state_dict(),
        artifact_dir / "resnet18_ternary_state.pt",
    )

    np.save(
        artifact_dir / "golden_input.npy",
        x.detach().cpu().numpy(),
    )

    np.save(
        artifact_dir / "golden_logits.npy",
        logits.detach().cpu().numpy(),
    )

    np.savez(
        artifact_dir / "golden_activations.npz",
        **{name: tensor.numpy() for name, tensor in activations.items()},
    )

    x.detach().cpu().numpy().astype(np.float32).tofile(artifact_dir / "golden_input_fp32.bin")

    logits.detach().cpu().numpy().astype(np.float32).tofile(artifact_dir / "golden_logits_fp32.bin")

    tensor_metadata = {
        "golden_input": {
            "filename": "golden_input_f32.bin",
            "dtype": "float32",
            "shape": [1, 3, 224, 224],
        },
        "golden_logits": {
            "filename": "golden_logits_f32.bin",
            "dtype": "float32",
            "shape": [1, 1000],
        },
    }

    with (artifact_dir / "tensor_metadata.json").open("w") as f:
        json.dump(tensor_metadata, f, indent=2)

    # Export ONNX
    onnx_path = artifact_dir / "resnet18_ternary_fp32.onnx"

    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
    )

    # Export custom binary v1
    export_custom_binary_v1(
        model,
        artifact_dir / "model_ternary_i8_weights_v1.bin",
    )

    # Export metadata
    export_metadata(
        model=model,
        output_path=artifact_dir / "model_metadata.json",
        model_seed=model_seed,
        input_seed=input_seed,
        threshold_factor=threshold_factor,
        weight_mode=weight_mode,
    )

    if not torch.isfinite(logits).all():
        raise RuntimeError("Golden logits contain NaN or Inf.")

    print("Export complete.")
    print(f"  ONNX:              {onnx_path}")
    print(f"  PyTorch state:     {artifact_dir / 'resnet18_ternary_state.pt'}")
    print(f"  Golden input:      {artifact_dir / 'golden_input.npy'}")
    print(f"  Golden logits:     {artifact_dir / 'golden_logits.npy'}")
    print(f"  Golden activs:     {artifact_dir / 'golden_activations.npz'}")
    print(f"  Custom binary v1:  {artifact_dir / 'model_ternary_i8_weights_v1.bin'}")
    print(f"  Metadata:          {artifact_dir / 'model_metadata.json'}")
    print(f"  Logits shape:      {tuple(logits.shape)}")
    print(f"  Logits sample:     {logits[0, :5].tolist()}")
    print(f"  Weight mode:       {weight_mode}")


if __name__ == "__main__":
    main()