"""Pack ternary conv weights into a dense 2-bit representation.

Encoding (2 bits per weight):
  00 = 0
  01 = +1
  10 = -1
  11 = unused (padding)

Weights are packed 4 per byte, LSB first (bits 0-1 = weight 0, bits 2-3 = weight 1, ...).
A header stores layer count, then per-layer (name_len, name, weight_count).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


ARTIFACT_DIR = Path("artifacts")
OUT_PATH = ARTIFACT_DIR / "model_ternary_weights_2bit_packed.bin"


TERNARY_ENCODE = {-1: 0b10, 0: 0b00, 1: 0b01}


def pack_layer_weights(weights: list[int]) -> bytes:
    """Pack a list of ternary weights (-1, 0, +1) into 2-bit-per-weight bytes."""
    n = len(weights)
    # Pad to multiple of 4
    padded = weights + [0] * ((-n) % 4)
    out = bytearray(len(padded) // 4)
    for i in range(0, len(padded), 4):
        byte = 0
        for j in range(4):
            code = TERNARY_ENCODE[padded[i + j]]
            byte |= (code << (j * 2))
        out[i // 4] = byte
    return bytes(out)


def load_conv_weights_from_metadata(artifact_dir: Path) -> list[tuple[str, list[int]]]:
    """
    Load ternary conv weights from model_metadata.json.
    Returns list of (layer_name, flat_weights) tuples.
    """
    meta_path = artifact_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run: uv run python tools/export_model.py"
        )

    # Try to load from state dict first (has actual weight values)
    state_path = artifact_dir / "resnet18_ternary_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} not found. Run: uv run python tools/export_model.py"
        )

    # Lazy import torch only here
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=True)

    with open(meta_path) as f:
        meta = json.load(f)

    conv_layer_names = [c["name"] for c in meta["conv_layers"]]

    result = []
    for name in conv_layer_names:
        key = f"{name}.weight"
        if key not in state:
            print(f"Warning: {key} not found in state dict, skipping", file=sys.stderr)
            continue
        w = state[key].cpu().numpy().astype(float).round().astype(int).flatten().tolist()
        # Verify ternary
        unique_vals = set(w)
        if not unique_vals.issubset({-1, 0, 1}):
            raise ValueError(f"{name}: non-ternary values {unique_vals - {-1, 0, 1}}")
        result.append((name, w))

    return result


def write_packed_binary(layers: list[tuple[str, list[int]]], out_path: Path) -> int:
    """Write 2-bit packed binary file. Returns total bytes written."""
    total_weights = sum(len(w) for _, w in layers)

    with open(out_path, "wb") as f:
        # File header
        f.write(b"TERN2BIT")            # magic
        f.write(struct.pack("<I", 1))   # version
        f.write(struct.pack("<I", len(layers)))  # layer count
        f.write(struct.pack("<Q", total_weights))  # total weight count

        for name, weights in layers:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(weights)))

            packed = pack_layer_weights(weights)
            f.write(struct.pack("<I", len(packed)))
            f.write(packed)

    return out_path.stat().st_size


def main() -> None:
    print("Loading ternary conv weights from state dict...", flush=True)
    layers = load_conv_weights_from_metadata(ARTIFACT_DIR)

    total_weights = sum(len(w) for _, w in layers)
    print(f"Layers: {len(layers)}, total weights: {total_weights:,}")

    theoretical_2bit = (total_weights * 2 + 7) // 8
    print(f"Theoretical 2-bit packed size: {theoretical_2bit / 1024 / 1024:.2f} MB")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    size_bytes = write_packed_binary(layers, OUT_PATH)

    overhead = size_bytes - theoretical_2bit
    print(f"\nWritten: {OUT_PATH}")
    print(f"  File size:   {size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Payload:     {theoretical_2bit / 1024 / 1024:.2f} MB")
    print(f"  Header+meta: {overhead} bytes")

    # Verify: round-trip one layer
    name0, w0 = layers[0]
    packed = pack_layer_weights(w0)
    recovered = []
    for byte in packed:
        for j in range(4):
            code = (byte >> (j * 2)) & 0b11
            if code == 0b00:
                recovered.append(0)
            elif code == 0b01:
                recovered.append(1)
            elif code == 0b10:
                recovered.append(-1)
    recovered = recovered[:len(w0)]
    assert recovered == w0, f"Round-trip verification failed for {name0}"
    print(f"\nRound-trip verification passed for {name0} ({len(w0)} weights)")


if __name__ == "__main__":
    main()
