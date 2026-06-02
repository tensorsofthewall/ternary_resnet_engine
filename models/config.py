from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

InputShape: TypeAlias = tuple[int,int,int,int]
WeightMode = Literal["random", "pretrained"]

DEFAULT_ARTIFACT_DIR: Final[Path] = Path("artifacts")
DEFAULT_INPUT_SHAPE: Final[InputShape] = (1,3,224,224)

@dataclass(frozen=True, slots=True)
class ExportConfig:
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    input_shape: InputShape = DEFAULT_INPUT_SHAPE
    model_seed: int = 42
    input_seed: int = 124
    num_classes: int = 1000
    threshold_factor: float = 0.7
    onnx_opset: int = 18
    custom_binary_version: int = 1
    weight_mode: WeightMode = "random"

    def validate(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive, got {self.num_classes}")
        if self.input_shape != DEFAULT_INPUT_SHAPE:
            raise ValueError("Only input shape {DEFAULT_INPUT_SHAPE} is supported, got {self.input_shape}")
        if not (0.0 < self.threshold_factor < 2.0):
            raise ValueError("threshold_factor must be in (0,2), got {self.threshold_factor}")
        if self.onnx_opset < 11:
            raise ValueError("onnx_opset >= 11, got {self.onnx_opset}")
        if self.weight_mode not in ("random", "pretrained"):
            raise ValueError("weight_mode must be 'random' or 'pretrained', got {self.weight_mode}")


        