from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

import torch
import torch.nn as nn
import torchvision.models as tv_models

WeightMode = Literal["random", "pretrained"]

@dataclass
class ConvTernaryStats:
    name: str
    shape: tuple[int,...]
    num_neg: int
    num_zero: int
    num_pos: int
    total: int

    @property
    def sparsity(self) -> float:
        if self.total == 0:
            return 0.0
        return self.num_zero / self.total
    
    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": self.shape,
            "num_neg": self.num_neg,
            "num_zero": self.num_zero,
            "num_pos": self.num_pos,
            "total": self.total,
            "sparsity": self.sparsity
        }

def build_resnet18(num_classes: int = 1000, seed: int = 42, weight_mode: WeightMode = "random") -> nn.Module:
    """
    Builds a deterministic random-initialized ResNet-18 model with the specified number of classes and random seed.

    Accuracy is not relevant here, as the model is not trained. The random seed ensures that the same weights are generated each time for reproducibility.
    """
    torch.manual_seed(seed)
    if weight_mode == "random":
        weights = None
    elif weight_mode == "pretrained":
        weights = tv_models.ResNet18_Weights.DEFAULT
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    model = tv_models.resnet18(weights=weights, num_classes=num_classes)
    model.eval()
    return model

def ternarize_weights(
        weight: torch.Tensor,
        threshold_factor: float = 0.7
) -> nn.Module:
    """
    Ternary-Weight-Networks style ternarization of weight tensors. 
    https://www.emergentmind.com/topics/ternary-weight-networks-twns

    For Conv2D weights shaped NCHW, threshold per output channel and compute delta as:
    delta = threshold_factor * mean(abs(W_oc))
    
    where W_oc are the weights per output channel oc.

    Values:
    W > delta -> +1
    W < -delta -> -1
    otherwise -> 0
    """

    if weight.ndim != 4:
        raise ValueError(f"Expected 4D weight tensor for Conv2D, got shape {weight.ndim}")
    
    reduce_dims = tuple(range(1, weight.ndim))  # reduce over all dims except output channels
    delta = threshold_factor * weight.abs().mean(dim=reduce_dims, keepdim=True)

    ternary_weight = torch.where(
        weight > delta, 
        torch.ones_like(weight),
        torch.where(
            weight < -delta, 
            -torch.ones_like(weight), 
            torch.zeros_like(weight)
        )
    )

    return ternary_weight


def ternarize_resnet18(
    model: nn.Module,
    threshold_factor: float = 0.7
) -> nn.Module:
    """
    In-place ternarization of all Conv2D Layers in ResNet18.

    Returns a new model with ternarized weights.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            with torch.no_grad():
                ternary_weight = ternarize_weights(module.weight.data, threshold_factor)
                module.weight.data.copy_(ternary_weight)
    
    return model

def build_ternary_resnet18(
    num_classes: int = 1000,
    seed: int = 42,
    threshold_factor: float = 0.7,
    weight_mode: WeightMode = "random",
) -> nn.Module:
    """
    Builds a ResNet-18 model with ternarized weights using the specified parameters.
    """
    model = build_resnet18(num_classes=num_classes, seed=seed, weight_mode=weight_mode)
    ternary_model = ternarize_resnet18(model, threshold_factor=threshold_factor)
    return ternary_model

def assert_conv_weights_are_ternary(model: nn.Module) -> None:
    allowed = {-1.0, 0.0, 1.0}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        values = torch.unique(module.weight.detach().cpu())
        values_set = set(float(v.item()) for v in values)

        if not values_set.issubset(allowed):
            raise AssertionError(
                f"{name} has non-ternary values: {sorted(values_set)}"
            )

def collect_conv_ternary_stats(model: nn.Module) -> list[ConvTernaryStats]:
    stats = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        
        weight = module.weight.detach().cpu()
        num_neg = int((weight == -1).sum().item())
        num_zero = int((weight == 0).sum().item())
        num_pos = int((weight == 1).sum().item())
        total = int(weight.numel())
        stats.append(ConvTernaryStats(
            name=name,
            shape=tuple(weight.shape),
            num_neg=num_neg,
            num_zero=num_zero,
            num_pos=num_pos,
            total=total
        ))
    
    return stats


def register_activation_hooks(
    model: nn.Module,
    activations: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Register hooks for layers useful for debugging the C++ engine.

    We save module outputs after Conv, BN, ReLU, pooling, and Linear layers.
    """
    handles = []

    useful_types = (
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Linear,
    )

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, useful_types):
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    return handles