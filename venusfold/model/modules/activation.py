import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_activation_name(name: str) -> str:
    normalized = name.lower()
    if normalized not in {"relu", "silu", "gelu"}:
        raise ValueError(
            f"Unsupported activation: {name}. Expected one of: relu, silu, gelu."
        )
    return normalized


def make_activation(name: str) -> nn.Module:
    name = normalize_activation_name(name)
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    return nn.GELU()


def apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    name = normalize_activation_name(name)
    if name == "relu":
        return F.relu(x)
    if name == "silu":
        return F.silu(x)
    return F.gelu(x)
