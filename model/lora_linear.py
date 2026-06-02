"""LoRA linear layer for PEFT diffusion experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen base linear layer plus trainable low-rank update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = float(alpha) / float(rank)

        self.register_buffer("base_weight", torch.empty(out_features, in_features))
        if has_bias:
            self.register_buffer("base_bias", torch.zeros(out_features))
        else:
            self.register_buffer("base_bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float = 1.0) -> "LoRALinear":
        layer = cls(linear.in_features, linear.out_features, rank, alpha, linear.bias is not None)
        with torch.no_grad():
            layer.base_weight.copy_(linear.weight.data)
            if linear.bias is not None:
                layer.base_bias.copy_(linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.base_weight, self.base_bias)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out
