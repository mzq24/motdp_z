import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    """Residual bottleneck adapter for parameter-efficient finetuning."""

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(rank, dim, bias=False)

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.act(self.down_proj(x))
        return x + self.up_proj(hidden)
