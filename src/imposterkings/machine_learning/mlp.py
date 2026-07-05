"""A parameterizable MLP q-evaluator (PyTorch). Any hidden-layer shape: [16], [16,16], [32,32,64], [] (linear)."""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """``in_dim -> [Linear,ReLU(,Dropout)]* over hidden_dims -> Linear(->1) -> Tanh`` (bounds to q's [-1,1])."""

    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims: List[int] = list(hidden_dims)
        layers: List[nn.Module] = []
        d = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers += [nn.Linear(d, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
