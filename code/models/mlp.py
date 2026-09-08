import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """Opacus-compatible MLP (no BatchNorm — uses ReLU + Dropout only)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_weights(self) -> list:
        return [p.data.cpu().clone() for p in self.parameters()]

    def set_weights(self, weights: list) -> None:
        for p, w in zip(self.parameters(), weights):
            p.data.copy_(w.to(p.device))
