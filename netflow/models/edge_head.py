"""Shared per-edge MLP head: [h_src ‖ h_dst ‖ e_feat] → class logits."""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeHead(nn.Module):
    """Two-layer MLP that maps a concatenated edge representation to class logits."""

    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
