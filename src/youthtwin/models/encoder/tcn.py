"""Temporal convolutional encoder: dilated residual 1D CNN over a trial's channels.

Input (batch, channels, time) -> embedding (batch, embedding_dim). Small by design (the brief's
budget is 200k parameters): the data are a few thousand trials. With ``cond_dim`` set, a FiLM
layer follows every residual block.
"""

from __future__ import annotations

import torch
from torch import nn

from youthtwin.models.encoder.film import FiLM


class ResidualBlock(nn.Module):
    """Two dilated convolutions with a skip connection; length-preserving."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class TCNEncoder(nn.Module):
    """Dilated residual CNN with optional FiLM conditioning after each block."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden: int,
        n_blocks: int,
        kernel_size: int,
        embedding_dim: int,
        dropout: float,
        cond_dim: int | None = None,
        film_hidden: int = 32,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so that blocks preserve length")
        self.stem = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden, kernel_size, 2**k, dropout) for k in range(n_blocks)
        )
        self.films = (
            nn.ModuleList(FiLM(cond_dim, hidden, film_hidden) for _ in range(n_blocks))
            if cond_dim is not None
            else None
        )
        self.project = nn.Linear(hidden, embedding_dim)

    def feature_map(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        """Per-time-step features (batch, hidden, time); used by self-supervised pre-training."""
        h = self.stem(x)
        for k, block in enumerate(self.blocks):
            h = block(h)
            if self.films is not None:
                if g is None:
                    raise ValueError("This encoder is FiLM-conditioned and needs a growth vector")
                h = self.films[k](h, g)
        return h

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        return self.project(self.feature_map(x, g).mean(dim=-1))
