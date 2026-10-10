"""Feature-wise linear modulation (FiLM; Perez et al. 2018).

A conditioning vector ``g`` (here the athlete's growth state) rescales and shifts each feature
channel of a hidden activation ``h``:

    out = (1 + d_gamma(g)) * h + beta(g)

The generator's last layer starts at zero, so a freshly built FiLM layer is the identity and the
network starts out equal to its unconditioned version; conditioning is then learned.
"""

from __future__ import annotations

import torch
from torch import nn


class FiLM(nn.Module):
    """Modulate ``h`` of shape (batch, channels, ...) with ``g`` of shape (batch, cond_dim)."""

    def __init__(self, cond_dim: int, n_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.generator = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2 * n_channels)
        )
        last = self.generator[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        d_gamma, beta = self.generator(g).split(self.n_channels, dim=-1)
        shape = (h.shape[0], self.n_channels) + (1,) * (h.dim() - 2)
        return (1.0 + d_gamma.reshape(shape)) * h + beta.reshape(shape)
