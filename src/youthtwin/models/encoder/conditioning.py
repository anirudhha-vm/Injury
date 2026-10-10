"""How the growth state conditions the movement encoder: the primary ablation (hypothesis H3).

One switch, four modes, around either backbone:

- ``none``: the encoder never sees the growth state.
- ``concat_input``: the growth vector is repeated along time and appended to the input channels
  (the v2 design).
- ``late_fusion``: the encoder runs unconditioned and the growth vector is concatenated with
  the trial embedding afterwards.
- ``film``: the growth vector modulates the encoder's features after every block.

All modes produce an embedding of the same size, so everything downstream is identical and a
difference between modes can only come from how growth enters. ``SessionEncoder`` then pools
the trial embeddings of a session with attention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from youthtwin.models.encoder.stgcn import STGCNEncoder
from youthtwin.models.encoder.tcn import TCNEncoder

MODES = ("none", "concat_input", "late_fusion", "film")


def _backbone(
    config: Mapping[str, Any],
    channels: Sequence[str],
    sides: Sequence[str],
    *,
    extra_inputs: int,
    cond_dim: int | None,
) -> nn.Module:
    common: dict[str, Any] = {
        "hidden": int(config["hidden"]),
        "n_blocks": int(config["n_blocks"]),
        "embedding_dim": int(config["embedding_dim"]),
        "dropout": float(config["dropout"]),
        "cond_dim": cond_dim,
        "film_hidden": int(config["film_hidden"]),
    }
    if config["backbone"] == "tcn":
        return TCNEncoder(
            len(channels) + extra_inputs, kernel_size=int(config["kernel_size"]), **common
        )
    if config["backbone"] == "stgcn":
        return STGCNEncoder(
            channels,
            sides,
            temporal_kernel=int(config["kernel_size"]),
            extra_node_features=extra_inputs,
            **common,
        )
    raise ValueError(f"Unknown backbone '{config['backbone']}'")


class ConditionedEncoder(nn.Module):
    """Trial encoder with a selectable conditioning mode. Output: (batch, embedding_dim)."""

    def __init__(
        self,
        config: Mapping[str, Any],
        channels: Sequence[str],
        sides: Sequence[str],
        *,
        mode: str,
        cond_dim: int,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown conditioning mode '{mode}'. Known: {MODES}")
        self.mode = mode
        embedding_dim = int(config["embedding_dim"])
        self.backbone = _backbone(
            config,
            channels,
            sides,
            extra_inputs=cond_dim if mode == "concat_input" else 0,
            cond_dim=cond_dim if mode == "film" else None,
        )
        self.fuse = (
            nn.Linear(embedding_dim + cond_dim, embedding_dim) if mode == "late_fusion" else None
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return self.backbone(x)
        if self.mode == "concat_input":
            repeated = g[:, :, None].expand(-1, -1, x.shape[-1])
            return self.backbone(torch.cat([x, repeated], dim=1))
        if self.mode == "late_fusion":
            assert self.fuse is not None
            return self.fuse(torch.cat([self.backbone(x), g], dim=1))
        return self.backbone(x, g)


class SessionEncoder(nn.Module):
    """Encode every trial of a session and pool them with attention.

    ``trials``: (batch, max_trials, channels, time); ``mask``: (batch, max_trials), 1 for real
    trials; ``g``: (batch, cond_dim). A session with no real trial gives a zero embedding.
    """

    def __init__(self, encoder: ConditionedEncoder, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.score = nn.Linear(embedding_dim, 1)

    def forward(self, trials: torch.Tensor, mask: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch, n_trials = mask.shape
        flat = trials.reshape(batch * n_trials, *trials.shape[2:])
        repeated = g[:, None, :].expand(-1, n_trials, -1).reshape(batch * n_trials, -1)
        embedding = self.encoder(flat, repeated).reshape(batch, n_trials, -1)
        scores = self.score(embedding).squeeze(-1).masked_fill(mask == 0, float("-inf"))
        any_real = mask.sum(dim=1, keepdim=True) > 0
        weights = torch.softmax(torch.where(any_real, scores, torch.zeros_like(scores)), dim=1)
        weights = weights * mask
        return (weights[..., None] * embedding).sum(dim=1) * any_real
