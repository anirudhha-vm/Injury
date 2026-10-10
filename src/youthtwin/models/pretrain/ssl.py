"""Self-supervised pre-training on unlabelled trials (no injury labels are involved).

Objective: masked-segment reconstruction. A contiguous span of time steps is hidden in every
channel and the network reconstructs it from the rest; the loss is the mean squared error on the
hidden span only. This uses every screening trial, including those of athletes who never get
injured, which is most of the data.

Augmentations (applied to standardised trials of shape (batch, channels, time)):

- time-warp: the trial is resampled by a factor in the configured range (e.g. 0.9-1.1);
- Gaussian noise;
- a small constant offset per angle channel, standing in for a small camera rotation.

There is deliberately **no left/right mirroring**: channels are limb-specific and swapping them
would destroy the limb label.

Pre-training applies to the TCN backbone without FiLM; its weights initialise the conditioned
encoders, whose FiLM layers start as the identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from youthtwin.models.encoder.tcn import TCNEncoder

FloatArray = NDArray[np.float64]


def time_warp(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Play the trial ``factor`` times faster (> 1) or slower (< 1), keeping its length.

    Positions beyond the end of a sped-up trial hold the final value.
    """
    n_time = x.shape[-1]
    position = torch.clamp(torch.arange(n_time, dtype=x.dtype) * factor, max=n_time - 1)
    low = position.floor().long()
    high = torch.clamp(low + 1, max=n_time - 1)
    weight = position - low
    return x[..., low] * (1.0 - weight) + x[..., high] * weight


def augment(x: torch.Tensor, generator: torch.Generator, config: Mapping[str, Any]) -> torch.Tensor:
    """One random augmentation of a batch; channel order (and so limb identity) is unchanged."""
    low, high = config["time_warp_range"]
    factor = float(torch.empty(1).uniform_(low, high, generator=generator))
    out = time_warp(x, factor)
    out = out + float(config["noise_sd"]) * torch.randn(out.shape, generator=generator)
    offset = float(config["offset_sd"]) * torch.randn(out.shape[:2], generator=generator)
    return out + offset[..., None]


def mask_span(
    x: torch.Tensor, generator: torch.Generator, fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero a random contiguous span (same for all channels of a trial). Returns (masked, mask)."""
    batch, _, n_time = x.shape
    length = max(1, round(fraction * n_time))
    start = torch.randint(0, n_time - length + 1, (batch,), generator=generator)
    steps = torch.arange(n_time)[None, :]
    hidden = (steps >= start[:, None]) & (steps < (start + length)[:, None])
    return x.masked_fill(hidden[:, None, :], 0.0), hidden


class MaskedReconstruction(nn.Module):
    """Encoder feature map -> per-time-step reconstruction of the input channels."""

    def __init__(self, encoder: TCNEncoder, hidden: int, n_channels: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Conv1d(hidden, n_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder.feature_map(x))


def pretrain(
    trials: FloatArray, encoder_config: Mapping[str, Any], config: Mapping[str, Any], *, seed: int
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Pre-train a TCN on standardised trials (n, channels, time).

    Returns the encoder's weights and the loss per epoch. Pass only training athletes' trials:
    held-out athletes must not shape the representation either.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    data = torch.as_tensor(trials, dtype=torch.float32)
    n_channels = data.shape[1]
    encoder = TCNEncoder(
        n_channels,
        hidden=int(encoder_config["hidden"]),
        n_blocks=int(encoder_config["n_blocks"]),
        kernel_size=int(encoder_config["kernel_size"]),
        embedding_dim=int(encoder_config["embedding_dim"]),
        dropout=float(encoder_config["dropout"]),
    )
    model = MaskedReconstruction(encoder, int(encoder_config["hidden"]), n_channels)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    batch_size = int(config["batch_size"])
    history: list[float] = []
    for _ in range(int(config["epochs"])):
        order = torch.randperm(len(data), generator=generator)
        total, count = 0.0, 0
        for start in range(0, len(data), batch_size):
            clean = augment(data[order[start : start + batch_size]], generator, config)
            masked, hidden = mask_span(clean, generator, float(config["mask_fraction"]))
            error = (model(masked) - clean) ** 2
            loss = error.masked_select(hidden[:, None, :].expand_as(error)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(clean)
            count += len(clean)
        history.append(total / count)
    return {k: v.detach().clone() for k, v in encoder.state_dict().items()}, history


def load_pretrained(target: nn.Module, weights: Mapping[str, torch.Tensor]) -> list[str]:
    """Copy pre-trained weights into ``target`` where name and shape match; return the names.

    Layers that differ by design (FiLM generators, an input layer widened by concatenation) are
    left at their initial values.
    """
    own = target.state_dict()
    matched = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    target.load_state_dict(own)
    return sorted(matched)
