"""Integrated gradients for the movement encoder (Sundararajan et al. 2017).

Attributes a network output to its input by integrating the gradient along the straight path
from a baseline to the input:

    attribution = (input - baseline) * mean over steps of d output / d input

**Completeness**: the attributions sum to ``output(input) - output(baseline)``, which the tests
check. The baseline for standardised trials is all zeros, i.e. the training-mean movement, so
attributions read as "relative to an average trial". They describe the fitted network, not the
biomechanics of injury.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def integrated_gradients(
    function: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    baseline: torch.Tensor | None = None,
    *,
    steps: int = 64,
) -> torch.Tensor:
    """Attributions with the shape of ``inputs``.

    ``function`` maps a batch shaped like ``inputs`` to one scalar per batch element.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    reference = torch.zeros_like(inputs) if baseline is None else baseline
    difference = inputs - reference
    total = torch.zeros_like(inputs)
    # Midpoint rule along the path.
    for k in range(steps):
        point = (reference + (k + 0.5) / steps * difference).detach().requires_grad_(True)
        output = function(point).sum()
        (gradient,) = torch.autograd.grad(output, point)
        total = total + gradient
    return difference * total / steps


def channel_time_attribution(attributions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum attributions over a session's real trials: (batch, trials, C, T) -> (batch, C, T)."""
    return (attributions * mask[:, :, None, None]).sum(dim=1)
