"""Spatio-temporal graph convolutional encoder (after Yan et al. 2018) on the lower-limb graph.

The trial's channels are arranged on a small joint graph: hip, knee and ankle of each side, with
edges along each leg and across the pelvis. Each node carries its own joint's signals over time
(knee: flexion and frontal-plane angle; hip: flexion; ankle: dorsiflexion; each with velocity).
A block is a graph convolution over joints followed by a convolution over time.

Input (batch, channels, time) in the channel order of ``sequences.channel_names`` -> embedding.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from youthtwin.models.encoder.film import FiLM

JOINTS = ("hip", "knee", "ankle")
# Which angle each joint owns.
JOINT_ANGLES: dict[str, tuple[str, ...]] = {
    "hip": ("hip_flexion",),
    "knee": ("knee_flexion", "knee_fppa"),
    "ankle": ("ankle_dorsiflexion",),
}
NODE_FEATURES = 4  # up to two angles, each with its velocity; unused slots are zero


def build_graph(sides: Sequence[str]) -> tuple[list[str], torch.Tensor]:
    """Node names and the symmetrically normalised adjacency (with self-loops)."""
    nodes = [f"{side}_{joint}" for side in sides for joint in JOINTS]
    adjacency = torch.eye(len(nodes))
    index = {name: k for k, name in enumerate(nodes)}

    def link(a: str, b: str) -> None:
        adjacency[index[a], index[b]] = adjacency[index[b], index[a]] = 1.0

    for side in sides:
        link(f"{side}_hip", f"{side}_knee")
        link(f"{side}_knee", f"{side}_ankle")
    if len(sides) == 2:
        link(f"{sides[0]}_hip", f"{sides[1]}_hip")
    degree = adjacency.sum(dim=1)
    scale = degree.rsqrt()
    return nodes, scale[:, None] * adjacency * scale[None, :]


def node_layout(channels: Sequence[str], nodes: Sequence[str]) -> torch.Tensor:
    """Index tensor (nodes, NODE_FEATURES) into the channel axis; -1 where a slot is unused."""
    position = {name: k for k, name in enumerate(channels)}
    layout = torch.full((len(nodes), NODE_FEATURES), -1, dtype=torch.long)
    for n, node in enumerate(nodes):
        side, joint = node.split("_", 1)
        slot = 0
        for angle in JOINT_ANGLES[joint]:
            for suffix in ("", "_velocity"):
                layout[n, slot] = position.get(f"{side}_{angle}{suffix}", -1)
                slot += 1
    return layout


class STGCNBlock(nn.Module):
    def __init__(self, channels: int, temporal_kernel: int, dropout: float) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, (temporal_kernel, 1), padding=((temporal_kernel - 1) // 2, 0)
            ),
            nn.GroupNorm(1, channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time, nodes)
        mixed = torch.einsum("bctv,vw->bctw", self.spatial(x), adjacency)
        return self.activation(x + self.temporal(mixed))


class STGCNEncoder(nn.Module):
    """Graph encoder with optional FiLM conditioning after each block."""

    def __init__(
        self,
        channels: Sequence[str],
        sides: Sequence[str],
        *,
        hidden: int,
        n_blocks: int,
        temporal_kernel: int,
        embedding_dim: int,
        dropout: float,
        cond_dim: int | None = None,
        film_hidden: int = 32,
        extra_node_features: int = 0,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd so that blocks preserve length")
        nodes, adjacency = build_graph(sides)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("layout", node_layout(channels, nodes))
        self.n_signal_channels = len(channels)
        self.stem = nn.Conv2d(NODE_FEATURES + extra_node_features, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            STGCNBlock(hidden, temporal_kernel, dropout) for _ in range(n_blocks)
        )
        self.films = (
            nn.ModuleList(FiLM(cond_dim, hidden, film_hidden) for _ in range(n_blocks))
            if cond_dim is not None
            else None
        )
        self.project = nn.Linear(hidden, embedding_dim)

    def to_graph(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, channels, time) -> (batch, node_features, time, nodes).

        Channels beyond the joint signals (used by input concatenation) are copied to every node.
        """
        layout: torch.Tensor = self.layout  # type: ignore[assignment]
        signals, extra = x[:, : self.n_signal_channels], x[:, self.n_signal_channels :]
        padded = torch.cat([signals, torch.zeros_like(signals[:, :1])], dim=1)  # index -1 -> zeros
        gathered = padded[:, layout]  # (batch, nodes, features, time)
        graph = gathered.permute(0, 2, 3, 1)
        if extra.shape[1]:
            graph = torch.cat([graph, extra[..., None].expand(-1, -1, -1, graph.shape[-1])], dim=1)
        return graph

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        adjacency: torch.Tensor = self.adjacency  # type: ignore[assignment]
        h = self.stem(self.to_graph(x))
        for k, block in enumerate(self.blocks):
            h = block(h, adjacency)
            if self.films is not None:
                if g is None:
                    raise ValueError("This encoder is FiLM-conditioned and needs a growth vector")
                h = self.films[k](h, g)
        return self.project(h.mean(dim=(2, 3)))
