"""Figure helpers. Every figure made from synthetic data must carry the SYNTHETIC watermark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

WATERMARK_TEXT = "SYNTHETIC"
WATERMARK_GID = "synthetic-watermark"
PROVENANCE_SUFFIX = ".provenance.json"


def watermark_synthetic(figure: Figure) -> Figure:
    """Stamp a large diagonal SYNTHETIC watermark across the whole figure (idempotent)."""
    if not is_watermarked(figure):
        figure.text(
            0.5,
            0.5,
            WATERMARK_TEXT,
            fontsize=60,
            color="red",
            alpha=0.18,
            ha="center",
            va="center",
            rotation=30,
            gid=WATERMARK_GID,
            zorder=1000,
        )
    return figure


def is_watermarked(figure: Figure) -> bool:
    return any(text.get_gid() == WATERMARK_GID for text in figure.texts)


def save_figure(figure: Figure, path: Path, metadata: dict[str, Any]) -> Path:
    """Save a figure with a provenance sidecar; synthetic figures are watermarked first.

    ``metadata`` must contain a boolean ``synthetic`` (see ``utils.provenance.run_metadata``).
    """
    if "synthetic" not in metadata:
        raise ValueError("Figure metadata must state whether the data are synthetic")
    if metadata["synthetic"]:
        watermark_synthetic(figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    sidecar = path.with_name(path.name + PROVENANCE_SUFFIX)
    sidecar.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return path
