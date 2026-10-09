"""Reproducible, independent random streams.

Each named stream (e.g. ``"growth"``, ``"injury"``) gets its own generator derived from the
master seed, so changing how one part of a simulation draws numbers does not shift the others.
"""

from __future__ import annotations

import zlib

import numpy as np


def make_rng(seed: int, *stream: str) -> np.random.Generator:
    """Return a generator for ``seed`` and an optional stream name path."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    keys = [zlib.crc32(name.encode("utf-8")) for name in stream]
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))
