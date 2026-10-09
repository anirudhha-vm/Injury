"""Locate the repository and load YAML configs from ``configs/`` (rule 10: config-driven)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

ROOT_ENV_VAR = "YOUTHTWIN_ROOT"


def repo_root() -> Path:
    """Return the repository root (override with the ``YOUTHTWIN_ROOT`` environment variable)."""
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return repo_root() / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML config as a plain dict.

    ``path`` is absolute, or relative to ``configs/`` (e.g. ``"growth/equations.yaml"``).
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = configs_dir() / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"Config not found: {resolved}")
    container = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"Config must be a YAML mapping at top level: {resolved}")
    return {str(key): value for key, value in container.items()}
