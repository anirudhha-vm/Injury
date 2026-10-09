"""Run metadata: every artifact is traceable to a config, a seed and a commit (rule 4)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from youthtwin import __version__
from youthtwin.utils.config import repo_root


def config_hash(config: Mapping[str, Any]) -> str:
    """Short, stable hash of a config (key order does not matter)."""
    canonical = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def git_commit() -> str | None:
    """Current commit hash, with ``-dirty`` if there are uncommitted changes; None outside git."""
    git = shutil.which("git")
    if git is None:
        return None
    root = str(repo_root())
    head = subprocess.run(
        [git, "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if head.returncode != 0:
        return None
    status = subprocess.run(
        [git, "-C", root, "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return head.stdout.strip() + ("-dirty" if status.stdout.strip() else "")


def run_metadata(*, seed: int, config: Mapping[str, Any], synthetic: bool) -> dict[str, Any]:
    """Metadata stored next to every generated artifact."""
    return {
        "synthetic": synthetic,
        "seed": seed,
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "youthtwin_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
