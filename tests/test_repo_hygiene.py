"""Privacy rule 7: no raw video and no secrets tracked by git."""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from youthtwin.utils.config import repo_root

VIDEO = re.compile(r"\.(mp4|mov|avi|mkv|m4v|webm)$", re.IGNORECASE)
ENV_FILE = re.compile(r"(^|/)\.env(\..*)?$")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None or not (repo_root() / ".git").exists():
        pytest.skip("not a git checkout")
    return subprocess.run(
        [git, "-C", str(repo_root()), *args], capture_output=True, text=True, check=False
    )


def test_no_video_tracked() -> None:
    tracked = _git("ls-files").stdout.splitlines()
    assert [f for f in tracked if VIDEO.search(f)] == []


def test_no_env_file_tracked() -> None:
    tracked = _git("ls-files").stdout.splitlines()
    assert [f for f in tracked if ENV_FILE.search(f) and not f.endswith(".env.example")] == []


@pytest.mark.parametrize(
    "path",
    [
        "data/raw_video_LOCAL_ONLY/athlete_dvj.mp4",
        "anywhere/clip.MOV",
        ".env",
        "data/keypoints/session.parquet",
        "data/synthetic/cohort.parquet",
    ],
)
def test_sensitive_paths_are_git_ignored(path: str) -> None:
    assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not git-ignored"


def test_precommit_blocks_video_and_secrets() -> None:
    config = (repo_root() / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    for hook in ("block-raw-video", "block-env-files", "block-secret-keys"):
        assert hook in config
