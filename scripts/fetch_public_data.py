"""Download the public datasets used in M10 into the git-ignored ``data/external/``.

- Berkeley Child Guidance Study, from the ``sitar`` R package source on CRAN (GPL >= 2).
- Competitive runners' training logs and injuries, DataverseNL doi:10.34894/UWU9PV (CC0).

Every file is checked against the SHA-256 recorded in ``configs/external/``; a mismatch stops
the script, because results must be traceable to exactly these bytes.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import typer

from youthtwin.utils.config import load_config, repo_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(url: str, path: Path, sha256: str) -> None:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        typer.echo(f"downloading {url}")
        urllib.request.urlretrieve(url, path)  # fixed https URLs from configs/external
    found = _sha256(path)
    if found != sha256:
        raise typer.Exit(f"checksum mismatch for {path}: {found} != {sha256}")  # type: ignore[arg-type]
    typer.echo(f"ok  {path.relative_to(repo_root())}")


def main() -> None:
    berkeley = load_config("external/berkeley.yaml")["source"]
    archive = repo_root() / berkeley["target"] / Path(berkeley["url"]).name
    _get(berkeley["url"], archive, berkeley["sha256"])
    with tarfile.open(archive) as tar:
        tar.extract(berkeley["member"], archive.parent, filter="data")
    typer.echo(f"ok  extracted {berkeley['member']}")

    runners = load_config("external/runners.yaml")["source"]
    for item in runners["files"]:
        url = runners["url_template"].format(id=item["id"])
        _get(url, repo_root() / runners["target"] / item["name"], item["sha256"])


if __name__ == "__main__":
    typer.run(main)
