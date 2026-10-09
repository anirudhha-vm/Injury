"""De-identification (privacy rule 7).

Athletes are known to the pipeline only by a random UUID. The map from real identity to UUID
is written **outside** the repository, readable only by its owner; this module refuses to put
it anywhere inside the repo. Dates of birth are reduced to month precision.
"""

from __future__ import annotations

import csv
import uuid
from datetime import date
from pathlib import Path

from youthtwin.utils.config import repo_root

MAP_COLUMNS = ["identity", "athlete_id"]
_OWNER_READ_WRITE = 0o600


def new_athlete_id() -> str:
    """A random pseudonymous athlete ID (UUID4)."""
    return str(uuid.uuid4())


def dob_to_month(date_of_birth: date) -> str:
    """Reduce a date of birth to ``YYYY-MM``."""
    return date_of_birth.strftime("%Y-%m")


def _assert_outside_repo(path: Path) -> None:
    resolved, root = path.resolve(), repo_root().resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(
            f"The identity map must live outside the repository; refusing to use {resolved}"
        )


def read_identity_map(map_path: Path) -> dict[str, str]:
    if not map_path.is_file():
        return {}
    with map_path.open(newline="", encoding="utf-8") as handle:
        return {row["identity"]: row["athlete_id"] for row in csv.DictReader(handle)}


def pseudonymise(identities: list[str], map_path: Path) -> dict[str, str]:
    """Return ``identity -> athlete_id``, reusing IDs already in the map and adding new ones.

    ``identities`` are whatever uniquely identifies a person at the academy (e.g. a registration
    number). The updated map is written back to ``map_path`` (outside the repo, mode 600).
    """
    _assert_outside_repo(map_path)
    mapping = read_identity_map(map_path)
    for identity in identities:
        if not identity.strip():
            raise ValueError("Empty identity")
        mapping.setdefault(identity, new_athlete_id())
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.touch(mode=_OWNER_READ_WRITE, exist_ok=True)
    map_path.chmod(_OWNER_READ_WRITE)
    with map_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MAP_COLUMNS)
        writer.writerows(sorted(mapping.items()))
    return {identity: mapping[identity] for identity in identities}
