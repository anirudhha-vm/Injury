"""Data splits (non-negotiable N3): everything is grouped by athlete.

No athlete ever appears on both sides of a split, so overlapping athlete-weeks from one person
cannot leak between training and testing (v2 split at the window level).

- ``grouped_stratified_folds``: K folds of athletes, balanced on whether the athlete ever has
  the event, deterministic for a seed.
- ``nested_folds``: inner folds drawn only from an outer fold's training athletes (for tuning).
- ``temporal_holdout``: the last part of follow-up time is held out for every athlete; training
  rows whose outcome window reaches into the held-out period are removed.
- ``leave_one_academy_out``.

Every split is returned as boolean row masks over the athlete-week table and can be checked
with ``assert_no_group_overlap``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Split:
    name: str
    train: BoolArray
    test: BoolArray


def assert_no_group_overlap(groups: pd.Series, split: Split) -> None:
    """Raise if any athlete has rows in both the training and the test part."""
    both = set(groups[split.train]) & set(groups[split.test])
    if both:
        raise AssertionError(f"Split '{split.name}': {len(both)} athlete(s) in both train and test")


def _fold_of_each_group(
    groups: pd.Series, event: pd.Series, n_splits: int, seed: int
) -> dict[str, int]:
    """Assign each athlete to a fold, balancing athletes with and without the event."""
    per_athlete = event.astype(bool).groupby(groups).any()
    if len(per_athlete) < n_splits:
        raise ValueError(f"Cannot make {n_splits} folds from {len(per_athlete)} athletes")
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    offset = 0
    for has_event in (True, False):
        members = np.array(sorted(per_athlete.index[per_athlete == has_event]))
        rng.shuffle(members)
        for position, athlete in enumerate(members):
            assignment[str(athlete)] = (position + offset) % n_splits
        offset += len(members)  # continue the rotation so fold sizes stay even
    return assignment


def grouped_stratified_folds(
    groups: pd.Series, event: pd.Series, *, n_splits: int, seed: int, name: str = "fold"
) -> list[Split]:
    """K athlete-grouped folds; each athlete is in the test part of exactly one."""
    fold = groups.astype(str).map(_fold_of_each_group(groups.astype(str), event, n_splits, seed))
    splits = [
        Split(f"{name}{k}", (fold != k).to_numpy(), (fold == k).to_numpy()) for k in range(n_splits)
    ]
    for split in splits:
        assert_no_group_overlap(groups, split)
    return splits


def nested_folds(
    groups: pd.Series,
    event: pd.Series,
    *,
    n_outer: int,
    n_inner: int,
    seed: int,
) -> Iterator[tuple[Split, list[Split]]]:
    """Outer folds, each with inner folds made only from its training athletes."""
    for k, outer in enumerate(grouped_stratified_folds(groups, event, n_splits=n_outer, seed=seed)):
        inner_assignment = _fold_of_each_group(
            groups[outer.train].astype(str), event[outer.train], n_inner, seed + 1 + k
        )
        inner_fold = groups.astype(str).map(inner_assignment)  # NaN for outer-test athletes
        inner = [
            Split(
                f"{outer.name}_inner{j}",
                outer.train & (inner_fold != j).to_numpy() & inner_fold.notna().to_numpy(),
                outer.train & (inner_fold == j).to_numpy(),
            )
            for j in range(n_inner)
        ]
        for split in inner:
            assert_no_group_overlap(groups, split)
            if (split.train & outer.test).any() or (split.test & outer.test).any():
                raise AssertionError("Inner fold uses outer-test rows")
        yield outer, inner


def temporal_holdout(week_start: pd.Series, *, fraction: float, horizon_weeks: int) -> Split:
    """Hold out the last ``fraction`` of calendar weeks.

    Training rows whose outcome window (the following ``horizon_weeks``) overlaps the held-out
    period are dropped from training, otherwise their labels would be defined by held-out time.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    weeks = pd.to_datetime(week_start)
    calendar = np.sort(weeks.unique())
    first_test = calendar[int(np.floor(len(calendar) * (1.0 - fraction)))]
    test = (weeks >= first_test).to_numpy()
    train = (weeks + pd.Timedelta(weeks=horizon_weeks) < first_test).to_numpy()
    return Split("temporal_holdout", train, test)


def leave_one_academy_out(academy: pd.Series, groups: pd.Series) -> list[Split]:
    """One split per academy (only meaningful with three or more academies)."""
    splits = [
        Split(f"academy_{value}", (academy != value).to_numpy(), (academy == value).to_numpy())
        for value in sorted(academy.unique())
    ]
    for split in splits:
        assert_no_group_overlap(groups, split)
    return splits
