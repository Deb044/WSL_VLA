"""Leakage-resistant episode and leave-one-suite-out split utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if not all(groups):
            raise ValueError("train, validation, and test episode splits must be non-empty")
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("episode splits overlap")


def split_episodes(
    episode_ids: Sequence[str],
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> EpisodeSplit:
    unique = tuple(sorted(set(episode_ids)))
    if len(unique) < 7:
        raise ValueError("at least seven episodes are required for a three-way split")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be less than one")

    def ordering_key(episode_id: str) -> str:
        return hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()

    ordered = sorted(unique, key=ordering_key)
    n_train = max(1, int(len(ordered) * train_fraction))
    n_validation = max(1, int(len(ordered) * validation_fraction))
    if n_train + n_validation >= len(ordered):
        n_validation = 1
        n_train = len(ordered) - 2
    split = EpisodeSplit(
        train=tuple(sorted(ordered[:n_train])),
        validation=tuple(sorted(ordered[n_train : n_train + n_validation])),
        test=tuple(sorted(ordered[n_train + n_validation :])),
    )
    split.validate()
    return split


@dataclass(frozen=True)
class SuiteFold:
    name: str
    train_suites: tuple[str, ...]
    validation_task_indices: tuple[int, ...]
    test_suite: str


def leave_one_suite_out_folds(suites: Sequence[str]) -> tuple[SuiteFold, ...]:
    suites = tuple(suites)
    if len(suites) != 4 or len(set(suites)) != 4:
        raise ValueError("the publication protocol requires exactly four unique suites")
    folds = []
    for test_suite in suites:
        train_suites = tuple(suite for suite in suites if suite != test_suite)
        folds.append(
            SuiteFold(
                name=f"holdout_{test_suite}",
                train_suites=train_suites,
                validation_task_indices=(8, 9),
                test_suite=test_suite,
            )
        )
    return tuple(folds)
