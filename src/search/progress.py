"""Solver-progress records and deterministic trajectory helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class SolverProgressRecord:
    """One point in a minimization solver's progress trajectory."""

    elapsed_seconds: float
    incumbent_objective: float | None
    best_bound: float | None
    relative_gap: float | None
    node_count: int | None
    solution_count: int | None


def first_incumbent_time(records: Iterable[SolverProgressRecord]) -> float | None:
    """Return when the first finite incumbent became available."""
    for record in records:
        if _is_finite(record.incumbent_objective):
            return float(record.elapsed_seconds)
    return None


def time_to_target_gap(
    records: Iterable[SolverProgressRecord], target_gap: float
) -> float | None:
    """Return the earliest time at which the relative MIP gap met ``target_gap``."""
    target_gap = float(target_gap)
    if not math.isfinite(target_gap) or target_gap < 0:
        raise ValueError("target_gap must be a finite non-negative number")
    for record in records:
        if _is_finite(record.relative_gap) and float(record.relative_gap) <= target_gap:
            return float(record.elapsed_seconds)
    return None


def validate_progress_trajectory(
    records: Iterable[SolverProgressRecord], *, tolerance: float = 1e-9
) -> None:
    """Validate ordering and the minimization incumbent monotonicity invariant."""
    previous_time = -math.inf
    previous_incumbent = math.inf
    for record in records:
        elapsed = float(record.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < -tolerance:
            raise ValueError("progress timestamps must be finite and non-negative")
        if elapsed + tolerance < previous_time:
            raise ValueError("progress timestamps must be nondecreasing")
        previous_time = max(previous_time, elapsed)

        incumbent = record.incumbent_objective
        if _is_finite(incumbent):
            incumbent = float(incumbent)
            if incumbent > previous_incumbent + tolerance:
                raise ValueError("incumbent objectives must be nonincreasing")
            previous_incumbent = min(previous_incumbent, incumbent)


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))
