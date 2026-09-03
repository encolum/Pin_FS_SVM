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


def primal_integral(
    records: Iterable[SolverProgressRecord],
    *,
    horizon: float,
    reference_objective: float | None = None,
    no_incumbent_penalty: float = 1.0,
) -> float:
    """Integrate normalized primal error over time for a minimization problem."""
    horizon = float(horizon)
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be a finite non-negative number")
    penalty = float(no_incumbent_penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("no_incumbent_penalty must be finite and non-negative")
    trajectory = sorted(list(records), key=lambda record: record.elapsed_seconds)
    incumbents = [
        float(record.incumbent_objective)
        for record in trajectory
        if _is_finite(record.incumbent_objective)
    ]
    if reference_objective is None:
        reference = min(incumbents) if incumbents else None
    else:
        reference = float(reference_objective)
        if not math.isfinite(reference):
            raise ValueError("reference_objective must be finite")

    previous_time = 0.0
    current_error = penalty
    integral = 0.0
    for record in trajectory:
        timestamp = min(horizon, max(previous_time, float(record.elapsed_seconds)))
        integral += (timestamp - previous_time) * current_error
        previous_time = timestamp
        if record.incumbent_objective is not None and reference is not None:
            incumbent = float(record.incumbent_objective)
            if math.isfinite(incumbent):
                scale = max(abs(reference), 1e-12)
                current_error = max(0.0, (incumbent - reference) / scale)
        if previous_time >= horizon:
            break
    integral += max(0.0, horizon - previous_time) * current_error
    return float(integral)


def solver_progress_summary(
    records: Iterable[SolverProgressRecord],
    *,
    horizon: float,
    reference_objective: float | None = None,
) -> dict[str, float | int | None]:
    """Return the progress fields required by the result schema."""
    trajectory = list(records)
    final = trajectory[-1] if trajectory else None
    return {
        "first_feasible_time": first_incumbent_time(trajectory),
        "time_to_10pct_gap": time_to_target_gap(trajectory, 0.10),
        "time_to_5pct_gap": time_to_target_gap(trajectory, 0.05),
        "time_to_1pct_gap": time_to_target_gap(trajectory, 0.01),
        "time_to_0_1pct_gap": time_to_target_gap(trajectory, 0.001),
        "primal_integral": primal_integral(
            trajectory,
            horizon=horizon,
            reference_objective=reference_objective,
        ),
        "final_objective": None if final is None else final.incumbent_objective,
        "final_best_bound": None if final is None else final.best_bound,
        "final_gap": None if final is None else final.relative_gap,
        "node_count": None if final is None else final.node_count,
    }


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))
