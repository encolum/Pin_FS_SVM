"""Anytime-optimization objectives shared by all VeraPin solver routes."""

from __future__ import annotations

import math
from typing import Iterable

from .progress import SolverProgressRecord, first_incumbent_time, time_to_target_gap


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
        if record.incumbent_objective is not None
        and math.isfinite(float(record.incumbent_objective))
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
