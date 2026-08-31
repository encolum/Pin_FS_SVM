"""Lazy DOcplex adapter for manuscript-level CPLEX solver parity."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix, issparse


@dataclass
class CplexResult:
    x: np.ndarray
    fun: float
    status: str
    message: str
    mip_dual_bound: float | None = None
    mip_gap: float | None = None
    mip_node_count: int | None = None
    backend: str = "docplex-cplex"
    progress: list[Any] = field(default_factory=list)
    mip_start_status: str | None = None
    model_build_time: float = 0.0


def solve_docplex(
    linear_objective: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraint_matrix: Any,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    integrality: np.ndarray | None = None,
    quadratic_indices: Iterable[int] = (),
    time_limit: float | None = None,
    mip_gap: float | None = None,
    threads: int = 1,
    model_name: str = "corrected-model",
    mip_start: Any | None = None,
    collect_progress: bool = False,
    deadline: float | None = None,
) -> CplexResult:
    """Solve a linear or convex quadratic model through local DOcplex/CPLEX."""
    build_started = perf_counter()
    try:
        from docplex.mp.constants import EffortLevel, WriteLevel
        from docplex.mp.model import Model
        from docplex.mp.progress import ProgressClock, ProgressDataRecorder
    except ImportError as exc:
        raise RuntimeError(
            "solver.backend='cplex' requires the optional packages in requirements-cplex.txt"
        ) from exc

    c = np.asarray(linear_objective, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    row_lower = np.asarray(constraint_lower, dtype=float)
    row_upper = np.asarray(constraint_upper, dtype=float)
    integer = np.zeros(c.size, dtype=int) if integrality is None else np.asarray(integrality, dtype=int)
    if not (c.shape == lower.shape == upper.shape == integer.shape):
        raise ValueError("objective, bounds, and integrality vectors must have equal length")
    matrix = csr_matrix(constraint_matrix) if not issparse(constraint_matrix) else constraint_matrix.tocsr()
    if matrix.shape != (row_lower.size, c.size) or row_lower.shape != row_upper.shape:
        raise ValueError("constraint matrix and bound dimensions disagree")

    model = Model(name=model_name, log_output=False)
    try:
        variables = []
        for index in range(c.size):
            if index % 128 == 0:
                _check_deadline(deadline)
            if integer[index]:
                variable = model.binary_var(name=f"x_{index}")
                variable.lb = max(0.0, float(lower[index]))
                variable.ub = min(1.0, float(upper[index]))
                variables.append(variable)
            else:
                lb = -model.infinity if np.isneginf(lower[index]) else float(lower[index])
                ub = model.infinity if np.isposinf(upper[index]) else float(upper[index])
                variables.append(model.continuous_var(lb=lb, ub=ub, name=f"x_{index}"))

        for row in range(matrix.shape[0]):
            if row % 128 == 0:
                _check_deadline(deadline)
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            expression = model.sum(
                float(value) * variables[int(column)]
                for column, value in zip(matrix.indices[start:end], matrix.data[start:end])
            )
            if np.isfinite(row_lower[row]):
                model.add_constraint(expression >= float(row_lower[row]))
            if np.isfinite(row_upper[row]):
                model.add_constraint(expression <= float(row_upper[row]))

        objective = model.sum(float(value) * variables[index] for index, value in enumerate(c) if value)
        quadratic = tuple(int(index) for index in quadratic_indices)
        if quadratic:
            objective += model.sum(0.5 * variables[index] * variables[index] for index in quadratic)
        model.minimize(objective)

        model.parameters.threads = int(threads)
        _check_deadline(deadline)
        effective_time_limit = time_limit
        if deadline is not None:
            remaining = float(deadline) - perf_counter()
            if remaining <= 1e-6:
                raise RuntimeError("CPLEX model construction exhausted the wall-clock budget")
            effective_time_limit = (
                remaining
                if effective_time_limit is None
                else min(float(effective_time_limit), remaining)
            )
        if effective_time_limit is not None:
            model.parameters.timelimit = float(effective_time_limit)
        if np.any(integer) and mip_gap is not None:
            model.parameters.mip.tolerances.mipgap = float(mip_gap)

        mip_start_status = None
        log_stream: StringIO | None = None
        if mip_start is not None:
            start_values = _validate_mip_start_vector(
                mip_start,
                lower_bounds=lower,
                upper_bounds=upper,
                integrality=integer,
            )
            effort_level = _mip_start_effort_level(mip_start.effort_level, EffortLevel)
            start_solution = model.new_solution(
                {variable: float(value) for variable, value in zip(variables, start_values)},
                name=str(mip_start.name),
            )
            registered = model.add_mip_start(
                start_solution,
                effort_level=effort_level,
                write_level=WriteLevel.AllVars,
                complete_vars=True,
            )
            if registered is None:
                raise ValueError("CPLEX rejected the MIP start during model registration")
            mip_start_status = "accepted"
            log_stream = StringIO()

        _check_deadline(deadline)

        recorder = None
        if collect_progress and np.any(integer):
            recorder = ProgressDataRecorder(clock=ProgressClock.All)
            model.add_progress_listener(recorder)

        solve_started = perf_counter()
        model_build_time = solve_started - build_started
        solution = model.solve(log_output=log_stream if log_stream is not None else False)
        solve_elapsed = perf_counter() - solve_started
        details = model.solve_details
        if log_stream is not None:
            mip_start_status = _mip_start_status_from_log(
                log_stream.getvalue(), default=mip_start_status
            )
        if solution is None:
            raise RuntimeError(f"CPLEX solve failed ({details.status}): {details}")
        status = _status(details.status, has_solution=True, mixed_integer=bool(np.any(integer)))
        if status not in {"optimal", "feasible_with_gap"}:
            raise RuntimeError(f"CPLEX solve failed ({status}): {details.status}")
        values = np.asarray([solution.get_value(variable) for variable in variables], dtype=float)
        backend = "docplex-cplex"
        try:
            backend = f"docplex-cplex-{version('cplex')}"
        except PackageNotFoundError:
            pass
        progress = (
            _progress_records(
                recorder.recorded if recorder is not None else [],
                solve_elapsed=solve_elapsed,
                final_objective=float(solution.objective_value),
                final_bound=_optional_number(details, "best_bound"),
                final_gap=_optional_number(details, "mip_relative_gap"),
                final_nodes=_optional_int(details, "nb_nodes_processed"),
            )
            if collect_progress
            else []
        )
        return CplexResult(
            x=values,
            fun=float(solution.objective_value),
            status=status,
            message=str(details.status),
            mip_dual_bound=_optional_number(details, "best_bound"),
            mip_gap=_optional_number(details, "mip_relative_gap"),
            mip_node_count=_optional_int(details, "nb_nodes_processed"),
            backend=backend,
            progress=progress,
            mip_start_status=mip_start_status,
            model_build_time=model_build_time,
        )
    finally:
        model.end()


def validate_backend(value: str) -> str:
    backend = str(value).lower()
    if backend not in {"scipy", "cplex"}:
        raise ValueError("solver backend must be 'scipy' or 'cplex'")
    return backend


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= float(deadline):
        raise RuntimeError("CPLEX model construction exhausted the wall-clock budget")


def _status(text: str, *, has_solution: bool, mixed_integer: bool) -> str:
    lowered = str(text).lower()
    if "optimal" in lowered:
        return "optimal"
    if "time limit" in lowered:
        return "feasible_with_gap" if has_solution and mixed_integer else "time_limit"
    if "infeasible" in lowered:
        return "infeasible"
    if "unbounded" in lowered:
        return "unbounded"
    return "feasible_with_gap" if has_solution and mixed_integer else "solver_error"


def _optional_number(value: object, name: str) -> float | None:
    number = getattr(value, name, None)
    return None if number is None or not np.isfinite(number) else float(number)


def _optional_int(value: object, name: str) -> int | None:
    number = getattr(value, name, None)
    return None if number is None else int(number)


def _validate_mip_start_vector(
    mip_start: Any,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    integrality: np.ndarray,
    tolerance: float = 1e-7,
) -> np.ndarray:
    if not hasattr(mip_start, "values") or not hasattr(mip_start, "effort_level"):
        raise TypeError("mip_start must provide values and effort_level")
    values = np.asarray(mip_start.values, dtype=float)
    if values.shape != lower_bounds.shape:
        raise ValueError(
            f"MIP-start length {values.size} does not match the model's "
            f"{lower_bounds.size} variables"
        )
    if not np.isfinite(values).all():
        raise ValueError("MIP-start values must all be finite")
    below = np.isfinite(lower_bounds) & (values < lower_bounds - tolerance)
    above = np.isfinite(upper_bounds) & (values > upper_bounds + tolerance)
    if np.any(below | above):
        index = int(np.flatnonzero(below | above)[0])
        raise ValueError(f"MIP-start value at index {index} violates its variable bounds")
    binary_indices = np.flatnonzero(integrality)
    if binary_indices.size:
        binary_values = values[binary_indices]
        rounded = np.rint(binary_values)
        invalid = (np.abs(binary_values - rounded) > tolerance) | ~np.isin(rounded, [0.0, 1.0])
        if np.any(invalid):
            index = int(binary_indices[np.flatnonzero(invalid)[0]])
            raise ValueError(f"MIP-start value at index {index} is not binary")
        values = values.copy()
        values[binary_indices] = rounded
    return values


def _mip_start_effort_level(value: str, enum: Any) -> Any:
    normalized = str(value).strip().lower().replace("-", "_")
    levels = {
        "auto": enum.Auto,
        "check_feas": enum.CheckFeas,
        "solve_fixed": enum.SolveFixed,
        "solve_mip": enum.SolveMIP,
        "repair": enum.Repair,
        "no_check": enum.NoCheck,
    }
    try:
        return levels[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported MIP-start effort level: {value!r}") from exc


def _mip_start_status_from_log(text: str, *, default: str | None) -> str | None:
    lowered = text.lower()
    if "no solution found from" in lowered and "mip start" in lowered:
        return "rejected"
    if "repair" in lowered and "mip start" in lowered:
        return "repaired"
    if "mip start" in lowered and (
        "provided solutions" in lowered or "defined initial solution" in lowered
    ):
        return "accepted"
    return default


def _progress_records(
    raw_records: Iterable[Any],
    *,
    solve_elapsed: float,
    final_objective: float,
    final_bound: float | None,
    final_gap: float | None,
    final_nodes: int | None,
) -> list[Any]:
    # Imported lazily to keep the generic model backend independent during package import.
    from src.search.progress import SolverProgressRecord

    records: list[SolverProgressRecord] = []
    incumbent_best: float | None = None
    solution_count = 0
    previous_time = 0.0
    for raw in raw_records:
        elapsed = max(previous_time, float(getattr(raw, "time", previous_time)))
        previous_time = elapsed
        candidate = getattr(raw, "current_objective", None)
        candidate = None if candidate is None or not np.isfinite(candidate) else float(candidate)
        if candidate is not None:
            if incumbent_best is None or candidate < incumbent_best - 1e-9:
                incumbent_best = candidate
                solution_count += 1
            else:
                incumbent_best = min(incumbent_best, candidate)
        records.append(
            SolverProgressRecord(
                elapsed_seconds=elapsed,
                incumbent_objective=incumbent_best,
                best_bound=_finite_or_none(getattr(raw, "best_bound", None)),
                relative_gap=_finite_or_none(getattr(raw, "mip_gap", None)),
                node_count=_int_or_none(getattr(raw, "current_nb_nodes", None)),
                solution_count=solution_count if incumbent_best is not None else 0,
            )
        )

    if incumbent_best is None or final_objective < incumbent_best - 1e-9:
        incumbent_best = final_objective
        solution_count += 1
    else:
        incumbent_best = min(incumbent_best, final_objective)
    records.append(
        SolverProgressRecord(
            elapsed_seconds=max(previous_time, float(solve_elapsed)),
            incumbent_objective=incumbent_best,
            best_bound=final_bound,
            relative_gap=final_gap,
            node_count=final_nodes,
            solution_count=max(1, solution_count),
        )
    )
    return records


def _finite_or_none(value: object) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _int_or_none(value: object) -> int | None:
    return None if value is None else int(value)
