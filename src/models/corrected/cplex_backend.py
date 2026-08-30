"""Lazy DOcplex adapter for manuscript-level CPLEX solver parity."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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
) -> CplexResult:
    """Solve a linear or convex quadratic model through local DOcplex/CPLEX."""
    try:
        from docplex.mp.model import Model
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
            if integer[index]:
                variables.append(model.binary_var(name=f"x_{index}"))
            else:
                lb = -model.infinity if np.isneginf(lower[index]) else float(lower[index])
                ub = model.infinity if np.isposinf(upper[index]) else float(upper[index])
                variables.append(model.continuous_var(lb=lb, ub=ub, name=f"x_{index}"))

        for row in range(matrix.shape[0]):
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
        if time_limit is not None:
            model.parameters.timelimit = float(time_limit)
        if np.any(integer) and mip_gap is not None:
            model.parameters.mip.tolerances.mipgap = float(mip_gap)

        solution = model.solve(log_output=False)
        details = model.solve_details
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
        return CplexResult(
            x=values,
            fun=float(solution.objective_value),
            status=status,
            message=str(details.status),
            mip_dual_bound=_optional_number(details, "best_bound"),
            mip_gap=_optional_number(details, "mip_relative_gap"),
            mip_node_count=_optional_int(details, "nb_nodes_processed"),
            backend=backend,
        )
    finally:
        model.end()


def validate_backend(value: str) -> str:
    backend = str(value).lower()
    if backend not in {"scipy", "cplex"}:
        raise ValueError("solver backend must be 'scipy' or 'cplex'")
    return backend


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
