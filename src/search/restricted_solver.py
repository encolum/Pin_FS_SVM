"""Reusable full- and restricted-kernel solvers for paper-aligned Pin-FS-SVM."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from src.models.corrected.base import (
    SolverDiagnostics,
    scipy_status,
    validate_coefficient_bounds,
    validate_positive,
    validate_training_data,
)
from src.models.corrected.cplex_backend import solve_docplex, validate_backend

from .mip_start import MIPStartData
from .progress import SolverProgressRecord, validate_progress_trajectory
from .states import PinFSProblemData, RestrictedSolveResult


def build_pin_fs_problem(
    X: np.ndarray,
    y: np.ndarray,
    *,
    B: int,
    C: float,
    tau: float,
    lower_bound: float,
    upper_bound: float,
    allowed_features: Iterable[int] | None = None,
) -> PinFSProblemData:
    """Build formulation (29)-(38), optionally fixing ``v_j=0`` outside a kernel."""
    X, y = validate_training_data(X, y)
    m, n = X.shape
    B = _validate_budget(B, n)
    C = validate_positive(C, "C")
    tau = validate_positive(tau, "tau")
    if tau > 1:
        raise ValueError("tau must satisfy 0 < tau <= 1")
    lower_bound, upper_bound = validate_coefficient_bounds(lower_bound, upper_bound)
    allowed = _validate_feature_set(allowed_features, n)

    w_slice = slice(0, n)
    b_index = n
    z_slice = slice(n + 1, 2 * n + 1)
    xi_slice = slice(2 * n + 1, 2 * n + m + 1)
    v_slice = slice(2 * n + m + 1, 3 * n + m + 1)
    total = v_slice.stop

    row_count = 2 * m + 4 * n + 1
    matrix = lil_matrix((row_count, total), dtype=float)
    row_lower = np.full(row_count, -np.inf)
    row_upper = np.full(row_count, np.inf)
    row = 0
    for i in range(m):
        matrix[row, w_slice] = y[i] * X[i]
        matrix[row, b_index] = y[i]
        matrix[row, xi_slice.start + i] = 1.0
        row_lower[row] = 1.0
        row += 1

        matrix[row, w_slice] = y[i] * X[i]
        matrix[row, b_index] = y[i]
        matrix[row, xi_slice.start + i] = -1.0 / tau
        row_upper[row] = 1.0
        row += 1

    for j in range(n):
        matrix[row, w_slice.start + j] = 1.0
        matrix[row, z_slice.start + j] = -1.0
        row_upper[row] = 0.0
        row += 1

        matrix[row, w_slice.start + j] = -1.0
        matrix[row, z_slice.start + j] = -1.0
        row_upper[row] = 0.0
        row += 1

        matrix[row, w_slice.start + j] = 1.0
        matrix[row, v_slice.start + j] = -upper_bound
        row_upper[row] = 0.0
        row += 1

        matrix[row, w_slice.start + j] = -1.0
        matrix[row, v_slice.start + j] = lower_bound
        row_upper[row] = 0.0
        row += 1

    matrix[row, v_slice] = 1.0
    row_upper[row] = B

    objective = np.zeros(total, dtype=float)
    objective[z_slice] = 1.0
    objective[xi_slice] = C
    variable_lower = np.concatenate(
        [np.full(n + 1, -np.inf), np.zeros(n + m + n)]
    )
    variable_upper = np.concatenate(
        [np.full(n + 1 + n + m, np.inf), np.ones(n)]
    )
    for j in range(n):
        if j not in allowed:
            variable_upper[v_slice.start + j] = 0.0
    integrality = np.zeros(total, dtype=int)
    integrality[v_slice] = 1

    return PinFSProblemData(
        c=objective,
        lower_bounds=variable_lower,
        upper_bounds=variable_upper,
        constraint_matrix=matrix.tocsr(),
        constraint_lower=row_lower,
        constraint_upper=row_upper,
        integrality=integrality,
        w_slice=w_slice,
        b_index=b_index,
        z_slice=z_slice,
        xi_slice=xi_slice,
        v_slice=v_slice,
        feature_budget=B,
        allowed_features=frozenset(allowed),
    )


def solve_restricted_pin_fs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: set[int],
    B: int,
    C: float,
    tau: float,
    coefficient_bounds: tuple[float, float],
    backend: str,
    time_limit: float | None,
    mip_gap: float | None,
    threads: int,
    mip_start: MIPStartData | None = None,
    collect_progress: bool = True,
    allow_kernel_smaller_than_budget: bool = False,
) -> RestrictedSolveResult:
    """Solve Pin-FS-SVM with binary selectors outside ``kernel`` fixed to zero."""
    X, y = validate_training_data(X, y)
    n = X.shape[1]
    kernel = _validate_feature_set(kernel, n)
    B = _validate_budget(B, n)
    if len(kernel) < B and not allow_kernel_smaller_than_budget:
        raise ValueError(
            f"kernel has {len(kernel)} features but B={B}; set "
            "allow_kernel_smaller_than_budget=True to allow this explicitly"
        )
    if len(coefficient_bounds) != 2:
        raise ValueError("coefficient_bounds must contain exactly (lower_bound, upper_bound)")
    lower_bound, upper_bound = validate_coefficient_bounds(*coefficient_bounds)
    backend = validate_backend(backend)
    threads = _validate_threads(threads)
    if time_limit is not None and float(time_limit) <= 0:
        raise ValueError("time_limit must be positive when provided")
    if mip_gap is not None and float(mip_gap) < 0:
        raise ValueError("mip_gap must be non-negative when provided")
    if mip_start is not None and backend != "cplex":
        raise ValueError("MIP starts are supported only when backend='cplex'")

    problem = build_pin_fs_problem(
        X,
        y,
        B=B,
        C=C,
        tau=tau,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        allowed_features=kernel,
    )
    started = perf_counter()
    if backend == "cplex":
        raw_result = solve_docplex(
            problem.c,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
            constraint_matrix=problem.constraint_matrix,
            constraint_lower=problem.constraint_lower,
            constraint_upper=problem.constraint_upper,
            integrality=problem.integrality,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            model_name="restricted-pin-fs-svm",
            mip_start=mip_start,
            collect_progress=collect_progress,
        )
        status = raw_result.status
        progress = list(raw_result.progress)
        mip_start_status = raw_result.mip_start_status
    else:
        options: dict[str, float] = {}
        if time_limit is not None:
            options["time_limit"] = float(time_limit)
        if mip_gap is not None:
            options["mip_rel_gap"] = float(mip_gap)
        raw_result = milp(
            problem.c,
            integrality=problem.integrality,
            bounds=Bounds(problem.lower_bounds, problem.upper_bounds),
            constraints=LinearConstraint(
                problem.constraint_matrix,
                problem.constraint_lower,
                problem.constraint_upper,
            ),
            options=options or None,
        )
        status = scipy_status(raw_result, mixed_integer=True)
        if raw_result.x is None or status not in {"optimal", "feasible_with_gap"}:
            raise RuntimeError(f"restricted Pin-FS solve failed ({status}): {raw_result.message}")
        progress = []
        mip_start_status = None

    solve_time = perf_counter() - started
    if collect_progress and not progress:
        progress = [
            SolverProgressRecord(
                elapsed_seconds=solve_time,
                incumbent_objective=float(raw_result.fun),
                best_bound=_optional_float(raw_result, "mip_dual_bound"),
                relative_gap=_optional_float(raw_result, "mip_gap"),
                node_count=_optional_int(raw_result, "mip_node_count"),
                solution_count=1,
            )
        ]
    validate_progress_trajectory(progress)

    values = np.asarray(raw_result.x, dtype=float)
    coefficients = values[problem.w_slice].copy()
    z = values[problem.z_slice].copy()
    xi = values[problem.xi_slice].copy()
    v = np.rint(values[problem.v_slice]).astype(int)
    outside = sorted(set(range(n)) - kernel)
    if outside and (np.any(v[outside] != 0) or np.any(np.abs(coefficients[outside]) > 1e-7)):
        raise RuntimeError("solver returned active variables outside the allowed feature kernel")

    diagnostics = SolverDiagnostics(
        status=status,
        objective_value=float(raw_result.fun),
        best_bound=_optional_float(raw_result, "mip_dual_bound"),
        mip_gap=_optional_float(raw_result, "mip_gap"),
        node_count=_optional_int(raw_result, "mip_node_count"),
        message=str(raw_result.message),
        backend=getattr(raw_result, "backend", "scipy-highs"),
    )
    return RestrictedSolveResult(
        objective=float(raw_result.fun),
        support=set(np.flatnonzero(np.abs(coefficients) > 1e-3).astype(int).tolist()),
        coefficients=coefficients,
        intercept=float(values[problem.b_index]),
        z=z,
        xi=xi,
        v=v,
        diagnostics=diagnostics,
        progress=progress,
        solve_time=solve_time,
        kernel=set(kernel),
        mip_start_status=mip_start_status,
    )


def _validate_budget(B: int, n_features: int) -> int:
    if isinstance(B, bool) or int(B) != B or int(B) < 1:
        raise ValueError("B must be a positive integer")
    B = int(B)
    if B > n_features:
        raise ValueError(f"B={B} exceeds the number of features ({n_features})")
    return B


def _validate_feature_set(features: Iterable[int] | None, n_features: int) -> set[int]:
    if features is None:
        return set(range(n_features))
    try:
        raw_features = list(features)
    except TypeError as exc:
        raise TypeError("allowed features must be an iterable of integer indices") from exc
    normalized: set[int] = set()
    for feature in raw_features:
        if isinstance(feature, (bool, np.bool_)) or not isinstance(feature, (int, np.integer)):
            raise ValueError("feature indices must be integers")
        index = int(feature)
        if index < 0 or index >= n_features:
            raise ValueError(f"feature index {index} is outside [0, {n_features})")
        normalized.add(index)
    return normalized


def _validate_threads(threads: int) -> int:
    if isinstance(threads, bool) or int(threads) != threads or int(threads) < 1:
        raise ValueError("threads must be a positive integer")
    return int(threads)


def _optional_float(result: object, name: str) -> float | None:
    value = getattr(result, name, None)
    return None if value is None or not np.isfinite(value) else float(value)


def _optional_int(result: object, name: str) -> int | None:
    value = getattr(result, name, None)
    return None if value is None else int(value)
