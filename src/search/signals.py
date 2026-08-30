"""Deterministic train-only feature signals and Pin-FS LP relaxation."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.feature_selection import mutual_info_classif

from src.models.corrected.base import scipy_status, validate_training_data
from src.models.corrected.cplex_backend import solve_docplex, validate_backend
from src.models.corrected.l1_svm import L1SVM
from src.models.corrected.pin_svm import PinSVM

from .restricted_solver import build_pin_fs_problem
from .states import FeatureState, RestrictedSolveResult


class TimeBudgetExceeded(RuntimeError):
    """Raised when signal work cannot continue inside the route's wall-clock budget."""


@dataclass(frozen=True)
class LPRelaxationResult:
    v_lp: np.ndarray
    w_lp: np.ndarray
    objective: float
    runtime: float
    status: str
    from_cache: bool = False


@dataclass
class NormalizationParameters:
    """Persistable min/max parameters used for deterministic signal scaling."""

    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit_transform(self, name: str, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"signal {name!r} contains NaN or infinite values")
        lower = float(values.min())
        upper = float(values.max())
        self.bounds[name] = (lower, upper)
        if upper - lower <= 1e-15:
            return np.zeros_like(values, dtype=float)
        return (values - lower) / (upper - lower)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {"minimum": lower, "maximum": upper}
            for name, (lower, upper) in sorted(self.bounds.items())
        }


@dataclass
class StaticSignalData:
    values: dict[str, np.ndarray]
    standardized_X: np.ndarray
    normalization: NormalizationParameters
    lp_relaxation: LPRelaxationResult | None
    overhead_seconds: dict[str, float]


@dataclass
class LPRelaxationCache:
    entries: dict[str, LPRelaxationResult] = field(default_factory=dict)


def solve_pin_fs_relaxation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    B: int,
    C: float,
    tau: float,
    coefficient_bounds: tuple[float, float],
    backend: str = "scipy",
    time_limit: float | None = None,
    threads: int = 1,
    allowed_features: set[int] | None = None,
    cache: LPRelaxationCache | None = None,
    deadline: float | None = None,
) -> LPRelaxationResult:
    """Solve the Pin-FS relaxation with selectors continuous in ``[0, 1]``."""
    X, y = validate_training_data(X, y)
    backend = validate_backend(backend)
    key = _relaxation_key(
        X,
        y,
        B=B,
        C=C,
        tau=tau,
        coefficient_bounds=coefficient_bounds,
        backend=backend,
        allowed_features=allowed_features,
    )
    if cache is not None and key in cache.entries:
        cached = cache.entries[key]
        return LPRelaxationResult(
            v_lp=cached.v_lp.copy(),
            w_lp=cached.w_lp.copy(),
            objective=cached.objective,
            runtime=0.0,
            status=cached.status,
            from_cache=True,
        )

    problem = build_pin_fs_problem(
        X,
        y,
        B=B,
        C=C,
        tau=tau,
        lower_bound=coefficient_bounds[0],
        upper_bound=coefficient_bounds[1],
        allowed_features=allowed_features,
        deadline=deadline,
    )
    effective_time_limit = _deadline_time_limit(time_limit, deadline)
    started = perf_counter()
    if backend == "cplex":
        raw = solve_docplex(
            problem.c,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
            constraint_matrix=problem.constraint_matrix,
            constraint_lower=problem.constraint_lower,
            constraint_upper=problem.constraint_upper,
            integrality=np.zeros_like(problem.integrality),
            time_limit=effective_time_limit,
            threads=threads,
            model_name="pin-fs-lp-relaxation",
            deadline=deadline,
        )
        status = raw.status
    else:
        options = (
            {"time_limit": float(effective_time_limit)}
            if effective_time_limit is not None
            else None
        )
        raw = milp(
            problem.c,
            integrality=np.zeros_like(problem.integrality),
            bounds=Bounds(problem.lower_bounds, problem.upper_bounds),
            constraints=LinearConstraint(
                problem.constraint_matrix,
                problem.constraint_lower,
                problem.constraint_upper,
            ),
            options=options,
        )
        status = scipy_status(raw, mixed_integer=False)
        if raw.x is None or status not in {"optimal", "feasible_with_gap"}:
            raise RuntimeError(f"Pin-FS LP relaxation failed ({status}): {raw.message}")
    result = LPRelaxationResult(
        v_lp=np.asarray(raw.x[problem.v_slice], dtype=float).copy(),
        w_lp=np.asarray(raw.x[problem.w_slice], dtype=float).copy(),
        objective=float(raw.fun),
        runtime=perf_counter() - started,
        status=status,
    )
    if cache is not None:
        cache.entries[key] = result
    return result


def compute_static_signals(
    X: np.ndarray,
    y: np.ndarray,
    *,
    B: int,
    C: float,
    tau: float,
    coefficient_bounds: tuple[float, float],
    seed: int,
    use_l1: bool = False,
    use_pin: bool = False,
    use_lp: bool = True,
    baseline_backend: str = "scipy",
    lp_backend: str = "scipy",
    lp_time_limit: float | None = None,
    threads: int = 1,
    correlation_chunk_size: int = 256,
    deadline: float | None = None,
    lp_cache: LPRelaxationCache | None = None,
) -> StaticSignalData:
    """Compute normalized signals from training data only."""
    X, y = validate_training_data(X, y)
    started = perf_counter()
    overhead: dict[str, float] = {}
    normalizer = NormalizationParameters()
    baseline_backend = validate_backend(baseline_backend)

    stage = perf_counter()
    standardized = _standardize(X)
    fisher = _fisher_scores(X, y)
    mi = mutual_info_classif(X, y, random_state=int(seed))
    overhead["univariate"] = perf_counter() - stage
    _check_deadline(deadline)

    stage = perf_counter()
    mean_corr, max_corr = _correlation_summaries(
        standardized,
        chunk_size=correlation_chunk_size,
        deadline=deadline,
    )
    overhead["correlation"] = perf_counter() - stage

    l1_coefficients = np.zeros(X.shape[1], dtype=float)
    if use_l1:
        _check_deadline(deadline)
        stage = perf_counter()
        l1 = L1SVM(
            C=C,
            time_limit=_remaining(deadline),
            backend=baseline_backend,
            threads=threads,
        ).fit(X, y)
        l1_coefficients = np.abs(np.asarray(l1.w_, dtype=float))
        overhead["l1"] = perf_counter() - stage
        _check_deadline(deadline)

    pin_coefficients = np.zeros(X.shape[1], dtype=float)
    if use_pin:
        _check_deadline(deadline)
        if deadline is not None and baseline_backend != "cplex":
            raise ValueError(
                "use_pin under a strict wall-clock budget requires baseline_backend='cplex'"
            )
        stage = perf_counter()
        pin = PinSVM(
            C=C,
            tau=tau,
            time_limit=_remaining(deadline),
            backend=baseline_backend,
            threads=threads,
        ).fit(X, y)
        pin_coefficients = np.abs(np.asarray(pin.w_, dtype=float))
        overhead["pin"] = perf_counter() - stage
        _check_deadline(deadline)

    lp_result = None
    lp_activation = np.zeros(X.shape[1], dtype=float)
    lp_coefficients = np.zeros(X.shape[1], dtype=float)
    if use_lp:
        _check_deadline(deadline)
        stage = perf_counter()
        remaining = _remaining(deadline)
        effective_limit = _minimum_positive(lp_time_limit, remaining)
        lp_result = solve_pin_fs_relaxation(
            X,
            y,
            B=B,
            C=C,
            tau=tau,
            coefficient_bounds=coefficient_bounds,
            backend=lp_backend,
            time_limit=effective_limit,
            threads=threads,
            cache=lp_cache,
            deadline=deadline,
        )
        lp_activation = lp_result.v_lp
        lp_coefficients = np.abs(lp_result.w_lp)
        overhead["lp_relaxation"] = perf_counter() - stage
        _check_deadline(deadline)

    raw_values = {
        "fisher_score": fisher,
        "mutual_information": mi,
        "mean_abs_correlation": mean_corr,
        "max_abs_correlation": max_corr,
        "l1_abs_coefficient": l1_coefficients,
        "pin_abs_coefficient": pin_coefficients,
        "lp_activation": lp_activation,
        "lp_abs_coefficient": lp_coefficients,
    }
    values = {
        name: normalizer.fit_transform(name, np.asarray(signal, dtype=float))
        for name, signal in raw_values.items()
    }
    overhead["total"] = perf_counter() - started
    return StaticSignalData(
        values=values,
        standardized_X=standardized,
        normalization=normalizer,
        lp_relaxation=lp_result,
        overhead_seconds=overhead,
    )


def build_feature_states(
    static: StaticSignalData,
    *,
    kernel: set[int],
    current_result: RestrictedSolveResult | None,
    selection_counts: np.ndarray,
    observations: int,
    inactive_iterations: np.ndarray,
    kernel_age: np.ndarray,
) -> tuple[list[FeatureState], dict[str, dict[str, float]]]:
    """Combine static, incumbent, residual, and search-history signals."""
    n = static.standardized_X.shape[1]
    coefficients = (
        np.zeros(n, dtype=float)
        if current_result is None
        else np.abs(np.asarray(current_result.coefficients, dtype=float))
    )
    selected = coefficients > 1e-3
    slack = (
        np.zeros(static.standardized_X.shape[0], dtype=float)
        if current_result is None
        else np.asarray(current_result.xi, dtype=float)
    )
    slack_association = _absolute_association(static.standardized_X, slack)
    support = np.flatnonzero(selected)
    redundancy = _support_redundancy(static.standardized_X, support)
    dynamic = NormalizationParameters()
    coefficient_signal = dynamic.fit_transform("abs_coefficient", coefficients)
    slack_signal = dynamic.fit_transform("slack_association", slack_association)
    frequencies = np.asarray(selection_counts, dtype=float) / max(1, int(observations))

    states = [
        FeatureState(
            index=j,
            in_kernel=j in kernel,
            is_selected=bool(selected[j]),
            abs_coefficient=float(coefficient_signal[j]),
            fisher_score=float(static.values["fisher_score"][j]),
            mutual_information=float(static.values["mutual_information"][j]),
            mean_abs_correlation=float(static.values["mean_abs_correlation"][j]),
            max_abs_correlation=float(static.values["max_abs_correlation"][j]),
            lp_activation=float(static.values["lp_activation"][j]),
            lp_abs_coefficient=float(static.values["lp_abs_coefficient"][j]),
            slack_association=float(slack_signal[j]),
            selection_frequency=float(frequencies[j]),
            inactive_iterations=int(inactive_iterations[j]),
            kernel_age=int(kernel_age[j]),
            l1_abs_coefficient=float(static.values["l1_abs_coefficient"][j]),
            pin_abs_coefficient=float(static.values["pin_abs_coefficient"][j]),
            support_redundancy=float(redundancy[j]),
        )
        for j in range(n)
    ]
    return states, dynamic.to_dict()


def _fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    positive = X[y == 1]
    negative = X[y == -1]
    numerator = np.square(positive.mean(axis=0) - negative.mean(axis=0))
    denominator = positive.var(axis=0) + negative.var(axis=0) + 1e-12
    return numerator / denominator


def _standardize(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale <= 1e-15] = 1.0
    return centered / scale


def _correlation_summaries(
    standardized_X: np.ndarray,
    *,
    chunk_size: int,
    deadline: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if int(chunk_size) < 1:
        raise ValueError("correlation_chunk_size must be positive")
    samples, features = standardized_X.shape
    denominator = max(1, samples)
    mean_values = np.zeros(features, dtype=float)
    max_values = np.zeros(features, dtype=float)
    for start in range(0, features, int(chunk_size)):
        _check_deadline(deadline)
        stop = min(features, start + int(chunk_size))
        correlation = np.abs(standardized_X[:, start:stop].T @ standardized_X / denominator)
        for local, index in enumerate(range(start, stop)):
            correlation[local, index] = 0.0
        divisor = max(1, features - 1)
        mean_values[start:stop] = correlation.sum(axis=1) / divisor
        max_values[start:stop] = correlation.max(axis=1, initial=0.0)
    return np.clip(mean_values, 0.0, 1.0), np.clip(max_values, 0.0, 1.0)


def _absolute_association(standardized_X: np.ndarray, values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    norm = float(np.linalg.norm(centered))
    if norm <= 1e-15:
        return np.zeros(standardized_X.shape[1], dtype=float)
    feature_norms = np.linalg.norm(standardized_X, axis=0)
    denominator = np.maximum(feature_norms * norm, 1e-15)
    return np.clip(np.abs(standardized_X.T @ centered) / denominator, 0.0, 1.0)


def _support_redundancy(standardized_X: np.ndarray, support: np.ndarray) -> np.ndarray:
    if support.size == 0:
        return np.zeros(standardized_X.shape[1], dtype=float)
    denominator = max(1, standardized_X.shape[0])
    values = np.abs(standardized_X.T @ standardized_X[:, support] / denominator)
    result = values.max(axis=1)
    result[support] = 0.0
    return np.clip(result, 0.0, 1.0)


def _relaxation_key(
    X: np.ndarray,
    y: np.ndarray,
    **parameters: Any,
) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(X).view(np.uint8))
    digest.update(np.ascontiguousarray(y).view(np.uint8))
    stable_parameters = dict(parameters)
    allowed = stable_parameters.get("allowed_features")
    if allowed is not None:
        stable_parameters["allowed_features"] = sorted(allowed)
    digest.update(
        json.dumps(stable_parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise TimeBudgetExceeded("signal computation exhausted the total route time budget")


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - perf_counter())


def _minimum_positive(first: float | None, second: float | None) -> float | None:
    values = [float(value) for value in (first, second) if value is not None]
    if any(value <= 0 for value in values):
        raise ValueError("solver time limits must be positive when provided")
    return min(values) if values else None


def _deadline_time_limit(
    time_limit: float | None,
    deadline: float | None,
) -> float | None:
    if deadline is None:
        return time_limit
    remaining = float(deadline) - perf_counter()
    if remaining <= 1e-6:
        raise TimeBudgetExceeded("LP model construction exhausted the wall-clock budget")
    return remaining if time_limit is None else min(float(time_limit), remaining)
