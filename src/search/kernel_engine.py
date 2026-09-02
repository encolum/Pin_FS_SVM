"""Policy-agnostic, wall-clock-budgeted kernel-search engine."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np
from src.models.base import validate_training_data

from .mip_start import result_to_mip_start
from .policies.base import KernelPolicy
from .progress import SolverProgressRecord, validate_progress_trajectory
from .restricted_solver import build_pin_fs_problem, solve_restricted_pin_fs
from .signals import (
    LPRelaxationCache,
    TimeBudgetExceeded,
    build_feature_states,
    compute_static_signals,
)
from .states import KernelSearchResult, RestrictedSolveResult, SearchState


def run_kernel_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    policy: KernelPolicy,
    B: int,
    C: float,
    tau: float,
    coefficient_bounds: tuple[float, float],
    total_time_limit: float,
    subproblem_time_limit: float,
    max_iterations: int,
    backend: str,
    threads: int,
    final_full_refinement: bool,
    final_refinement_fraction: float,
    seed: int,
    mip_gap: float | None = None,
    acceptance_epsilon: float = 1e-9,
    signal_options: dict[str, Any] | None = None,
    lp_cache: LPRelaxationCache | None = None,
) -> KernelSearchResult:
    """Run Static KS, ADKS, or VeraPin through one shared search loop."""
    route_started = perf_counter()
    X, y = validate_training_data(X, y)
    total_time_limit = _positive(total_time_limit, "total_time_limit")
    subproblem_time_limit = _positive(subproblem_time_limit, "subproblem_time_limit")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    if not 0 <= float(final_refinement_fraction) < 1:
        raise ValueError("final_refinement_fraction must lie in [0, 1)")
    if float(acceptance_epsilon) < 0:
        raise ValueError("acceptance_epsilon must be non-negative")

    deadline = route_started + total_time_limit
    reserve = total_time_limit * float(final_refinement_fraction) if final_full_refinement else 0.0
    search_deadline = deadline - reserve
    options = dict(signal_options or {})
    options.setdefault("use_l1", False)
    options.setdefault("use_pin", False)
    options.setdefault("use_lp", True)
    options.setdefault("lp_backend", backend)
    options.setdefault("threads", threads)
    options.setdefault("lp_cache", lp_cache)
    options["deadline"] = search_deadline

    static = compute_static_signals(
        X,
        y,
        B=B,
        C=C,
        tau=tau,
        coefficient_bounds=coefficient_bounds,
        seed=int(seed),
        **options,
    )
    n = X.shape[1]
    selection_counts = np.zeros(n, dtype=int)
    inactive_iterations = np.zeros(n, dtype=int)
    kernel_age = np.zeros(n, dtype=int)
    empty_states, initial_dynamic_normalization = build_feature_states(
        static,
        kernel=set(),
        current_result=None,
        selection_counts=selection_counts,
        observations=0,
        inactive_iterations=inactive_iterations,
        kernel_age=kernel_age,
    )
    initial_search = SearchState(
        iteration=0,
        current_objective=0.0,
        best_objective=0.0,
        current_gap=None,
        best_bound=None,
        kernel_size=0,
        feature_budget=int(B),
        total_features=n,
        stagnation_iterations=0,
        elapsed_seconds=perf_counter() - route_started,
        remaining_seconds=max(0.0, deadline - perf_counter()),
        C=float(C),
        tau=float(tau),
    )
    policy_started = perf_counter()
    kernel = _validate_or_complete_kernel(
        policy.initialize_kernel(empty_states, initial_search),
        features=empty_states,
        search=initial_search,
        policy=policy,
    )
    policy_overhead = perf_counter() - policy_started
    initial_kernel = set(kernel)
    initial_solve_limit = _available_solve_time(search_deadline, subproblem_time_limit)
    first_solve_started = perf_counter()
    current = solve_restricted_pin_fs(
        X,
        y,
        kernel=kernel,
        B=B,
        C=C,
        tau=tau,
        coefficient_bounds=coefficient_bounds,
        backend=backend,
        time_limit=initial_solve_limit,
        mip_gap=mip_gap,
        threads=threads,
        collect_progress=True,
        deadline=search_deadline,
    )
    route_progress: list[SolverProgressRecord] = []
    _append_route_progress(
        route_progress,
        current.progress,
        offset=first_solve_started - route_started,
        comparable_full_model_bound=False,
    )
    best = current
    observations = 1
    _update_feature_history(
        current,
        kernel,
        selection_counts=selection_counts,
        inactive_iterations=inactive_iterations,
        kernel_age=kernel_age,
    )
    history: list[dict[str, Any]] = [
        _history_record(
            iteration=0,
            kernel=kernel,
            result=current,
            best=best,
            improved=True,
            added=kernel,
            removed=set(),
            stagnation=0,
        )
    ]
    normalization_history = [initial_dynamic_normalization]
    stagnation = 0
    improving_iterations = 1
    mip_start_overhead = 0.0
    failures: list[dict[str, Any]] = []

    for iteration in range(1, int(max_iterations)):
        if perf_counter() >= search_deadline:
            break
        try:
            states, dynamic_normalization = build_feature_states(
                static, kernel=kernel, current_result=current,
                selection_counts=selection_counts, observations=observations,
                inactive_iterations=inactive_iterations, kernel_age=kernel_age)
        except TimeBudgetExceeded:
            break  # retain the feasible incumbent and reserve final refinement time
        normalization_history.append(dynamic_normalization)
        search_state = _search_state(
            iteration=iteration - 1,
            current=current,
            best=best,
            kernel=kernel,
            B=B,
            C=C,
            tau=tau,
            stagnation=stagnation,
            route_started=route_started,
            deadline=deadline,
            improved=history[-1]["improved"],
            total_features=n,
        )
        policy_started = perf_counter()
        raw_target = policy.target_kernel_size(search_state)
        if isinstance(raw_target, (bool, np.bool_)) or int(raw_target) != raw_target:
            raise ValueError("kernel policy target_kernel_size must return an integer")
        target = int(raw_target)
        next_kernel = _update_kernel(
            states,
            current_kernel=kernel,
            incumbent=best,
            target_size=target,
            search=search_state,
            policy=policy,
        )
        policy_overhead += perf_counter() - policy_started
        if perf_counter() >= search_deadline:
            break

        added = next_kernel - kernel
        removed = kernel - next_kernel
        mip_start = None
        if backend == "cplex":
            start_started = perf_counter()
            target_problem = build_pin_fs_problem(
                X,
                y,
                B=B,
                C=C,
                tau=tau,
                lower_bound=coefficient_bounds[0],
                upper_bound=coefficient_bounds[1],
                allowed_features=next_kernel,
                deadline=search_deadline,
            )
            mip_start = result_to_mip_start(
                best,
                target_problem,
                name=f"kernel_incumbent_{iteration}",
            )
            mip_start_overhead += perf_counter() - start_started
        if perf_counter() >= search_deadline:
            break

        solve_limit = _available_solve_time(search_deadline, subproblem_time_limit)
        solve_started = perf_counter()
        try:
            candidate = solve_restricted_pin_fs(
                X,
                y,
                kernel=next_kernel,
                B=B,
                C=C,
                tau=tau,
                coefficient_bounds=coefficient_bounds,
                backend=backend,
                time_limit=solve_limit,
                mip_gap=mip_gap,
                threads=threads,
                mip_start=mip_start,
                collect_progress=True,
                deadline=search_deadline,
            )
        except RuntimeError as exc:
            failures.append(
                {"iteration": iteration, "exception_type": type(exc).__name__, "message": str(exc)}
            )
            break
        _append_route_progress(
            route_progress,
            candidate.progress,
            offset=solve_started - route_started,
            comparable_full_model_bound=False,
        )
        improved = candidate.objective < best.objective - float(acceptance_epsilon)
        if improved:
            best = candidate
            stagnation = 0
            improving_iterations += 1
        else:
            stagnation += 1
        current = candidate
        kernel = next_kernel
        observations += 1
        _update_feature_history(
            current,
            kernel,
            selection_counts=selection_counts,
            inactive_iterations=inactive_iterations,
            kernel_age=kernel_age,
        )
        history.append(
            _history_record(
                iteration=iteration,
                kernel=kernel,
                result=current,
                best=best,
                improved=improved,
                added=added,
                removed=removed,
                stagnation=stagnation,
            )
        )

    final_refinement_performed = False
    if final_full_refinement and perf_counter() < deadline:
        full_kernel = set(range(n))
        previous_kernel = set(kernel)
        mip_start = None
        if backend == "cplex":
            start_started = perf_counter()
            full_problem = build_pin_fs_problem(
                X,
                y,
                B=B,
                C=C,
                tau=tau,
                lower_bound=coefficient_bounds[0],
                upper_bound=coefficient_bounds[1],
                deadline=deadline,
            )
            mip_start = result_to_mip_start(best, full_problem, name="final_full_refinement")
            mip_start_overhead += perf_counter() - start_started
        if perf_counter() < deadline:
            refinement_limit = max(1e-6, deadline - perf_counter())
            refinement_started = perf_counter()
            try:
                refined = solve_restricted_pin_fs(
                    X,
                    y,
                    kernel=full_kernel,
                    B=B,
                    C=C,
                    tau=tau,
                    coefficient_bounds=coefficient_bounds,
                    backend=backend,
                    time_limit=refinement_limit,
                    mip_gap=mip_gap,
                    threads=threads,
                    mip_start=mip_start,
                    collect_progress=True,
                    deadline=deadline,
                )
            except RuntimeError as exc:
                failures.append(
                    {
                        "iteration": "final_refinement",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            else:
                _append_route_progress(
                    route_progress,
                    refined.progress,
                    offset=refinement_started - route_started,
                    comparable_full_model_bound=True,
                )
                improved = refined.objective < best.objective - float(acceptance_epsilon)
                if improved:
                    best = refined
                    improving_iterations += 1
                elif refined.objective <= best.objective:
                    best = refined
                current = refined
                kernel = full_kernel
                history.append(
                    _history_record(
                        iteration="final_refinement",
                        kernel=kernel,
                        result=refined,
                        best=best,
                        improved=improved,
                        added=full_kernel - previous_kernel,
                        removed=set(),
                        stagnation=0 if improved else stagnation + 1,
                    )
                )
                final_refinement_performed = True

    total_runtime = perf_counter() - route_started
    validate_progress_trajectory(route_progress)
    metadata = {
        "restricted_solves": observations,
        "improving_iterations": improving_iterations,
        "signal_overhead": max(
            0.0,
            static.overhead_seconds.get("total", 0.0)
            - static.overhead_seconds.get("lp_relaxation", 0.0),
        ),
        "policy_overhead": policy_overhead,
        "lp_relaxation_overhead": static.overhead_seconds.get("lp_relaxation", 0.0),
        "mip_start_overhead": mip_start_overhead,
        "total_node_count": sum(int(record.get("node_count") or 0) for record in history),
        "signal_normalization": static.normalization.to_dict(),
        "skipped_signals": static.skipped_signals,
        "dynamic_normalization": normalization_history,
        "route_progress": [asdict(record) for record in route_progress],
        "final_full_refinement": final_refinement_performed,
        "time_budget": total_time_limit,
        "time_budget_exceeded": total_runtime > total_time_limit + 0.05,
        "failures": failures,
        "seed": int(seed),
    }
    return KernelSearchResult(
        best_result=best,
        history=history,
        final_kernel=set(kernel),
        total_runtime=total_runtime,
        initial_kernel=initial_kernel,
        method=str(policy.name),
        metadata=metadata,
    )


def _validate_or_complete_kernel(
    kernel: set[int],
    *,
    features: list,
    search: SearchState,
    policy: KernelPolicy,
) -> set[int]:
    try:
        raw_indices = list(kernel)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy initialize_kernel must return integer feature indices") from exc
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
        for index in raw_indices
    ):
        raise ValueError("policy initialize_kernel must return integer feature indices")
    normalized = {int(index) for index in raw_indices}
    if any(index < 0 or index >= search.total_features for index in normalized):
        raise ValueError("policy initialize_kernel returned an out-of-range feature")
    if len(normalized) < search.feature_budget:
        ranked = sorted(
            (feature for feature in features if feature.index not in normalized),
            key=lambda feature: (-_finite_score(policy.add_score(feature, search)), feature.index),
        )
        needed = search.feature_budget - len(normalized)
        normalized.update(feature.index for feature in ranked[:needed])
    return normalized


def _update_kernel(
    features: list,
    *,
    current_kernel: set[int],
    incumbent: RestrictedSolveResult,
    target_size: int,
    search: SearchState,
    policy: KernelPolicy,
) -> set[int]:
    minimum = max(search.feature_budget, len(incumbent.support))
    target = max(minimum, min(search.total_features, int(target_size)))
    retained = set(incumbent.support)
    candidates = []
    for feature in features:
        if feature.index in retained:
            continue
        score = (
            policy.keep_score(feature, search)
            if feature.index in current_kernel
            else policy.add_score(feature, search)
        )
        candidates.append((-_finite_score(score), feature.index))
    candidates.sort()
    next_kernel = set(retained)
    next_kernel.update(index for _, index in candidates[: max(0, target - len(next_kernel))])
    return next_kernel


def _search_state(
    *,
    iteration: int,
    current: RestrictedSolveResult,
    best: RestrictedSolveResult,
    kernel: set[int],
    B: int,
    C: float,
    tau: float,
    stagnation: int,
    route_started: float,
    deadline: float,
    improved: bool,
    total_features: int,
) -> SearchState:
    return SearchState(
        iteration=iteration,
        current_objective=current.objective,
        best_objective=best.objective,
        current_gap=current.diagnostics.mip_gap,
        best_bound=best.diagnostics.best_bound,
        kernel_size=len(kernel),
        feature_budget=int(B),
        total_features=total_features,
        stagnation_iterations=stagnation,
        elapsed_seconds=perf_counter() - route_started,
        remaining_seconds=max(0.0, deadline - perf_counter()),
        C=float(C),
        tau=float(tau),
        improved_last_iteration=bool(improved),
    )


def _update_feature_history(
    result: RestrictedSolveResult,
    kernel: set[int],
    *,
    selection_counts: np.ndarray,
    inactive_iterations: np.ndarray,
    kernel_age: np.ndarray,
) -> None:
    selected = np.abs(result.coefficients) > 1e-3
    selection_counts += selected.astype(int)
    inactive_iterations[:] = np.where(selected, 0, inactive_iterations + 1)
    in_kernel = np.asarray([index in kernel for index in range(selection_counts.size)])
    kernel_age[:] = np.where(in_kernel & ~selected, kernel_age + 1, 0)


def _history_record(
    *,
    iteration: int | str,
    kernel: set[int],
    result: RestrictedSolveResult,
    best: RestrictedSolveResult,
    improved: bool,
    added: set[int],
    removed: set[int],
    stagnation: int,
) -> dict[str, Any]:
    gap_scope = "full_model" if iteration == "final_refinement" else "restricted_kernel"
    return {
        "iteration": iteration,
        "kernel_size": len(kernel),
        "kernel_hash": _kernel_hash(kernel),
        "kernel_features": sorted(kernel),
        "support_size": len(result.support),
        "support_features": sorted(result.support),
        "objective": result.objective,
        "best_objective": best.objective,
        "best_bound": result.diagnostics.best_bound,
        "gap": result.diagnostics.mip_gap,
        "gap_scope": gap_scope,
        "node_count": result.diagnostics.node_count,
        "solve_time": result.solve_time,
        "improved": bool(improved),
        "added_features": sorted(added),
        "removed_features": sorted(removed),
        "stagnation": int(stagnation),
        "mip_start_status": result.mip_start_status,
        "solver_status": result.diagnostics.status,
    }


def _append_route_progress(
    destination: list[SolverProgressRecord],
    source: list[SolverProgressRecord],
    *,
    offset: float,
    comparable_full_model_bound: bool,
) -> None:
    global_incumbent = destination[-1].incumbent_objective if destination else None
    node_offset = int(destination[-1].node_count or 0) if destination else 0
    solution_offset = int(destination[-1].solution_count or 0) if destination else 0
    for record in source:
        candidate = record.incumbent_objective
        if candidate is not None:
            global_incumbent = (
                candidate if global_incumbent is None else min(global_incumbent, candidate)
            )
        bound = record.best_bound if comparable_full_model_bound else None
        relative_gap = _full_model_gap(global_incumbent, bound)
        destination.append(
            SolverProgressRecord(
                elapsed_seconds=max(0.0, offset + record.elapsed_seconds),
                incumbent_objective=global_incumbent,
                best_bound=bound,
                relative_gap=relative_gap,
                node_count=(
                    None if record.node_count is None else node_offset + int(record.node_count)
                ),
                solution_count=(
                    None
                    if record.solution_count is None
                    else solution_offset + int(record.solution_count)
                ),
            )
        )


def _full_model_gap(incumbent: float | None, bound: float | None) -> float | None:
    if incumbent is None or bound is None:
        return None
    incumbent = float(incumbent)
    bound = float(bound)
    if not np.isfinite(incumbent) or not np.isfinite(bound):
        return None
    return max(0.0, (incumbent - bound) / max(abs(incumbent), 1e-12))


def _kernel_hash(kernel: set[int]) -> str:
    payload = ",".join(str(index) for index in sorted(kernel)).encode("utf-8")
    return sha256(payload).hexdigest()


def _available_solve_time(deadline: float, requested: float) -> float:
    remaining = deadline - perf_counter()
    if remaining <= 1e-6:
        raise TimeBudgetExceeded("no wall-clock budget remains for a restricted solve")
    return max(1e-6, min(float(requested), remaining))


def _finite_score(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("kernel policy returned a NaN or infinite score")
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value
