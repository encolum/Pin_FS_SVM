"""Configuration-driven VeraPin hardness, search, evolution, and final routes."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.evaluation.metrics import classification_metrics
from src.reporting.kernel_search_tables import (
    kernel_search_result_row,
    write_kernel_search_results,
)
from src.reporting.solver_profiles import write_solver_profiles
from src.search.kernel_engine import run_kernel_search
from src.search.llm_evolution.candidate_parser import parse_candidates, validate_candidate
from src.search.llm_evolution.evaluator import (
    FitnessNormalization,
    FitnessWeights,
    PolicyInstance,
)
from src.search.llm_evolution.evolution import EvolutionConfig, run_evolution
from src.search.llm_evolution.provider import EnvironmentLLMProvider, MockProvider
from src.search.llm_evolution.replay import load_replay_provider
from src.search.llm_evolution.references import prepare_fitness_references
from src.search.llm_evolution.schemas import PolicyCandidate
from src.search.progress import SolverProgressRecord, solver_progress_summary
from src.search.policies.frozen_verapin import FrozenVeraPinPolicy
from src.search.policies.handcrafted_adks import ADKSWeights, HandcraftedADKSPolicy
from src.search.policies.static_ks import StaticKSPolicy
from src.search.restricted_solver import solve_restricted_pin_fs
from src.utils.serialization import read_json, write_json
from src.utils.matrices import matrix_metadata
from .benchmark_instances import (
    SYNTHETIC_FIELDS, build_prepared_instances, corruption_choices,
    assert_research_groups_disjoint,
)
from .readiness import check_execution_readiness, cplex_environment_report


def validate_verapin_config(config: dict[str, Any], *, command: str) -> None:
    """Reject unresolved author decisions before any run directory or solve is created."""
    if command not in {"hardness", "kernel-search", "adks", "evolve-verapin", "evaluate-verapin"}:
        raise ValueError(f"unknown VeraPin command: {command}")
    _require_mapping(config, "problem")
    solver = _require_mapping(config, "solver")
    instances = config.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("instances must be a non-empty list")
    unresolved: list[str] = []
    instance_ids: list[str] = []
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise ValueError(f"instances[{index}] must be a mapping")
        kind = instance.get("kind")
        if kind not in {"synthetic", "benchmark"}:
            raise ValueError(f"instances[{index}].kind must be synthetic or benchmark")
        required = (
            {"id", "kind", "research_split", "condition", *SYNTHETIC_FIELDS}
            if kind == "synthetic"
            else {"id", "kind", "research_split", "dataset", "condition", "feature_budget"}
        )
        if kind == "benchmark":
            required.add("source_partition_policy")
        missing = sorted(required - set(instance))
        if missing:
            raise ValueError(f"instances[{index}] is missing fields: {missing}")
        unresolved.extend(
            f"instances[{index}].{name}" for name in required if instance.get(name) is None
        )
        if instance.get("id") is not None:
            instance_ids.append(str(instance["id"]))
        if instance.get("research_split") not in {None, "train", "validation", "test"}:
            raise ValueError(f"instances[{index}].research_split must be train, validation, or test")
        if instance.get("id") is not None:
            import re
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(instance["id"])):
                raise ValueError("instance IDs must be safe filename components")
        if kind == "benchmark":
            from src.data.benchmark_data import read_benchmark_registry, validate_partition_policy, DEFAULT_REGISTRY_PATH
            registry, _ = read_benchmark_registry(config.get("benchmark_registry", DEFAULT_REGISTRY_PATH))
            if instance["dataset"] not in registry:
                raise ValueError("unknown benchmark dataset")
            validate_partition_policy(instance["dataset"], instance["source_partition_policy"])
            if command == "evolve-verapin" and config.get("allow_real_benchmark_evolution") is not True:
                raise ValueError("real benchmarks are held out from evolution unless explicitly overridden")
        corruption_choices(config, instance)
        if command == "hardness" and instance["condition"] != "clean":
            raise ValueError("initial solver hardness protocol must be clean")
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("instance IDs must be unique")
    required_paths = [
        "problem.C",
        "problem.tau",
        "problem.coefficient_bounds.lower",
        "problem.coefficient_bounds.upper",
        "solver.backend",
        "solver.threads",
        "solver.total_time_limit",
        "solver.mip_gap",
    ]
    if command != "hardness":
        required_paths.extend(
            [
                "solver.subproblem_time_limit",
                "search.max_iterations",
                "search.final_full_refinement",
                "search.final_refinement_fraction",
                "search.signal_options.use_lp",
            ]
        )
    if command == "kernel-search":
        required_paths.extend(
            [
                "static_policy.score_name",
                "static_policy.initial_kernel_size",
                "static_policy.bucket_size",
            ]
        )
    if command in {"adks", "evolve-verapin", "evaluate-verapin"}:
        required_paths.extend(_adks_required_paths())
    if command == "evolve-verapin":
        required_paths.extend(
            [
                "evolution.generations",
                "evolution.population_size",
                "evolution.parent_count",
                "evolution.candidates_per_generation",
                "evolution.maximum_similarity",
                "evolution.seed",
                "fitness.weights.primal_integral",
                "fitness.weights.final_gap",
                "fitness.weights.failure_rate",
                "fitness.weights.overhead",
                "fitness.normalization.primal_integral_scale",
                "fitness.normalization.final_gap_scale",
                "fitness.normalization.overhead_scale",
                "fitness.target_gap",
                "llm.provider",
                "seed_policies",
            ]
        )
    if command == "evaluate-verapin":
        required_paths.extend(
            [
                "frozen_policy_path",
                "classification.outer_folds",
                "classification.outer_seed",
                "classification.inner_folds",
                "classification.inner_seed",
                "classification.parameter_grid",
                "classification.tuning_solver.time_limit",
                "classification.tuning_solver.mip_gap",
            ]
        )
    unresolved.extend(path for path in required_paths if _get_path(config, path) is None)
    splits = {instance.get("research_split") for instance in instances}
    if command == "evolve-verapin" and "test" in splits:
        raise ValueError("evolution configuration must not contain held-out test instances")
    if command == "evaluate-verapin" and splits != {"test"}:
        raise ValueError("final VeraPin evaluation accepts held-out test instances only")
    if unresolved:
        raise ValueError(
            "unresolved author decisions: " + ", ".join(sorted(set(unresolved)))
        )
    if solver.get("backend") not in {"scipy", "cplex"}:
        raise ValueError("solver.backend must be scipy or cplex")
    if command in {"hardness", "adks", "evaluate-verapin"} and solver.get("backend") != "cplex":
        raise ValueError("Cold CPLEX comparison requires solver.backend='cplex'")
    if float(solver["total_time_limit"]) <= 0 or int(solver["threads"]) < 1:
        raise ValueError("solver time limits and threads must be positive")
    if float(solver["mip_gap"]) < 0:
        raise ValueError("solver.mip_gap must be non-negative")
    if command != "hardness" and float(solver["subproblem_time_limit"]) <= 0:
        raise ValueError("solver.subproblem_time_limit must be positive")
    if float(config["problem"]["C"]) <= 0 or not 0 < float(config["problem"]["tau"]) <= 1:
        raise ValueError("problem requires C > 0 and 0 < tau <= 1")
    bounds = config["problem"]["coefficient_bounds"]
    if not float(bounds["lower"]) < 0 < float(bounds["upper"]):
        raise ValueError("problem coefficient bounds must satisfy lower < 0 < upper")
    if not np.isfinite([config["problem"]["C"], config["problem"]["tau"], bounds["lower"], bounds["upper"],
                        solver["total_time_limit"], solver["mip_gap"]]).all():
        raise ValueError("problem and solver parameters must be finite")
    purpose = config.get("execution", {}).get("purpose", "scientific")
    if bounds.get("author_confirmed") is not True and not (
            purpose == "provisional_pilot" and config.get("execution", {}).get("parameters_provisional") is True):
        raise ValueError("coefficient bounds require author approval or explicit provisional_pilot status")
    if command == "evaluate-verapin":
        _validate_nested_classification(config)
    if command == "evolve-verapin":
        if not {"train", "validation"}.issubset(splits):
            raise ValueError("evolution requires separate train and validation instances")
        if not isinstance(config["seed_policies"], list) or not config["seed_policies"]:
            raise ValueError("seed_policies must be a non-empty list")
        provider = config["llm"]["provider"]
        if provider == "environment":
            for path in (
                "llm.temperature",
                "llm.timeout_seconds",
                "llm.input_cost_per_million",
                "llm.output_cost_per_million",
            ):
                if _get_path(config, path) is None:
                    raise ValueError(f"unresolved author decisions: {path}")
        elif provider == "mock":
            if not config["llm"].get("responses"):
                raise ValueError("llm.responses must be non-empty for mock evolution")
        elif provider == "replay":
            if config["llm"].get("replay_run_dir") is None:
                raise ValueError("llm.replay_run_dir is required for replay evolution")
        else:
            raise ValueError("llm.provider must be mock, replay, or environment")
        EvolutionConfig(**config["evolution"])
        FitnessWeights(**config["fitness"]["weights"])
        FitnessNormalization(**config["fitness"]["normalization"])
    if command == "kernel-search":
        _static_policy(config)
    if command in {"adks", "evolve-verapin", "evaluate-verapin"}:
        _adks_policy(config)
    if command == "evaluate-verapin":
        verify_policy_file(config["frozen_policy_path"])


def _validate_nested_classification(config):
    classification = config.get("classification", {})
    tolerance = float(classification.get("selection_tolerance", 1e-12))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("classification.selection_tolerance must be finite and non-negative")
    required = ("inner_folds", "inner_seed", "parameter_grid", "tuning_solver")
    if any(classification.get(name) is None for name in required):
        raise ValueError("benchmark classification requires inner_folds, inner_seed, parameter_grid, tuning_solver")
    for name in ("inner_folds", "outer_folds"):
        if type(classification.get(name)) is not int or classification[name] < 2:
            raise ValueError(f"classification.{name} must be an integer >= 2")
    for name in ("inner_seed", "outer_seed"):
        if type(classification.get(name)) is not int or classification[name] < 0:
            raise ValueError(f"classification.{name} must be a nonnegative integer")
    if config.get("execution", {}).get("purpose") != "provisional_pilot" and (
            classification["outer_folds"], classification["inner_folds"]) != (5, 3):
        raise ValueError("scientific classification protocol requires 5 outer x 3 inner folds")
    grid = classification["parameter_grid"]
    if not isinstance(grid, dict) or set(grid) != {"B", "C", "tau"}:
        raise ValueError("parameter_grid must declare exactly B, C and tau")
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError("each parameter grid must be a nonempty list")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError("tuning parameters must be finite positive numbers")
            if (name == "B" and type(value) is not int) or (name == "tau" and value > 1):
                raise ValueError("B must be integer and tau must be <= 1")
    tuning = classification["tuning_solver"]
    if not isinstance(tuning, dict) or set(tuning) != {"backend", "time_limit", "mip_gap", "threads"}:
        raise ValueError("tuning_solver needs explicit backend, time_limit, mip_gap, threads")
    if tuning["backend"] not in {"cplex", "scipy"} or not np.isfinite(tuning["time_limit"]) or tuning["time_limit"] <= 0:
        raise ValueError("invalid tuning backend or time limit")
    if type(tuning["threads"]) is not int or tuning["threads"] < 1 or not np.isfinite(tuning["mip_gap"]) or tuning["mip_gap"] < 0:
        raise ValueError("invalid tuning threads or gap")


def run_hardness_benchmark(config: dict[str, Any]) -> Path:
    validate_verapin_config(config, command="hardness")
    check_execution_readiness(config, "hardness")
    run_dir = _create_run_dir(config, "hardness")
    instances = _policy_instances(config, run_dir=run_dir)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        row, detail = _run_cold(instance, config["solver"])
        rows.append(row)
        details[instance.instance_id] = detail
    write_solver_profiles(run_dir, rows)
    write_json(run_dir / "cold_cplex_details.json", details)
    successful = sum(row.get("final_objective") is not None for row in rows)
    _write_run_manifest(run_dir, config, instances, routes=["cold_cplex"],
                        status="complete" if successful == len(rows) else "partial" if successful else "failed")
    return run_dir


def run_static_kernel_search(config: dict[str, Any]) -> Path:
    validate_verapin_config(config, command="kernel-search")
    check_execution_readiness(config, "kernel-search")
    run_dir = _create_run_dir(config, "static-ks")
    instances = _policy_instances(config, run_dir=run_dir)
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        result = _run_engine(instance, config, _static_policy(config))
        rows.append(_kernel_route_row(result, route="static_ks", instance=instance))
        iterations.extend(
            {"instance_id": instance.instance_id, "route": "static_ks", **record}
            for record in result.history
        )
        details[f"{instance.instance_id}:static_ks"] = _kernel_detail(result)
    _apply_common_reference(rows, details)
    write_kernel_search_results(run_dir, rows, iterations, details)
    _write_run_manifest(run_dir, config, instances, routes=["static_ks"])
    return run_dir


def run_adks(config: dict[str, Any]) -> Path:
    validate_verapin_config(config, command="adks")
    check_execution_readiness(config, "adks")
    run_dir = _create_run_dir(config, "adks")
    instances = _policy_instances(config, run_dir=run_dir)
    route_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        cold_row, cold_detail = _run_cold(instance, config["solver"])
        route_rows.append(cold_row)
        details[f"{instance.instance_id}:cold_cplex"] = cold_detail
        result = _run_engine(instance, config, _adks_policy(config))
        route_rows.append(
            _kernel_route_row(
                result,
                route="handcrafted_adks",
                instance=instance,
            )
        )
        iteration_rows.extend(
            {"instance_id": instance.instance_id, "route": "handcrafted_adks", **record}
            for record in result.history
        )
        details[f"{instance.instance_id}:handcrafted_adks"] = _kernel_detail(result)
    _apply_common_reference(route_rows, details)
    write_kernel_search_results(run_dir, route_rows, iteration_rows, details)
    _write_run_manifest(
        run_dir,
        config,
        instances,
        routes=["cold_cplex", "handcrafted_adks"],
    )
    return run_dir


def run_verapin_evolution(config: dict[str, Any], *, resume_dir: str | Path | None = None) -> Path:
    validate_verapin_config(config, command="evolve-verapin")
    check_execution_readiness(config, "evolve-verapin")
    seed_candidates = [_candidate(value, config) for value in config["seed_policies"]]
    run_dir = Path(resume_dir).resolve() if resume_dir else _create_run_dir(config, "evolution")
    if resume_dir is not None and not (run_dir / "checkpoint.json").is_file():
        raise ValueError("resume directory does not contain checkpoint.json")
    instances = _policy_instances(config, run_dir=run_dir)
    instances = prepare_fitness_references(instances, solver_config=_engine_config(config),
        adks_config=config["adks_policy"], output_path=run_dir / "fitness_references.json",
        reuse=resume_dir is not None)
    provider = _provider(config)
    training = [instance for instance in instances if instance.research_split == "train"]
    validation = [instance for instance in instances if instance.research_split == "validation"]
    fitness = config["fitness"]
    result = run_evolution(
        seed_candidates=seed_candidates,
        training_instances=training,
        validation_instances=validation,
        provider=provider,
        evolution_config=EvolutionConfig(**config["evolution"]),
        solver_config=_engine_config(config),
        fitness_weights=FitnessWeights(**fitness["weights"]),
        normalization=FitnessNormalization(**fitness["normalization"]),
        target_gap=float(fitness["target_gap"]),
        run_dir=run_dir,
        resume=resume_dir is not None,
    )
    frozen_output = Path(
        config.get(
            "frozen_policy_output",
            "artifacts_verapin/policies/frozen_verapin_policy.json",
        )
    )
    write_json(frozen_output, result.frozen_candidate.to_dict())
    _write_validation_baseline_comparison(
        run_dir,
        config=config,
        instances=validation,
        frozen=result.frozen_candidate,
    )
    _write_run_manifest(run_dir, config, instances, routes=["verapin_evolution"])
    return run_dir


def run_verapin_final(config: dict[str, Any]) -> Path:
    """Evaluate cold CPLEX, ADKS, and one frozen policy with no LLM provider."""
    validate_verapin_config(config, command="evaluate-verapin")
    check_execution_readiness(config, "evaluate-verapin")
    run_dir = _create_run_dir(config, "final")
    instances = _policy_instances(config, run_dir=run_dir, outer_evaluation=True)
    candidate = PolicyCandidate.from_dict(read_json(config["frozen_policy_path"]))
    route_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        cold_row, cold_detail = _run_cold(instance, config["solver"])
        route_rows.append(cold_row)
        details[f"{instance.instance_id}:cold_cplex"] = cold_detail
        for route, policy in (
            ("handcrafted_adks", _adks_policy(config)),
            ("verapin_ks", FrozenVeraPinPolicy(candidate)),
        ):
            result = _run_engine(instance, config, policy)
            classification = _classification(instance, result.best_result)
            route_rows.append(
                _kernel_route_row(
                    result,
                    route=route,
                    instance=instance,
                    classification=classification,
                )
            )
            iteration_rows.extend(
                {"instance_id": instance.instance_id, "route": route, **record}
                for record in result.history
            )
            details[f"{instance.instance_id}:{route}"] = _kernel_detail(result)
    _apply_common_reference(route_rows, details)
    write_kernel_search_results(run_dir, route_rows, iteration_rows, details)
    _write_run_manifest(
        run_dir,
        config,
        instances,
        routes=["cold_cplex", "handcrafted_adks", "verapin_ks"],
    )
    return run_dir


def verify_policy_file(path: str | Path) -> dict[str, Any]:
    candidate = validate_candidate(PolicyCandidate.from_dict(read_json(path)))
    FrozenVeraPinPolicy(candidate)
    return {
        "policy_id": candidate.policy_id,
        "policy_hash": candidate.policy_hash,
        "schema_version": candidate.schema_version,
        "valid": True,
    }


def _write_validation_baseline_comparison(
    run_dir: Path,
    *,
    config: dict[str, Any],
    instances: list[PolicyInstance],
    frozen: PolicyCandidate,
) -> None:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        for route, policy in (
            ("handcrafted_adks", _adks_policy(config)),
            ("verapin_ks", FrozenVeraPinPolicy(frozen)),
        ):
            result = _run_engine(instance, config, policy)
            rows.append(
                _kernel_route_row(
                    result,
                    route=route,
                    instance=instance,
                )
            )
            details[f"{instance.instance_id}:{route}"] = _kernel_detail(result)
    _apply_common_reference(rows, details)
    route_integrals = {
        route: float(
            np.mean(
                [float(row["primal_integral"]) for row in rows if row["route"] == route]
            )
        )
        for route in {row["route"] for row in rows}
    }
    write_json(
        run_dir / "validation_adks_comparison.json",
        {
            "rows": rows,
            "mean_primal_integral": route_integrals,
            "verapin_improves_after_online_overhead": (
                route_integrals["verapin_ks"]
                < route_integrals["handcrafted_adks"]
            ),
            "selection_data": "validation",
            "llm_calls": 0,
        },
    )


def _kernel_route_row(
    result,
    *,
    route: str,
    instance: PolicyInstance,
    classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if classification is None:
        classification = _classification(instance, result.best_result)
    row = kernel_search_result_row(
        result,
        route=route,
        instance_id=instance.instance_id,
        classification=classification,
    )
    row.update(_instance_identity(instance))
    return row


def _instance_identity(instance: PolicyInstance) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "base_instance_id": instance.base_instance_id or instance.instance_id,
    }
    if instance.outer_fold is not None:
        identity["outer_fold"] = int(instance.outer_fold)
    return identity


def _apply_common_reference(
    rows: list[dict[str, Any]],
    details: dict[str, Any],
) -> None:
    """Recompute anytime metrics using one best-known objective per instance."""
    instance_ids = sorted({str(row["instance_id"]) for row in rows})
    for instance_id in instance_ids:
        instance_rows = [row for row in rows if str(row["instance_id"]) == instance_id]
        feasible = [float(row["final_objective"]) for row in instance_rows
                    if row.get("final_objective") is not None and np.isfinite(row["final_objective"])]
        reference = min(feasible) if feasible else None
        horizon = max(max(float(details[f"{instance_id}:{row['route']}"]["total_runtime"]),
                          float(details[f"{instance_id}:{row['route']}"].get("time_budget", 0)))
                      for row in instance_rows)
        for row in instance_rows:
            key = f"{instance_id}:{row['route']}"
            if key not in details:
                raise KeyError(f"missing progress detail for {key}")
            detail = details[key]
            trajectory = [
                SolverProgressRecord(**record) for record in detail.get("progress", [])
            ]
            summary = solver_progress_summary(
                trajectory,
                horizon=horizon,
                reference_objective=reference,
            )
            row.update(summary)
            row["primal_integral_reference_objective"] = reference
            row["primal_integral_reference_scope"] = "best_known_across_routes"
            row["primal_integral_horizon"] = horizon
            row["gap_scope"] = "full_model_only"


def _offset_progress_to_route_time(
    records: list[SolverProgressRecord],
    total_runtime: float,
) -> list[SolverProgressRecord]:
    if not records:
        return []
    offset = max(0.0, float(total_runtime) - float(records[-1].elapsed_seconds))
    return [
        SolverProgressRecord(
            elapsed_seconds=offset + float(record.elapsed_seconds),
            incumbent_objective=record.incumbent_objective,
            best_bound=record.best_bound,
            relative_gap=record.relative_gap,
            node_count=record.node_count,
            solution_count=record.solution_count,
        )
        for record in records
    ]


def _run_engine(instance: PolicyInstance, config: dict[str, Any], policy):
    return run_kernel_search(
        instance.X,
        instance.y,
        policy=policy,
        B=instance.B,
        C=instance.C,
        tau=instance.tau,
        coefficient_bounds=instance.coefficient_bounds,
        **_engine_config(config),
    )


def _run_cold(
    instance: PolicyInstance, solver: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = perf_counter()
    try:
        return _run_cold_impl(instance, solver)
    except Exception as exc:
        # Solver/license/time-limit failures are evidence, not successful hard
        # instances. Keep an auditable failure instead of losing the run report.
        runtime = perf_counter() - started
        status = "license_limit" if "1016" in str(exc) else "failed"
        return ({"instance_id": instance.instance_id, **_instance_identity(instance),
                 "route": "cold_cplex", "solver_status": status, "error": str(exc),
                 "final_objective": None, "final_gap": None, "node_count": None,
                 "first_feasible_time": None, "total_runtime": runtime,
                 "memory_estimate": matrix_metadata(instance.X)},
                {"progress": [], "total_runtime": runtime, "time_budget": solver["total_time_limit"],
                 "diagnostics": {"status": status, "error": str(exc)}})


def _run_cold_impl(instance: PolicyInstance, solver: dict[str, Any]):
    started = perf_counter()
    result = solve_restricted_pin_fs(
        instance.X,
        instance.y,
        kernel=set(range(instance.X.shape[1])),
        B=instance.B,
        C=instance.C,
        tau=instance.tau,
        coefficient_bounds=instance.coefficient_bounds,
        backend="cplex",
        time_limit=float(solver["total_time_limit"]),
        mip_gap=float(solver["mip_gap"]),
        threads=int(solver["threads"]),
        collect_progress=True,
        deadline=started + float(solver["total_time_limit"]),
    )
    runtime = perf_counter() - started
    route_progress = _offset_progress_to_route_time(result.progress, runtime)
    progress = solver_progress_summary(
        route_progress,
        horizon=runtime,
        reference_objective=result.objective,
    )
    classification = _classification(instance, result)
    row = {
        "instance_id": instance.instance_id,
        **_instance_identity(instance),
        "route": "cold_cplex",
        "kernel_policy": "none",
        "initial_kernel_size": instance.X.shape[1],
        "final_kernel_size": instance.X.shape[1],
        "iterations": 1,
        "improving_iterations": 1,
        "restricted_solves": 1,
        **progress,
        "final_objective": result.objective,
        "final_best_bound": result.diagnostics.best_bound,
        "final_gap": result.diagnostics.mip_gap,
        "node_count": result.diagnostics.node_count,
        "total_runtime": runtime,
        "signal_overhead": 0.0,
        "policy_overhead": 0.0,
        "lp_relaxation_overhead": 0.0,
        "mip_start_status": None,
        "solver_status": result.diagnostics.status,
        "model_build_time": result.model_build_time,
        "memory_estimate": matrix_metadata(instance.X),
        "selected_feature_count": len(result.support),
        "selected_feature_indices": sorted(result.support),
    }
    if classification:
        row["classification_scope"] = "outer_test"
        row.update(classification)
    detail = {
        "progress": [asdict(record) for record in route_progress],
        "time_budget": solver["total_time_limit"],
        "total_runtime": runtime,
        "diagnostics": result.diagnostics.to_dict(),
        "coefficients": result.coefficients,
        "intercept": result.intercept,
        "support": sorted(result.support),
    }
    return row, detail


def _classification(instance: PolicyInstance, result) -> dict[str, float]:
    if instance.X_test is None and instance.y_test is None:
        return {}
    if instance.X_test is None or instance.y_test is None:
        raise ValueError("outer-test features and labels must be provided together")
    predictions = np.where(
        instance.X_test @ result.coefficients + result.intercept >= 0,
        1,
        -1,
    )
    return classification_metrics(instance.y_test, predictions)


def _policy_instances(
    config: dict[str, Any],
    *,
    run_dir: Path,
    outer_evaluation: bool = False,
) -> list[PolicyInstance]:
    result: list[PolicyInstance] = []
    for specification in config["instances"]:
        result.extend(build_prepared_instances(
            config, specification, run_dir=run_dir, outer_evaluation=outer_evaluation
        ))
    assert_research_groups_disjoint(result)
    return result


def _engine_config(config: dict[str, Any]) -> dict[str, Any]:
    solver = config["solver"]
    search = config["search"]
    return {
        "total_time_limit": float(solver["total_time_limit"]),
        "subproblem_time_limit": float(solver["subproblem_time_limit"]),
        "max_iterations": int(search["max_iterations"]),
        "backend": str(solver["backend"]),
        "threads": int(solver["threads"]),
        "final_full_refinement": bool(search["final_full_refinement"]),
        "final_refinement_fraction": float(search["final_refinement_fraction"]),
        "seed": int(search.get("seed", 0)),
        "mip_gap": float(solver["mip_gap"]),
        "acceptance_epsilon": float(search.get("acceptance_epsilon", 1e-9)),
        "signal_options": dict(search["signal_options"]),
    }


def _static_policy(config: dict[str, Any]) -> StaticKSPolicy:
    return StaticKSPolicy(**config["static_policy"])


def _adks_policy(config: dict[str, Any]) -> HandcraftedADKSPolicy:
    section = dict(config["adks_policy"])
    weights = ADKSWeights(**section.pop("weights"))
    return HandcraftedADKSPolicy(weights=weights, **section)


def _provider(config: dict[str, Any]):
    section = config["llm"]
    provider = section["provider"]
    if provider == "mock":
        responses = section.get("responses")
        if not isinstance(responses, list) or not responses:
            raise ValueError("llm.responses must be non-empty for the mock provider")
        return MockProvider([str(value) for value in responses])
    if provider == "replay":
        replay_dir = section.get("replay_run_dir")
        if replay_dir is None:
            raise ValueError("llm.replay_run_dir is required for replay")
        return load_replay_provider(replay_dir)
    if provider == "environment":
        return EnvironmentLLMProvider(
            temperature=float(section["temperature"]),
            timeout_seconds=float(section["timeout_seconds"]),
            input_cost_per_million=section.get("input_cost_per_million"),
            output_cost_per_million=section.get("output_cost_per_million"),
        )
    raise ValueError("llm.provider must be mock, replay, or environment")


def _candidate(value: Any, config: dict[str, Any]) -> PolicyCandidate:
    if isinstance(value, dict):
        return validate_candidate(PolicyCandidate.from_dict(value))
    path = Path(str(value))
    if not path.is_absolute() and config.get("_config_path"):
        path = Path(config["_config_path"]).resolve().parent / path
    candidates = parse_candidates(path.read_text(encoding="utf-8"))
    if len(candidates) != 1:
        raise ValueError(f"seed policy file {path} must contain exactly one candidate")
    return candidates[0]


def _kernel_detail(result) -> dict[str, Any]:
    return {
        "time_budget": result.metadata.get("time_budget", result.total_runtime),
        "progress": list(result.metadata.get("route_progress", [])),
        "total_runtime": result.total_runtime,
        "history": result.history,
        "metadata": result.metadata,
        "best": {
            "objective": result.best_result.objective,
            "support": sorted(result.best_result.support),
            "coefficients": result.best_result.coefficients,
            "intercept": result.best_result.intercept,
            "diagnostics": result.best_result.diagnostics.to_dict(),
        },
    }


def _create_run_dir(config: dict[str, Any], label: str) -> Path:
    root = Path(config.get("output", {}).get("root", "results_verapin"))
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}"
    run_dir = (root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.yaml",
        {key: value for key, value in config.items() if not key.startswith("_")},
    )
    return run_dir


def _write_run_manifest(
    run_dir: Path,
    config: dict[str, Any],
    instances: list[PolicyInstance],
    *,
    routes: list[str],
    status: str = "complete",
) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "routes": routes,
            "instances": [
                {
                    "instance_id": instance.instance_id,
                    "reference_objective": instance.reference_objective,
                    "fitness_horizon": instance.fitness_horizon,
                    "base_instance_id": instance.base_instance_id
                    or instance.instance_id,
                    "outer_fold": instance.outer_fold,
                    "research_split": instance.research_split,
                    "data_preparation": instance.metadata,
                    "B": instance.B, "C": instance.C, "tau": instance.tau,
                    "coefficient_bounds": instance.coefficient_bounds,
                    "instance_hash": instance.instance_hash,
                    "optimization_partition": "outer_train"
                    if instance.X_test is not None
                    else "full_instance",
                    "classification_partition": "outer_test"
                    if instance.X_test is not None
                    else None,
                }
                for instance in instances
            ],
            "fair_comparison": {
                "problem": config["problem"],
                "solver": config["solver"],
                "same_data": True,
                "same_preprocessing": True,
                "online_overhead_included": True,
            },
            "status": status,
            "execution": config.get("execution", {}),
            "cplex_environment": cplex_environment_report(),
            "real_benchmarks_held_out_from_evolution": not config.get("allow_real_benchmark_evolution", False),
        },
    )


def _adks_required_paths() -> list[str]:
    names = list(ADKSWeights.__dataclass_fields__)
    paths = [f"adks_policy.weights.{name}" for name in names]
    paths.extend(
        [
            "adks_policy.initial_kernel_size",
            "adks_policy.minimum_kernel_size",
            "adks_policy.maximum_kernel_size",
            "adks_policy.stagnation_threshold",
            "adks_policy.focus_fraction",
            "adks_policy.expansion_fraction",
        ]
    )
    return paths


def _get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _require_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value
