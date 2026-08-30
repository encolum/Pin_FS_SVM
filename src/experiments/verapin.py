"""Configuration-driven VeraPin hardness, search, evolution, and final routes."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.data.synthetic import generate_synthetic_instance, save_synthetic_instance
from src.data.loaders import load_dataset
from src.data.preprocessing import fit_transform_training
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
from src.search.llm_evolution.replay import evolution_audit, load_replay_provider
from src.search.llm_evolution.schemas import PolicyCandidate
from src.search.objectives import solver_progress_summary
from src.search.policies.frozen_verapin import FrozenVeraPinPolicy
from src.search.policies.handcrafted_adks import ADKSWeights, HandcraftedADKSPolicy
from src.search.policies.static_ks import StaticKSPolicy
from src.search.restricted_solver import solve_restricted_pin_fs
from src.utils.serialization import read_json, write_json


_SYNTHETIC_FIELDS = {
    "n_samples",
    "n_features",
    "informative_ratio",
    "redundant_ratio",
    "correlation_strength",
    "positive_class_fraction",
    "label_noise_rate",
    "outlier_sample_rate",
    "outlier_feature_rate",
    "outlier_scale",
    "feature_budget_ratio",
    "seed",
}


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
        kind = instance.get("kind", "synthetic")
        if kind not in {"synthetic", "dataset"}:
            raise ValueError(f"instances[{index}].kind must be synthetic or dataset")
        required = (
            {"id", "split", *_SYNTHETIC_FIELDS}
            if kind == "synthetic"
            else {"id", "split", "dataset", "condition", "feature_budget"}
        )
        missing = sorted(required - set(instance))
        if missing:
            raise ValueError(f"instances[{index}] is missing fields: {missing}")
        unresolved.extend(
            f"instances[{index}].{name}" for name in required if instance.get(name) is None
        )
        if instance.get("id") is not None:
            instance_ids.append(str(instance["id"]))
        if instance.get("split") not in {None, "train", "validation", "test"}:
            raise ValueError(f"instances[{index}].split must be train, validation, or test")
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
        required_paths.append("frozen_policy_path")
    unresolved.extend(path for path in required_paths if _get_path(config, path) is None)
    splits = {instance.get("split") for instance in instances}
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


def run_hardness_benchmark(config: dict[str, Any]) -> Path:
    validate_verapin_config(config, command="hardness")
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
    _write_run_manifest(run_dir, config, instances, routes=["cold_cplex"])
    return run_dir


def run_static_kernel_search(config: dict[str, Any]) -> Path:
    validate_verapin_config(config, command="kernel-search")
    run_dir = _create_run_dir(config, "static-ks")
    instances = _policy_instances(config, run_dir=run_dir)
    return _run_kernel_route(
        config,
        run_dir=run_dir,
        instances=instances,
        route="static_ks",
        policy_factory=lambda: _static_policy(config),
    )


def run_adks(config: dict[str, Any]) -> Path:
    validate_verapin_config(config, command="adks")
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
            kernel_search_result_row(
                result,
                route="handcrafted_adks",
                instance_id=instance.instance_id,
                classification=_classification(instance, result.best_result),
                reference_objective=float(cold_row["final_objective"]),
            )
        )
        iteration_rows.extend(
            {"instance_id": instance.instance_id, "route": "handcrafted_adks", **record}
            for record in result.history
        )
        details[f"{instance.instance_id}:handcrafted_adks"] = _kernel_detail(result)
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
    provider = _provider(config)
    seed_candidates = [_candidate(value, config) for value in config["seed_policies"]]
    run_dir = Path(resume_dir).resolve() if resume_dir else _create_run_dir(config, "evolution")
    if resume_dir is not None and not (run_dir / "checkpoint.json").is_file():
        raise ValueError("resume directory does not contain checkpoint.json")
    instances = _policy_instances(config, run_dir=run_dir)
    training = [instance for instance in instances if instance.split == "train"]
    validation = [instance for instance in instances if instance.split == "validation"]
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
    run_dir = _create_run_dir(config, "final")
    instances = _policy_instances(config, run_dir=run_dir)
    candidate = PolicyCandidate.from_dict(read_json(config["frozen_policy_path"]))
    route_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        cold_row, cold_detail = _run_cold(instance, config["solver"])
        route_rows.append(cold_row)
        details[f"{instance.instance_id}:cold_cplex"] = cold_detail
        reference = float(cold_row["final_objective"])
        for route, policy in (
            ("handcrafted_adks", _adks_policy(config)),
            ("verapin_ks", FrozenVeraPinPolicy(candidate)),
        ):
            result = _run_engine(instance, config, policy)
            classification = _classification(instance, result.best_result)
            route_rows.append(
                kernel_search_result_row(
                    result,
                    route=route,
                    instance_id=instance.instance_id,
                    classification=classification,
                    reference_objective=reference,
                )
            )
            iteration_rows.extend(
                {"instance_id": instance.instance_id, "route": route, **record}
                for record in result.history
            )
            details[f"{instance.instance_id}:{route}"] = _kernel_detail(result)
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


def replay_evolution_audit(run_dir: str | Path) -> dict[str, Any]:
    return evolution_audit(run_dir)


def _run_kernel_route(
    config: dict[str, Any],
    *,
    run_dir: Path,
    instances: list[PolicyInstance],
    route: str,
    policy_factory,
) -> Path:
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for instance in instances:
        result = _run_engine(instance, config, policy_factory())
        rows.append(
            kernel_search_result_row(
                result,
                route=route,
                instance_id=instance.instance_id,
                classification=_classification(instance, result.best_result),
            )
        )
        iterations.extend(
            {"instance_id": instance.instance_id, "route": route, **record}
            for record in result.history
        )
        details[instance.instance_id] = _kernel_detail(result)
    write_kernel_search_results(run_dir, rows, iterations, details)
    _write_run_manifest(run_dir, config, instances, routes=[route])
    return run_dir


def _write_validation_baseline_comparison(
    run_dir: Path,
    *,
    config: dict[str, Any],
    instances: list[PolicyInstance],
    frozen: PolicyCandidate,
) -> None:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for route, policy in (
            ("handcrafted_adks", _adks_policy(config)),
            ("verapin_ks", FrozenVeraPinPolicy(frozen)),
        ):
            result = _run_engine(instance, config, policy)
            rows.append(
                kernel_search_result_row(
                    result,
                    route=route,
                    instance_id=instance.instance_id,
                    classification=_classification(instance, result.best_result),
                )
            )
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
    progress = solver_progress_summary(
        result.progress,
        horizon=runtime,
        reference_objective=result.objective,
    )
    row = {
        "instance_id": instance.instance_id,
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
        "selected_feature_count": len(result.support),
        "selected_feature_indices": sorted(result.support),
        "classification_scope": "optimization_instance",
        **_classification(instance, result),
    }
    detail = {
        "progress": [asdict(record) for record in result.progress],
        "diagnostics": result.diagnostics.to_dict(),
        "coefficients": result.coefficients,
        "intercept": result.intercept,
        "support": sorted(result.support),
    }
    return row, detail


def _classification(instance: PolicyInstance, result) -> dict[str, float]:
    predictions = np.where(
        instance.X @ result.coefficients + result.intercept >= 0,
        1,
        -1,
    )
    return classification_metrics(instance.y, predictions)


def _policy_instances(config: dict[str, Any], *, run_dir: Path) -> list[PolicyInstance]:
    problem = config["problem"]
    bounds = problem["coefficient_bounds"]
    result = []
    for specification in config["instances"]:
        if specification.get("kind", "synthetic") == "dataset":
            raw_X, y = load_dataset(
                str(specification["dataset"]),
                str(specification["condition"]),
                data_root=specification.get("data_root"),
                validate_classes=specification.get("condition") == "clean",
            )
            X, _, _ = fit_transform_training(raw_X)
            feature_budget = int(specification["feature_budget"])
            if feature_budget < 1 or feature_budget > X.shape[1]:
                raise ValueError(
                    f"instance {specification['id']} has invalid feature_budget={feature_budget}"
                )
            write_json(
                run_dir / "instances" / f"{specification['id']}.json",
                {
                    "instance_id": specification["id"],
                    "kind": "dataset",
                    "dataset": specification["dataset"],
                    "condition": specification["condition"],
                    "split": specification["split"],
                    "shape": list(X.shape),
                    "feature_budget": feature_budget,
                },
            )
        else:
            generated = generate_synthetic_instance(
                **{name: specification[name] for name in _SYNTHETIC_FIELDS},
                split=str(specification["split"]),
            )
            save_synthetic_instance(
                generated,
                run_dir / "instances",
                instance_id=str(specification["id"]),
            )
            X, y, feature_budget = generated.X, generated.y, generated.feature_budget
        result.append(
            PolicyInstance(
                instance_id=str(specification["id"]),
                split=str(specification["split"]),
                X=X,
                y=y,
                B=feature_budget,
                C=float(problem["C"]),
                tau=float(problem["tau"]),
                coefficient_bounds=(float(bounds["lower"]), float(bounds["upper"])),
            )
        )
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
) -> None:
    write_json(
        run_dir / "manifest.json",
        {
            "routes": routes,
            "instances": [
                {
                    "instance_id": instance.instance_id,
                    "split": instance.split,
                    "instance_hash": instance.instance_hash,
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
            "status": "complete",
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
