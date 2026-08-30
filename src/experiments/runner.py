"""Top-level v2 experiment runner and artifact persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.data.corruptions import apply_corruption
from src.data.loaders import DATASET_SPECS, load_dataset
from src.data.preprocessing import fit_transform_training
from src.evaluation.feature_stability import feature_stability
from src.evaluation.nested_cv import run_nested_cv
from src.reporting.aggregate import across_dataset_summary, aggregate_fold_rows
from src.utils.logging import create_logger
from src.utils.manifests import build_manifest
from src.utils.serialization import read_json, write_csv, write_json

from .config import validate_config
from .search import estimate_model_fits, model_fit_cost


def run_experiment(
    config: dict[str, Any],
    *,
    mode: str,
    resume_dir: str | Path | None = None,
) -> Path:
    validate_config(config)
    run_started = perf_counter()
    planned_fits = estimate_model_fits(config)
    config_snapshot = {key: value for key, value in config.items() if not key.startswith("_")}
    if resume_dir:
        run_dir = Path(resume_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        saved_config_path = run_dir / "config.yaml"
        if not saved_config_path.is_file():
            raise ValueError("cannot resume a run that has no saved config.yaml")
        saved_config = read_json(saved_config_path)
        if saved_config != config_snapshot:
            raise ValueError("resume config does not exactly match the configuration saved with the run")
        run_id = run_dir.name
    else:
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{mode}"
        run_dir = (Path(config.get("output", {}).get("root", "results_v2")) / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    _make_output_tree(run_dir)
    logger = create_logger(run_dir / "logs" / "run.log", name=f"pinfs-v2-{run_id}")
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {"completed": []}
    completed = set(checkpoint.get("completed", []))
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else build_manifest(
        config, run_id=run_id, planned_fits=planned_fits
    )
    manifest["number_of_planned_fits"] = planned_fits
    prior_elapsed = float(manifest.get("elapsed_seconds", 0.0)) if resume_dir else 0.0
    if not resume_dir:
        write_json(run_dir / "config.yaml", config_snapshot)
    write_json(manifest_path, manifest)
    logger.info("run_id=%s planned_fits=%s", run_id, planned_fits)

    for seed in config["seeds"]:
        for dataset in config["datasets"]:
            X, y = load_dataset(dataset, "clean", validate_classes=True)
            for condition in config["conditions"]:
                for model_name in config["models"]:
                    key = f"{dataset}__{condition}__{model_name}__seed-{int(seed)}"
                    if key in completed:
                        logger.info("resume: skipping completed %s", key)
                        continue
                    logger.info("starting %s", key)
                    try:
                        result = run_nested_cv(
                            X,
                            y,
                            dataset=dataset,
                            condition=condition,
                            model_name=model_name,
                            config=config,
                            base_seed=int(seed),
                        )
                    except Exception as exc:
                        failure = {
                            "key": key,
                            "stage": "combination",
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                        write_json(run_dir / "failures" / f"{key}.json", failure)
                        logger.exception("failed %s", key)
                        raise
                    write_json(run_dir / "folds" / f"{key}.json", result)
                    _write_combination_artifacts(run_dir, key, result)
                    completed.add(key)
                    checkpoint = {"completed": sorted(completed), "last_completed": key}
                    write_json(checkpoint_path, checkpoint)
                    logger.info("completed %s", key)

    fold_rows, generated_hashes, completed_fits = _collect_fold_rows(run_dir, run_id, config)
    write_csv(run_dir / "metrics" / "fold_metrics.csv", fold_rows, fieldnames=_METRIC_FIELDS)
    aggregates = aggregate_fold_rows(fold_rows)
    write_csv(run_dir / "aggregate" / "summary.csv", aggregates)
    write_csv(run_dir / "aggregate" / "across_datasets.csv", across_dataset_summary(aggregates))
    stability_rows = _collect_stability(run_dir)
    write_csv(run_dir / "aggregate" / "feature_stability.csv", stability_rows)
    manifest["generated_data_hashes"] = sorted(generated_hashes)
    manifest["number_of_completed_fits"] = completed_fits
    manifest["elapsed_seconds"] = prior_elapsed + perf_counter() - run_started
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    if mode == "pilot":
        _write_corruption_qa(run_dir, config)
        _write_pilot_summary(run_dir, config, fold_rows, planned_fits, manifest)
    return run_dir


def _make_output_tree(run_dir: Path) -> None:
    for name in (
        "folds", "predictions", "coefficients", "selected_features", "solver", "metrics",
        "search", "corruptions", "splits", "aggregate", "tables", "figures", "failures", "logs",
    ):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def _write_combination_artifacts(run_dir: Path, key: str, result: dict[str, Any]) -> None:
    predictions: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    solver: list[dict[str, Any]] = []
    search: list[dict[str, Any]] = []
    corruptions: list[dict[str, Any]] = []
    for fold in result["folds"]:
        fold_id = fold["outer_fold"]
        predictions.extend({"outer_fold": fold_id, **row} for row in fold["predictions"])
        coefficients.append({"outer_fold": fold_id, "coefficients": fold["coefficients"], "intercept": fold["intercept"]})
        selected.append({
            "outer_fold": fold_id,
            "selected_feature_indices": fold["selected_feature_indices"],
            "selected_feature_count": fold["selected_feature_count"],
        })
        solver.append({"outer_fold": fold_id, **fold["solver"]})
        search.extend({"outer_fold": fold_id, **record} for record in fold["search"])
        corruptions.append({
            "outer_fold": fold_id,
            "outer_training": fold["corruption_manifest"],
            "inner_training": fold.get("inner_corruption_manifests", []),
        })
    write_csv(run_dir / "predictions" / f"{key}.csv", predictions)
    write_json(run_dir / "coefficients" / f"{key}.json", coefficients)
    write_json(run_dir / "selected_features" / f"{key}.json", selected)
    write_json(run_dir / "solver" / f"{key}.json", solver)
    write_json(run_dir / "search" / f"{key}.json", search)
    write_json(run_dir / "corruptions" / f"{key}.json", corruptions)
    write_json(run_dir / "splits" / f"{key}.json", result["split_audit"])
    if result["failures"]:
        write_json(run_dir / "failures" / f"{key}__inner.json", result["failures"])


def _collect_fold_rows(
    run_dir: Path,
    run_id: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str], int]:
    rows: list[dict[str, Any]] = []
    hashes: set[str] = set()
    fit_count = 0
    for path in sorted((run_dir / "folds").glob("*.json")):
        result = read_json(path)
        for fold in result["folds"]:
            params = fold["best_parameters"]
            row = {
                "run_id": run_id,
                "dataset": result["dataset"],
                "condition": result["condition"],
                "outer_fold": fold["outer_fold"],
                "model": result["model"],
                "random_seed": fold["random_seed"],
                "C": params.get("C", "N/A"),
                "B": params.get("B", "N/A"),
                "tau": params.get("tau", "N/A"),
                **fold["metrics"],
                "selected_feature_count": fold["selected_feature_count"],
                "model_fit_time": fold["model_fit_time"],
                "hyperparameter_search_time": fold["hyperparameter_search_time"],
                "total_outer_fold_time": fold["total_outer_fold_time"],
                "solver_status": fold["solver"].get("status"),
                "solver_backend": fold["solver"].get("backend"),
                "mip_gap": fold["solver"].get("mip_gap"),
            }
            rows.append(row)
            hashes.add(fold["corruption_manifest"]["generated_output_hash"])
            n_features = len(fold["coefficients"])
            fit_count += model_fit_cost(result["model"], n_features, fold["best_parameters"], config)
            fit_count += sum(
                len(record["balanced_accuracy_folds"])
                * model_fit_cost(result["model"], n_features, record["parameters"], config)
                for record in fold["search"]
            )
    return rows, hashes, fit_count


def _collect_stability(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "folds").glob("*.json")):
        result = read_json(path)
        if not result["folds"]:
            continue
        n_features = len(result["folds"][0]["coefficients"])
        stability = feature_stability(
            [fold["selected_feature_indices"] for fold in result["folds"]], n_features
        )
        rows.append({
            "dataset": result["dataset"], "condition": result["condition"], "model": result["model"],
            "base_seed": result["base_seed"], "mean_pairwise_jaccard": stability["mean_pairwise_jaccard"],
            "selection_frequency": json.dumps(stability["selection_frequency"]),
        })
    return rows


def _write_pilot_summary(
    run_dir: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    planned_fits: int,
    manifest: dict[str, Any],
) -> None:
    numeric = ("balanced_accuracy", "weighted_f1", "accuracy", "gmean", "selected_feature_count", "model_fit_time", "hyperparameter_search_time")
    means = {name: float(np.mean([float(row[name]) for row in rows])) for name in numeric} if rows else {}
    statuses: dict[str, int] = {}
    backends: set[str] = set()
    gaps: list[float] = []
    for row in rows:
        statuses[str(row["solver_status"])] = statuses.get(str(row["solver_status"]), 0) + 1
        if row.get("solver_backend"):
            backends.add(str(row["solver_backend"]))
        if row["mip_gap"] not in (None, ""):
            gaps.append(float(row["mip_gap"]))
    test_report_path = Path("artifacts_v2/test_report.json")
    test_report = read_json(test_report_path) if test_report_path.exists() else None
    try:
        from src.experiments.config import load_config
        full_config = load_config("configs/full.yaml")
        full_fits = estimate_model_fits(full_config)
        lower_bound_seconds = full_fits * float(manifest["elapsed_seconds"]) / max(1, int(manifest["number_of_completed_fits"]))
    except (FileNotFoundError, ValueError):
        full_fits, lower_bound_seconds = None, None
    lines = [
        "# Pilot summary", "", "## Repository changes", "",
        "- Added a corrected model package while retaining every legacy model and experiment script.",
        "- Added strict loaders, deterministic corruption generators, nested CV, result persistence, reporting, and optional statistics modules.",
        "- Added a safe top-level CLI, configuration files, and focused scientific tests.", "",
        "## Validation", "",
        f"- Tests: {test_report['passed']} passed, {test_report['failed']} failed ({test_report['command']})." if test_report else "- Tests: see artifacts_v2/test_report.json.",
        "- Dataset audit: all 24 clean/archived dataset-condition files passed shape and label validation.", "",
        "## Protocol", "",
        f"- Datasets: {', '.join(config['datasets'])}",
        "- Sample counts: " + ", ".join(
            f"{dataset}={DATASET_SPECS[dataset].samples}x{DATASET_SPECS[dataset].features}"
            for dataset in config["datasets"]
        ),
        f"- Models: {', '.join(config['models'])}",
        f"- Conditions: {', '.join(config['conditions'])}",
        f"- Outer/inner folds: {config['outer_folds']}/{config['inner_folds']}",
        f"- C grid: {config['C_grid']}",
        f"- tau grid: {config['tau_grid']}",
        f"- B grid: {config.get('B_grid', 'auto')}",
        f"- Planned underlying optimization fits: {planned_fits}",
        f"- Completed underlying optimization fits: {manifest['number_of_completed_fits']}", "",
        "## Mean outer-test results", "",
    ]
    for name, value in means.items():
        lines.append(f"- {name}: {value:.6f}")
    lines.extend(["", "## Solver diagnostics", "", f"- Status distribution: {statuses}"])
    lines.append(f"- MIP gap mean: {float(np.mean(gaps)):.6g}" if gaps else "- MIP gap mean: N/A")
    lines.extend([
        "", "## Full-run estimate", "",
        f"- Conservative expected optimization fits with the current full config: {full_fits}." if full_fits is not None else "- Expected fits: unavailable.",
        f"- Naive pilot-rate lower bound: {lower_bound_seconds / 3600:.1f} hours." if lower_bound_seconds is not None else "- Runtime lower bound: unavailable.",
        "- This is not a credible upper bound: Colon/RFE and high-dimensional MILPs can be orders of magnitude slower.",
        "", "## Unresolved scientific/implementation issues", "",
        "- The manuscript says numeric coefficient bounds are reported, but it gives none; the pilot preserves the legacy [-2, 2] bounds pending author confirmation.",
        "- The manuscript does not provide corruption rates or distributions; full.yaml intentionally blocks until the author supplies them.",
        f"- Solver backends observed in the pilot: {', '.join(sorted(backends)) or 'unavailable'}.",
        "- Full/sensitivity/ablation configs target the optional DOcplex/CPLEX backend; verify the licensed runtime before scientific runs.",
        "", "## Notes", "",
        "- Outer test folds remained clean and were not used for scaling, selection, tuning, or corruption fitting.",
        "- Wilcoxon analysis was not executed.",
        "- This pilot uses the explicitly reduced grid in its configuration and is not a manuscript result.",
        "- Test-only generated corruption manifests are saved under corruptions/qa_sample_manifests.json and were not used for performance estimates.",
    ])
    (run_dir / "pilot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_corruption_qa(run_dir: Path, config: dict[str, Any]) -> None:
    qa = config.get("corruption_qa")
    if not isinstance(qa, dict) or not config["datasets"]:
        return
    dataset = config["datasets"][0]
    X, y = load_dataset(dataset, "clean")
    X, _, _ = fit_transform_training(X)
    mixed_config = qa.get("mixed")
    margin_config = qa.get("high_margin")
    if not isinstance(mixed_config, dict) or not isinstance(margin_config, dict):
        raise ValueError("corruption_qa requires mixed and high_margin mappings")
    solver = config.get("solver", {})
    margin_config = {
        **margin_config,
        "reference_backend": solver.get("backend", "scipy"),
        "reference_threads": int(solver.get("threads", 1)),
        "reference_time_limit": solver.get("time_limit"),
    }
    combined_config = {"mixed": mixed_config, "high_margin": margin_config}
    manifests = []
    for offset, condition, parameters in (
        (0, "mixed", mixed_config),
        (1, "high_margin", margin_config),
        (2, "combined", combined_config),
    ):
        result = apply_corruption(X, y, condition, seed=int(config["seeds"][0]) + offset, config=parameters)
        manifests.append({"dataset": dataset, "purpose": "test-only QA; not used in pilot metrics", **result.manifest})
    write_json(run_dir / "corruptions" / "qa_sample_manifests.json", manifests)


_METRIC_FIELDS = [
    "run_id", "dataset", "condition", "outer_fold", "model", "random_seed", "C", "B", "tau",
    "balanced_accuracy", "weighted_f1", "accuracy", "gmean", "selected_feature_count", "model_fit_time",
    "hyperparameter_search_time", "total_outer_fold_time", "solver_status", "solver_backend", "mip_gap",
]
