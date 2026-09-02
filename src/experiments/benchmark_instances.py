"""Prepare benchmark/clean-synthetic instances before any solver-route comparison.

Nested tuning receives outer-training observations only. Scaling and corruption
are fitted/generated independently inside each inner/outer training partition.
"""

from itertools import product
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.benchmark_adapter import load_solver_ready_benchmark
from src.data.benchmark_registry import DEFAULT_REGISTRY_PATH
from src.data.corruptions import apply_corruption, array_hash, validate_corruption_profile
from src.data.preprocessing import fit_preprocessor, transform_partition
from src.data.synthetic import generate_clean_synthetic_instance, save_synthetic_instance
from src.evaluation.metrics import classification_metrics
from src.experiments.selection import selection_tie_key
from src.search.llm_evolution.evaluator import PolicyInstance
from src.search.restricted_solver import solve_restricted_pin_fs
from src.utils.matrices import matrix_metadata
from src.utils.serialization import write_json


CLEAN_SYNTHETIC_FIELDS = {"n_samples", "n_features", "informative_ratio", "redundant_ratio",
    "correlation_strength", "positive_class_fraction", "feature_budget_ratio", "seed"}


def research_split(spec):
    if "split" in spec and "research_split" in spec and spec["split"] != spec["research_split"]:
        raise ValueError("conflicting split and research_split")
    return spec.get("research_split", spec.get("split"))


def corruption_choices(config, spec):
    condition = spec.get("condition", "clean")
    if condition == "clean":
        return [(0, {})]
    section = config.get("corruption", {})
    seeds = [spec["corruption_seed"]] if "corruption_seed" in spec else section.get("seeds")
    profile = section.get("profiles", {}).get(condition)
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("non-clean instances require explicit nonnegative corruption seeds")
    if len(seeds) != len(set(seeds)) or not isinstance(profile, dict):
        raise ValueError("corruption needs unique seeds and an explicit profile mapping")
    if condition == "combined" and set(profile) != {"mixed", "feature_outlier"}:
        raise ValueError("scientific combined profile requires mixed + feature_outlier")
    if condition == "high_margin":
        raise ValueError("use explicit high_margin_label_attack for the optional legacy analysis")
    validate_corruption_profile(condition, profile)
    return [(seed, profile) for seed in seeds]


def prepare_partitions(X_train, y_train, X_test, y_test, *, preprocessing, condition, seed, corruption):
    before = None if X_test is None else array_hash(X_test, y_test)
    fitted = fit_preprocessor(X_train, **preprocessing)
    train = transform_partition(fitted, X_train)
    test = None if X_test is None else transform_partition(fitted, X_test)
    clean_test = None if test is None else array_hash(test, y_test)
    corrupted = apply_corruption(train, y_train, condition, seed=seed, config=corruption)
    if set(np.unique(corrupted.y)) != {-1, 1}:
        raise ValueError("training corruption removed one binary class")
    if test is not None and (array_hash(X_test, y_test) != before or array_hash(test, y_test) != clean_test):
        raise RuntimeError("test observations changed during training preparation")
    return corrupted.X, corrupted.y, test, {
        "preprocessing_policy": fitted.name, "preprocessing_parameters": fitted.metadata,
        "corruption_manifest": corrupted.manifest, "raw_test_hash": before,
        "untouched_preprocessed_test_hash": clean_test, "training_hash": array_hash(corrupted.X, corrupted.y),
        "test_unchanged_by_corruption": True, "densified": fitted.metadata["densified"],
    }


def _folds(X, y, count, seed):
    if type(count) is not int or count < 2 or min(np.unique(y, return_counts=True)[1]) < count:
        raise ValueError("each class must contain at least the requested number of stratified folds")
    return list(StratifiedKFold(n_splits=count, shuffle=True, random_state=seed).split(X, y))


def select_inner_parameters(X_train, y_train, *, classification, preprocessing, condition, seed,
                            corruption, coefficient_bounds, sample_ids):
    """Exact/reduced solver tuning on outer train only, never ADKS or VeraPin."""
    grid = classification["parameter_grid"]
    tuning = classification["tuning_solver"]
    selection_tolerance = float(classification.get("selection_tolerance", 1e-12))
    if not np.isfinite(selection_tolerance) or selection_tolerance < 0:
        raise ValueError("selection_tolerance must be finite and non-negative")
    folds = _folds(X_train, y_train, classification["inner_folds"], classification["inner_seed"])
    prepared = []
    fold_records = []
    for fold, (train, validation) in enumerate(folds, 1):
        X, y, X_validation, metadata = prepare_partitions(X_train[train], y_train[train],
            X_train[validation], y_train[validation], preprocessing=preprocessing,
            condition=condition, seed=seed, corruption=corruption)
        prepared.append((X, y, X_validation, y_train[validation]))
        fold_records.append({"inner_fold": fold, "train_sample_ids": sample_ids[train].tolist(),
                             "validation_sample_ids": sample_ids[validation].tolist(), **metadata})
    records = []
    for B, C, tau in product(grid["B"], grid["C"], grid["tau"]):
        parameters = {"B": int(B), "C": float(C), "tau": float(tau)}
        scores, selected_counts, errors, fold_results = [], [], [], []
        for fold, (X, y, X_validation, y_validation) in enumerate(prepared, 1):
            try:
                result = solve_restricted_pin_fs(X, y, kernel=set(range(X.shape[1])),
                    **parameters, coefficient_bounds=coefficient_bounds, backend=tuning["backend"],
                    time_limit=tuning["time_limit"], mip_gap=tuning["mip_gap"], threads=tuning["threads"],
                    deadline=perf_counter() + tuning["time_limit"])
                prediction = np.where(X_validation @ result.coefficients + result.intercept >= 0, 1, -1)
                scores.append(classification_metrics(y_validation, prediction)["balanced_accuracy"])
                selected_counts.append(int(np.count_nonzero(np.abs(result.coefficients) > 1e-3)))
                fold_results.append({"inner_fold": fold, "balanced_accuracy": scores[-1],
                                     "selected_feature_count": selected_counts[-1]})
            except (ValueError, RuntimeError) as exc:
                errors.append({"inner_fold": fold, "error": str(exc)})
                fold_results.append({"inner_fold": fold, "balanced_accuracy": None,
                                     "selected_feature_count": None, "error": str(exc)})
        records.append({"parameters": parameters, "balanced_accuracy_by_fold": scores,
                        "selected_feature_count_folds": selected_counts,
                        "mean_selected_feature_count": float(np.mean(selected_counts)) if not errors else None,
                        "fold_results": fold_results,
                        "mean_balanced_accuracy": float(np.mean(scores)) if not errors else None,
                        "errors": errors})
    valid = [record for record in records if record["mean_balanced_accuracy"] is not None]
    if not valid:
        raise RuntimeError("all inner-tuning candidates failed; cannot select scientific parameters")
    best = valid[0]
    for row in valid[1:]:
        score, best_score = row["mean_balanced_accuracy"], best["mean_balanced_accuracy"]
        if score > best_score + selection_tolerance or (
            abs(score - best_score) <= selection_tolerance
            and selection_tie_key(row["parameters"], row["mean_selected_feature_count"])
            < selection_tie_key(best["parameters"], best["mean_selected_feature_count"])
        ):
            best = row
    return best["parameters"], {"selection": "inner_balanced_accuracy", "test_data_used": False,
        "route": "full_pin_fs_only", "folds": fold_records, "candidates": records,
        "selected_parameters": best["parameters"], "solver": dict(tuning),
        "selection_tolerance": selection_tolerance,
        "tie_break": ["mean_selected_feature_count", "B", "parameter_order"],
        "active_feature_threshold": 1e-3}


def build_prepared_instances(config, spec, *, run_dir, outer_evaluation=False):
    split = research_split(spec)
    base_id = spec["id"]
    holdout = None
    if spec["kind"] == "benchmark":
        data = load_solver_ready_benchmark(spec["dataset"], data_root=spec.get("data_root"),
            partition_policy=spec["source_partition_policy"],
            registry_path=config.get("benchmark_registry", DEFAULT_REGISTRY_PATH))
        X, y, ids, holdout = data.X, data.y, data.sample_ids, data.holdout
        B = spec["feature_budget"]
        default_policy = data.preprocessing_policy
        source = {"dataset": data.dataset, "source_files": list(data.source_files),
                  "source_hashes": {row["path"]: row["sha256"] for row in data.source_files},
                  "source_partition_policy": data.partition_policy, "label_mapping": data.label_mapping,
                  "source_partitions": list(data.source_partitions), "warnings": list(data.warnings),
                  "adapter_metadata": data.metadata}
    else:
        generated = generate_clean_synthetic_instance(**{name: spec[name] for name in CLEAN_SYNTHETIC_FIELDS},
                                                       research_split=split)
        save_synthetic_instance(generated, Path(run_dir) / "instances", instance_id=base_id)
        X, y, B = generated.X, generated.y, generated.feature_budget
        ids = np.array([f"{base_id}:pool:{index}" for index in range(len(y))])
        default_policy = "standard"
        source = {"dataset": "synthetic", "source_partition_policy": "pool",
                  "label_mapping": {"-1": -1, "1": 1}, **generated.metadata()}
    if type(B) is not int or not 1 <= B <= X.shape[1]:
        raise ValueError("feature_budget must be an integer between 1 and the feature count")
    preprocessing = {"policy": default_policy, **config.get("preprocessing", {}), **spec.get("preprocessing", {})}
    source.update(kind=spec["kind"], research_split=split,
                  preprocessing_overridden=preprocessing["policy"] != default_policy,
                  clean_source_hash=array_hash(X, y))
    problem = config["problem"]
    bounds = (problem["coefficient_bounds"]["lower"], problem["coefficient_bounds"]["upper"])
    classification = config.get("classification", {})
    if outer_evaluation and holdout is None:
        folds = _folds(X, y, classification["outer_folds"], classification["outer_seed"])
    else:
        folds = [(np.arange(len(y)), np.array([], dtype=int))]
    instances = []
    for outer_fold, (train, test) in enumerate(folds, 1):
        test_X = holdout.X if outer_evaluation and holdout is not None else X[test] if len(test) else None
        test_y = holdout.y if outer_evaluation and holdout is not None else y[test] if len(test) else None
        test_ids = holdout.sample_ids if outer_evaluation and holdout is not None else ids[test]
        for seed, profile in corruption_choices(config, spec):
            condition = spec.get("condition", "clean")
            parameters = {"B": int(B), "C": float(problem["C"]), "tau": float(problem["tau"])}
            tuning = {"selection": "fixed_config", "selected_parameters": parameters}
            if outer_evaluation and "parameter_grid" in classification:
                parameters, tuning = select_inner_parameters(X[train], y[train], classification=classification,
                    preprocessing=preprocessing, condition=condition, seed=seed, corruption=profile,
                    coefficient_bounds=bounds, sample_ids=ids[train])
            train_X, train_y, prepared_test, metadata = prepare_partitions(X[train], y[train], test_X, test_y,
                preprocessing=preprocessing, condition=condition, seed=seed, corruption=profile)
            instance_id = base_id + (f":outer-{outer_fold}" if outer_evaluation else "")
            if condition != "clean":
                instance_id += f":seed-{seed}"
            metadata.update({**source, **matrix_metadata(train_X)})
            metadata.update(condition=condition,
                train_indices=train.tolist(), test_indices=(list(range(len(test_y)))
                    if outer_evaluation and holdout is not None else test.tolist()),
                train_sample_ids=ids[train].tolist(), test_sample_ids=test_ids.tolist(),
                outer_fold=outer_fold if outer_evaluation else None,
                evaluation_protocol="official_holdout" if holdout is not None else "outer_cv" if outer_evaluation else "solver_only",
                inner_tuning=tuning, **parameters, coefficient_bounds=problem["coefficient_bounds"],
                holdout_reserved=holdout is not None, all_routes_share_training_data=True)
            write_json(Path(run_dir) / "instances" / f"{instance_id.replace(':', '-')}.json",
                       {"instance_id": instance_id, **metadata})
            instances.append(PolicyInstance(instance_id=instance_id, split=split, X=train_X, y=train_y,
                **parameters, coefficient_bounds=bounds, X_test=prepared_test, y_test=test_y,
                base_instance_id=base_id, outer_fold=outer_fold if outer_evaluation else None, metadata=metadata))
    return instances


def assert_research_groups_disjoint(instances):
    """Reject reused source observations across evolution train/validation/test."""
    for index, first in enumerate(instances):
        for second in instances[index + 1:]:
            if first.research_split == second.research_split or not first.metadata or not second.metadata:
                continue
            first_hash, second_hash = first.metadata.get("clean_source_hash"), second.metadata.get("clean_source_hash")
            same_source = first_hash is not None and first_hash == second_hash
            shared_ids = set(first.metadata.get("train_sample_ids", [])) & set(second.metadata.get("train_sample_ids", []))
            if same_source or shared_ids:
                raise ValueError("research train/validation/test groups reuse source observations")
