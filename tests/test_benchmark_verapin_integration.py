from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse

from src.utils.config import load_config
from src.experiments.verapin import _policy_instances, validate_verapin_config
from src.experiments import benchmark_instances as preparation
from src.experiments.readiness import check_execution_readiness
from src.data.corruptions import array_hash
from src.utils.serialization import read_json


@pytest.mark.parametrize("index", range(6))
def test_each_real_benchmark_builds_policy_instance_without_solving(index, tmp_path):
    config = load_config("configs/hardness_real_pilot.yaml")
    config["instances"] = [config["instances"][index]]
    validate_verapin_config(config, command="hardness")
    instance, = _policy_instances(config, run_dir=tmp_path)
    metadata = instance.metadata
    assert set(instance.y) == {-1, 1}
    assert instance.research_split == "test" and instance.outer_fold is None
    assert instance.X_test is None
    assert metadata["source_hashes"] and metadata["label_mapping"]
    assert metadata["preprocessing_parameters"]["fit_samples"] == len(instance.y)
    assert metadata["training_hash"] == array_hash(instance.X, instance.y)
    assert metadata["corruption_manifest"]["condition"] == "clean"
    assert not metadata["densified"]
    assert len(instance.instance_hash) == 64
    if metadata["dataset"] in {"colon", "basehock"}:
        assert sparse.isspmatrix_csr(instance.X)


@pytest.mark.parametrize("dataset,n_train,n_test", [("hill_valley", 606, 606), ("madelon", 2000, 600)])
def test_official_holdout_instance_is_separate_and_train_scaled(dataset, n_train, n_test, tmp_path):
    config = load_config("configs/hardness_real_pilot.yaml")
    spec = next(spec for spec in config["instances"] if spec["dataset"] == dataset)
    spec["source_partition_policy"] = "official_holdout"
    config["instances"] = [spec]
    # Data preparation only, with fixed parameters; no scientific run/inner solver.
    instance, = _policy_instances(config, run_dir=tmp_path, outer_evaluation=True)
    assert instance.X.shape[0] == n_train and instance.X_test.shape[0] == n_test
    assert instance.metadata["preprocessing_parameters"]["fit_samples"] == n_train
    assert set(instance.metadata["train_sample_ids"]).isdisjoint(instance.metadata["test_sample_ids"])
    assert instance.metadata["test_unchanged_by_corruption"]
    assert instance.metadata["evaluation_protocol"] == "official_holdout"


def tiny_config():
    config = load_config("configs/hardness_synthetic_pilot.yaml")
    config["instances"][0].update(n_samples=24, n_features=6, informative_ratio=.5, redundant_ratio=0., feature_budget_ratio=.5)
    config["classification"] = {"outer_folds": 2, "inner_folds": 2, "outer_seed": 19, "inner_seed": 23,
        "parameter_grid": {"B": [1, 2], "C": [1.], "tau": [.5]},
        "tuning_solver": {"backend": "scipy", "time_limit": 1., "mip_gap": 0., "threads": 1}}
    return config


def test_nested_tuning_never_sees_outer_test_and_freezes_parameters(tmp_path, monkeypatch):
    calls = []
    def solve(X, y, **kwargs):
        calls.append((X.copy(), y.copy(), kwargs))
        return SimpleNamespace(coefficients=np.zeros(X.shape[1]), intercept=1.)
    monkeypatch.setattr(preparation, "solve_restricted_pin_fs", solve)
    instances = _policy_instances(tiny_config(), run_dir=tmp_path, outer_evaluation=True)
    assert len(instances) == 2 and len(calls) == 8
    for instance in instances:
        assert len(instance.y) == len(instance.y_test) == 12
        assert instance.B == 1  # deterministic smallest-budget tie break
        tuning = instance.metadata["inner_tuning"]
        assert tuning["selection"] == "inner_balanced_accuracy" and tuning["test_data_used"] is False
        outer_test = set(instance.metadata["test_sample_ids"])
        outer_train = set(instance.metadata["train_sample_ids"])
        for fold in tuning["folds"]:
            inner_train, validation = set(fold["train_sample_ids"]), set(fold["validation_sample_ids"])
            assert inner_train.isdisjoint(validation)
            assert (inner_train | validation) == outer_train
            assert (inner_train | validation).isdisjoint(outer_test)
            assert fold["preprocessing_parameters"]["fit_samples"] == 6
        assert instance.metadata["inner_tuning"]["selected_parameters"] == {"B": instance.B, "C": instance.C, "tau": instance.tau}
    assert all(len(y) == 6 for _, y, _ in calls)


@pytest.mark.parametrize("support_by_budget,grid,expected", [
    ({1: 1, 2: 0}, {"B": [1, 2], "C": [1.], "tau": [.5]}, {"B": 2, "C": 1., "tau": .5}),
    ({1: 0, 2: 0}, {"B": [2, 1], "C": [1.], "tau": [.5]}, {"B": 1, "C": 1., "tau": .5}),
    ({1: 0}, {"B": [1], "C": [10., 2.], "tau": [.7, .3]}, {"B": 1, "C": 2., "tau": .3}),
])
def test_nested_tie_break_matches_main_pipeline_sparsity_budget_and_parameter_order(
        tmp_path, monkeypatch, support_by_budget, grid, expected):
    def solve(X, y, **kwargs):
        weights = np.full(X.shape[1], 1e-3)  # Boundary is inactive, even if v_j = 1.
        weights[:support_by_budget[kwargs["B"]]] = 1.
        return SimpleNamespace(coefficients=weights, intercept=1000.)
    monkeypatch.setattr(preparation, "solve_restricted_pin_fs", solve)
    config = tiny_config()
    config["classification"]["parameter_grid"] = grid
    instances = _policy_instances(config, run_dir=tmp_path, outer_evaluation=True)
    for instance in instances:
        tuning = instance.metadata["inner_tuning"]
        assert tuning["selected_parameters"] == expected
        for candidate in tuning["candidates"]:
            expected_count = support_by_budget[candidate["parameters"]["B"]]
            assert candidate["selected_feature_count_folds"] == [expected_count] * 2
            assert candidate["mean_selected_feature_count"] == expected_count
            assert [fold["selected_feature_count"] for fold in candidate["fold_results"]] == [expected_count] * 2


def test_nested_tie_break_applies_score_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "solve_restricted_pin_fs", lambda X, y, **k:
        SimpleNamespace(coefficients=np.full(X.shape[1], 1. if k["B"] == 1 else 0.), intercept=1000.))
    scores = iter([.8, .8, .8 - 5e-13, .8 - 5e-13] * 2)
    monkeypatch.setattr(preparation, "classification_metrics", lambda *a: {"balanced_accuracy": next(scores)})
    instances = _policy_instances(tiny_config(), run_dir=tmp_path, outer_evaluation=True)
    assert all(instance.B == 2 for instance in instances)


def test_all_failed_inner_candidates_abort_instead_of_selecting_partial_scores(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("no incumbent")
    monkeypatch.setattr(preparation, "solve_restricted_pin_fs", fail)
    with pytest.raises(RuntimeError, match="all inner-tuning"):
        _policy_instances(tiny_config(), run_dir=tmp_path, outer_evaluation=True)


def test_real_hardness_default_cannot_launch_all_six():
    config = load_config("configs/hardness_real_pilot.yaml")
    validate_verapin_config(config, command="hardness")
    with pytest.raises(ValueError, match="--instance"):
        check_execution_readiness(config, "hardness")


def test_adks_pilot_requires_measured_hardness():
    config = load_config("configs/adks_real_pilot.yaml")
    config["instances"] = config["instances"][:1]
    validate_verapin_config(config, command="adks")
    with pytest.raises(ValueError, match="at least two"):
        check_execution_readiness(config, "adks")


def test_new_scientific_paths_do_not_silently_approve_bounds():
    config = load_config("configs/hardness_real_pilot.yaml")
    del config["execution"]
    with pytest.raises(ValueError, match="author approval"):
        validate_verapin_config(config, command="hardness")


def test_validate_only_does_not_start_solver_or_create_run(tmp_path, monkeypatch, capsys):
    import main
    from src.experiments import verapin
    def forbidden(*args, **kwargs):
        raise AssertionError("validation started a run")
    monkeypatch.setattr(verapin, "run_hardness_benchmark", forbidden)
    assert main.main(["hardness", "--config", "configs/hardness_real_pilot.yaml", "--validate-only"]) == 0
    assert "no run directory" in capsys.readouterr().out


def test_source_and_research_split_cannot_be_confused():
    config = load_config("configs/hardness_real_pilot.yaml")
    config["instances"][0]["split"] = "train"
    with pytest.raises(ValueError, match="conflicting"):
        validate_verapin_config(config, command="hardness")


def test_nested_final_three_routes_share_corrupted_input_and_never_call_llm(tmp_path, monkeypatch):
    import csv
    from src.experiments import verapin
    from src.utils.serialization import write_json
    pytest.importorskip("cplex")
    config = load_config("configs/adks_real_pilot.yaml")
    tiny = tiny_config()
    config.update(instances=tiny["instances"], classification=tiny["classification"],
                  execution={"purpose": "provisional_pilot", "parameters_provisional": True})
    config["instances"][0]["condition"] = "feature_outlier"
    config["corruption"] = {"seeds": [11], "profiles": {"feature_outlier": {
        "sample_rate": .2, "feature_rate": .5, "scale": .2}}}
    config["solver"].update(total_time_limit=3, subproblem_time_limit=1)
    config["search"].update(max_iterations=1, final_full_refinement=True)
    config["search"]["signal_options"].update(use_lp=False)
    config["adks_policy"].update(initial_kernel_size=2, minimum_kernel_size=2, maximum_kernel_size=6)
    policy = {"schema_version": 1, "policy_id": "fixture", "name": "fixture",
        "initial_kernel_size": 2, "initial_score": {"feature": "fisher_score"},
        "add_score": {"feature": "fisher_score"}, "keep_score": {"feature": "is_selected"},
        "target_kernel_size": {"search": "feature_budget"}, "metadata": {"selected_on": "validation"}}
    policy_path = tmp_path / "frozen.json"
    write_json(policy_path, policy)
    config["frozen_policy_path"] = str(policy_path)
    config["output"] = {"root": str(tmp_path / "runs")}
    observed = {}
    cold, engine = verapin._run_cold_impl, verapin._run_engine
    def observe(instance):
        digest = array_hash(instance.X, instance.y)
        assert digest == instance.metadata["training_hash"]
        observed.setdefault(instance.instance_id, []).append((digest, instance.B, instance.C, instance.tau))
    def spy_cold(instance, *args, **kwargs):
        observe(instance)
        return cold(instance, *args, **kwargs)
    def spy_engine(instance, *args, **kwargs):
        observe(instance)
        return engine(instance, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("final evaluation must never construct an LLM provider")
    monkeypatch.setattr(verapin, "_provider", forbidden)
    monkeypatch.setattr(verapin, "_run_cold_impl", spy_cold)
    monkeypatch.setattr(verapin, "_run_engine", spy_engine)
    run_dir = verapin.run_verapin_final(config)
    assert len(observed) == 2
    assert all(len(values) == 3 and len(set(values)) == 1 for values in observed.values())
    manifest = read_json(run_dir / "manifest.json")
    assert all(row["data_preparation"]["test_unchanged_by_corruption"] for row in manifest["instances"])
    with (run_dir / "route_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and {row["classification_scope"] for row in rows} == {"outer_test"}
    assert all(row["balanced_accuracy"] for row in rows)


def test_license_failure_is_reported_not_mislabeled_as_hardness(tmp_path, monkeypatch):
    from src.experiments import verapin
    from src.reporting.solver_profiles import summarize_hardness
    config = tiny_config()
    instance, = _policy_instances(config, run_dir=tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("CPLEX Error 1016: Community Edition")
    monkeypatch.setattr(verapin, "solve_restricted_pin_fs", fail)
    row, detail = verapin._run_cold(instance, config["solver"])
    assert row["solver_status"] == "license_limit"
    assert row["final_objective"] is None and detail["progress"] == []
    summary = summarize_hardness([row])
    assert summary["failed_instances"] == 1 and summary["nontrivial_instances"] == 0


def test_identical_clean_sources_cannot_cross_research_groups(tmp_path):
    config = tiny_config()
    first = config["instances"][0]
    first["research_split"] = "train"
    second = deepcopy(first)
    second.update(id="different-name-same-data", research_split="validation")
    config["instances"].append(second)
    with pytest.raises(ValueError, match="reuse source observations"):
        _policy_instances(config, run_dir=tmp_path)


def test_anytime_comparison_uses_shared_horizon_even_when_route_stops_early():
    from src.experiments.verapin import _apply_common_reference
    rows = [{"instance_id": "x", "route": "cold", "final_objective": 10.},
            {"instance_id": "x", "route": "adks", "final_objective": 20.}]
    def record(objective):
        return {"elapsed_seconds": 1., "incumbent_objective": objective, "best_bound": None,
                "relative_gap": None, "node_count": 0, "solution_count": 1}
    details = {"x:cold": {"total_runtime": 3., "time_budget": 10., "progress": [record(10.)]},
               "x:adks": {"total_runtime": 2., "time_budget": 10., "progress": [record(20.)]}}
    _apply_common_reference(rows, details)
    assert {row["primal_integral_horizon"] for row in rows} == {10.}
    assert rows[1]["primal_integral"] == pytest.approx(10.)
