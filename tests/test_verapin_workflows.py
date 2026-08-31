import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.verapin import run_static_kernel_search, run_verapin_final
from src.utils.serialization import read_json, write_json


@pytest.mark.parametrize("references_fail", [False, True])
def test_evolution_workflow_prepares_references_before_provider(tmp_path, monkeypatch, references_fail):
    from src.experiments import verapin
    from src.search.llm_evolution import references
    from src.search.llm_evolution.schemas import PolicyCandidate

    config = {
        "instances": [_instance("train"), {**_instance("validation"), "seed": 11}],
        "problem": {"C": 1., "tau": .5, "coefficient_bounds": {"lower": -4., "upper": 4.}},
        "solver": {"backend": "scipy", "threads": 1, "total_time_limit": 3., "subproblem_time_limit": 1., "mip_gap": 0.},
        "search": _search(), "adks_policy": _adks(), "seed_policies": [_candidate()],
        "evolution": {"generations": 1, "population_size": 1, "parent_count": 1,
                      "candidates_per_generation": 1, "maximum_similarity": 1., "seed": 1},
        "fitness": {"weights": {"primal_integral": 1., "final_gap": 1., "failure_rate": 1., "overhead": 1.},
                    "normalization": {"primal_integral_scale": 3., "final_gap_scale": 1., "overhead_scale": 1.},
                    "target_gap": .01},
        "llm": {"provider": "mock", "responses": ["unused"]},
        "output": {"root": str(tmp_path / "runs")}, "frozen_policy_output": str(tmp_path / "frozen.json"),
    }
    if references_fail:
        monkeypatch.setattr(references, "_reference_route", lambda *a, **k: {"objective": None})
    events = []
    def provider(*args):
        assert not references_fail
        artifact, = (tmp_path / "runs").glob("*/fitness_references.json")
        assert read_json(artifact)["status"] == "complete"
        events.append("provider")
        return object()
    def evolve(**kwargs):
        for instance in kwargs["training_instances"] + kwargs["validation_instances"]:
            assert instance.reference_objective is not None and instance.fitness_horizon == 3.
        events.append("evolution")
        return SimpleNamespace(frozen_candidate=PolicyCandidate.from_dict(_candidate()))
    monkeypatch.setattr(verapin, "_provider", provider)
    monkeypatch.setattr(verapin, "run_evolution", evolve)  # No candidate generation or LLM call.
    monkeypatch.setattr(verapin, "_write_validation_baseline_comparison", lambda *a, **k: None)
    if references_fail:
        with pytest.raises(RuntimeError, match="evolution aborted"):
            verapin.run_verapin_evolution(config)
        assert not events
    else:
        output = verapin.run_verapin_evolution(config)
        assert events == ["provider", "evolution"]
        assert all(row["fitness_horizon"] == 3. for row in read_json(output / "manifest.json")["instances"])


def _instance(split):
    return {
        "id": f"tiny-{split}",
        "split": split,
        "n_samples": 20,
        "n_features": 8,
        "informative_ratio": 0.25,
        "redundant_ratio": 0.25,
        "correlation_strength": 0.9,
        "positive_class_fraction": 0.5,
        "label_noise_rate": 0.0,
        "outlier_sample_rate": 0.0,
        "outlier_feature_rate": 0.0,
        "outlier_scale": 0.0,
        "feature_budget_ratio": 0.25,
        "seed": 8,
    }


def _search():
    return {
        "max_iterations": 1,
        "final_full_refinement": False,
        "final_refinement_fraction": 0.0,
        "acceptance_epsilon": 1e-9,
        "seed": 3,
        "signal_options": {"use_lp": False},
    }


def _weights():
    return {
        "initial_fisher": 1.0,
        "initial_mutual_information": 1.0,
        "initial_lp_activation": 1.0,
        "keep_selected": 4.0,
        "keep_abs_coefficient": 2.0,
        "keep_selection_frequency": 1.0,
        "keep_slack_association": 1.0,
        "keep_lp_activation": 1.0,
        "keep_redundancy_penalty": 1.0,
        "keep_inactivity_penalty": 0.1,
        "keep_kernel_age_penalty": 0.1,
        "add_fisher": 1.0,
        "add_mutual_information": 1.0,
        "add_lp_activation": 1.0,
        "add_slack_association": 1.0,
        "add_nonredundancy": 1.0,
        "add_selection_stability": 1.0,
    }


def _adks():
    return {
        "initial_kernel_size": 2,
        "minimum_kernel_size": 2,
        "maximum_kernel_size": 8,
        "stagnation_threshold": 2,
        "focus_fraction": 0.25,
        "expansion_fraction": 0.5,
        "weights": _weights(),
    }


def _candidate():
    return {
        "schema_version": 1,
        "policy_id": "frozen-test",
        "name": "frozen-test",
        "initial_kernel_size": 2,
        "initial_score": {"feature": "fisher_score"},
        "add_score": {"feature": "mutual_information"},
        "keep_score": {"feature": "is_selected"},
        "target_kernel_size": {
            "op": "clip",
            "value": {"op": "add", "args": [{"search": "kernel_size"}, 1]},
            "lower": {"search": "feature_budget"},
            "upper": {"search": "total_features"},
        },
        "metadata": {"selected_on": "validation"},
    }


def test_static_workflow_persists_route_and_iteration_schemas(tmp_path):
    config = {
        "instances": [_instance("train")],
        "problem": {"C": 1.0, "tau": 0.5, "coefficient_bounds": {"lower": -4.0, "upper": 4.0}},
        "solver": {"backend": "scipy", "threads": 1, "total_time_limit": 2.0, "subproblem_time_limit": 0.5, "mip_gap": 0.0},
        "search": _search(),
        "static_policy": {"score_name": "fisher_score", "initial_kernel_size": 2, "bucket_size": 2, "maximum_kernel_size": 8},
        "output": {"root": str(tmp_path / "static")},
    }
    run_dir = run_static_kernel_search(config)
    assert (run_dir / "route_results.csv").is_file()
    assert (run_dir / "iteration_results.csv").is_file()
    assert (run_dir / "instances" / "tiny-train.json").is_file()
    with (run_dir / "route_results.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert "classification_scope" not in row
    assert "balanced_accuracy" not in row


def test_final_workflow_uses_three_routes_without_constructing_llm_provider(tmp_path, monkeypatch):
    pytest.importorskip("docplex")
    pytest.importorskip("cplex")
    policy_path = tmp_path / "frozen.json"
    write_json(policy_path, _candidate())

    def forbidden_provider(*args, **kwargs):
        raise AssertionError("held-out evaluation attempted to construct an LLM provider")

    monkeypatch.setattr("src.experiments.verapin.EnvironmentLLMProvider", forbidden_provider)
    config = {
        "instances": [_instance("test")],
        "problem": {"C": 1.0, "tau": 0.5, "coefficient_bounds": {"lower": -4.0, "upper": 4.0}},
        "solver": {"backend": "cplex", "threads": 1, "total_time_limit": 2.0, "subproblem_time_limit": 0.5, "mip_gap": 0.0},
        "search": _search(),
        "adks_policy": _adks(),
        "classification": {"outer_folds": 2, "outer_seed": 19},
        "frozen_policy_path": str(policy_path),
        "output": {"root": str(tmp_path / "final")},
    }
    run_dir = run_verapin_final(config)
    with (run_dir / "route_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    routes = {row["route"] for row in rows}
    assert routes == {"cold_cplex", "handcrafted_adks", "verapin_ks"}
    assert len(rows) == 6
    assert {row["outer_fold"] for row in rows} == {"1", "2"}
    assert {row["classification_scope"] for row in rows} == {"outer_test"}
    assert all(row["balanced_accuracy"] for row in rows)
    for fold in {"1", "2"}:
        fold_rows = [row for row in rows if row["outer_fold"] == fold]
        references = {row["primal_integral_reference_objective"] for row in fold_rows}
        assert len(references) == 1
        assert float(next(iter(references))) == pytest.approx(
            min(float(row["final_objective"]) for row in fold_rows)
        )
        assert next(row for row in fold_rows if row["route"] == "cold_cplex")[
            "final_gap"
        ]
        assert all(
            row["final_gap"] == ""
            for row in fold_rows
            if row["route"] != "cold_cplex"
        )
    for fold in (1, 2):
        fold_manifest = read_json(
            run_dir / "instances" / f"tiny-test-outer-{fold}.json"
        )
        train = set(fold_manifest["train_indices"])
        test = set(fold_manifest["test_indices"])
        assert train.isdisjoint(test)
        assert len(train | test) == 20
        assert fold_manifest["preprocessing"] == (
            "standard_scaler_fit_on_outer_train_only"
        )
