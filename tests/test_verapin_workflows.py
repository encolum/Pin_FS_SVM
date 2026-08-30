import csv
from pathlib import Path

import pytest

from src.experiments.verapin import run_static_kernel_search, run_verapin_final
from src.utils.serialization import write_json


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
        "frozen_policy_path": str(policy_path),
        "output": {"root": str(tmp_path / "final")},
    }
    run_dir = run_verapin_final(config)
    with (run_dir / "route_results.csv").open(newline="", encoding="utf-8") as stream:
        routes = {row["route"] for row in csv.DictReader(stream)}
    assert routes == {"cold_cplex", "handcrafted_adks", "verapin_ks"}
