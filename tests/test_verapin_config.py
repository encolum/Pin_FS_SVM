import copy

import pytest

from src.experiments.config import load_config
from src.experiments.verapin import validate_verapin_config


def test_distributed_verapin_configs_preserve_author_decision_gates():
    config = load_config("configs/static_ks_pilot.yaml")
    with pytest.raises(ValueError, match="unresolved author decisions") as error:
        validate_verapin_config(config, command="kernel-search")
    assert "solver.total_time_limit" in str(error.value)
    assert "instances[0].n_samples" in str(error.value)


def test_tiny_explicit_static_config_passes_validation():
    config = load_config("configs/static_ks_pilot.yaml")
    instance = config["instances"][0]
    instance.update(
        {
            "n_samples": 20,
            "informative_ratio": 0.2,
            "redundant_ratio": 0.2,
            "correlation_strength": 0.9,
            "positive_class_fraction": 0.5,
            "label_noise_rate": 0.0,
            "outlier_sample_rate": 0.0,
            "outlier_feature_rate": 0.0,
            "outlier_scale": 0.0,
            "feature_budget_ratio": 0.1,
            "seed": 1,
        }
    )
    config["problem"] = {
        "C": 1.0,
        "tau": 0.5,
        "coefficient_bounds": {"lower": -3.0, "upper": 3.0},
    }
    config["solver"].update(
        {"backend": "scipy", "total_time_limit": 2.0, "subproblem_time_limit": 0.5, "mip_gap": 0.0}
    )
    config["search"].update({"max_iterations": 2, "final_refinement_fraction": 0.2})
    config["static_policy"].update({"initial_kernel_size": 10, "bucket_size": 10})
    validate_verapin_config(config, command="kernel-search")


def test_evolution_config_rejects_any_heldout_test_instance():
    config = load_config("configs/verapin_evolution.yaml")
    # Resolve only enough to ensure the split guard itself remains independently testable.
    extra = copy.deepcopy(config["instances"][0])
    extra["id"] = "forbidden-test"
    extra["split"] = "test"
    config["instances"].append(extra)
    with pytest.raises(ValueError, match="held-out test"):
        validate_verapin_config(config, command="evolve-verapin")


def test_final_config_keeps_outer_classification_choices_author_gated():
    config = load_config("configs/verapin_final.yaml")
    with pytest.raises(ValueError, match="unresolved author decisions") as error:
        validate_verapin_config(config, command="evaluate-verapin")
    assert "classification.outer_folds" in str(error.value)
    assert "classification.outer_seed" in str(error.value)
