from src.experiments.config import load_config
from src.experiments.search import (
    build_budget_grid,
    estimate_model_fits,
    estimate_top_level_fits,
    parameter_candidates,
)


def test_budget_grid_is_exhaustive_only_for_small_data():
    assert build_budget_grid(5, "auto", exhaustive_max=30) == [1, 2, 3, 4, 5]
    assert build_budget_grid(2000, "auto", exhaustive_max=30) == [1, 2, 3, 5, 10, 15, 25, 50, 100, 250, 500, 2000]


def test_explicit_budget_grid_is_filtered_and_deduplicated():
    assert build_budget_grid(10, [10, 2, 2, 99]) == [2, 10]


def test_fisher_threshold_is_an_explicit_inner_cv_hyperparameter():
    config = load_config("configs/pilot.yaml")
    candidates = parameter_candidates("fisher_l1_svm", 8, config)
    assert candidates == [
        {"C": 1.0, "threshold_percentile": 25},
        {"C": 1.0, "threshold_percentile": 50},
        {"C": 1.0, "threshold_percentile": 75},
    ]


def test_fit_estimate_includes_fisher_candidates_and_rfe_optimizations():
    config = load_config("configs/pilot.yaml")
    assert estimate_top_level_fits(config) == 50
    assert estimate_model_fits(config) == 86
