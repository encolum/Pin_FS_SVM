import pytest

from src.experiments.runner import run_experiment
from src.utils.serialization import write_json


def test_resume_rejects_a_different_configuration_before_loading_data(tmp_path):
    config = {
        "datasets": ["diabetes"],
        "models": ["l2_svm"],
        "conditions": ["clean"],
        "metrics": ["balanced_accuracy", "weighted_f1", "accuracy", "gmean"],
        "outer_folds": 2,
        "inner_folds": 2,
        "seeds": [42],
        "C_grid": [1.0],
        "tau_grid": [0.5],
        "B_grid": [2],
        "solver": {"backend": "scipy", "threads": 1},
        "corruption": {},
        "output": {"root": "results_v2"},
    }
    write_json(tmp_path / "config.yaml", {**config, "seeds": [99]})
    with pytest.raises(ValueError, match="does not exactly match"):
        run_experiment(config, mode="pilot", resume_dir=tmp_path)
