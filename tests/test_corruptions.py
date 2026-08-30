import numpy as np
import pytest

from src.data.corruptions import apply_corruption


def synthetic_data():
    X = np.column_stack([np.linspace(-2, 2, 20), np.linspace(1, -1, 20)])
    y = np.where(X[:, 0] >= 0, 1, -1)
    return X, y


def test_mixed_corruption_is_deterministic_and_auditable():
    X, y = synthetic_data()
    config = {
        "label_flip_rate": 0.1,
        "additive_rate": 0.1,
        "multiplicative_rate": 0.1,
        "additive_std": 0.2,
        "multiplicative_std": 0.1,
    }
    first = apply_corruption(X, y, "mixed", seed=17, config=config)
    second = apply_corruption(X, y, "mixed", seed=17, config=config)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.manifest == second.manifest
    assert first.manifest["input_hash"] != first.manifest["generated_output_hash"]
    assert len(first.manifest["flipped_label_indices"]) == 2


def test_missing_scientific_parameters_fail_loudly():
    X, y = synthetic_data()
    with pytest.raises(ValueError, match="missing explicit corruption parameters"):
        apply_corruption(X, y, "mixed", seed=1, config={})


def test_high_margin_uses_only_passed_partition_and_is_deterministic():
    X, y = synthetic_data()
    config = {"flip_rate": 0.2, "reference_C": 1.0}
    result = apply_corruption(X[5:15], y[5:15], "high_margin", seed=9, config=config)
    assert result.manifest["samples"] == 10
    assert len(result.manifest["flipped_label_indices"]) == 2
    assert max(result.manifest["flipped_label_indices"]) < 10
