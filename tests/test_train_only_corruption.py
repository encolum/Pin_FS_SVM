import numpy as np
import pytest
from scipy import sparse
from src.data.corruptions import apply_corruption, array_hash
from src.data.synthetic import generate_clean_synthetic_instance
from src.experiments.benchmark_instances import prepare_partitions


MIXED = {"label_flip_rate": .1, "additive_rate": .2, "multiplicative_rate": .2,
         "additive_std": .4, "multiplicative_std": .3, "max_modified_cells": 1000}
OUTLIER = {"sample_rate": .2, "feature_rate": .5, "scale": 3., "max_modified_cells": 1000}


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("condition,profile", [("clean", {}), ("mixed", MIXED),
    ("feature_outlier", OUTLIER), ("combined", {"mixed": MIXED, "feature_outlier": OUTLIER})])
def test_only_training_is_corrupted_and_masks_replay(as_sparse, condition, profile):
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.tile([-1, 1], 10)
    X_test, y_test = X[-4:].copy() * 100, y[-4:].copy()
    X = X[:16]
    if as_sparse:
        X, X_test = sparse.csr_matrix(X), sparse.csr_matrix(X_test)
    raw_test_hash = array_hash(X_test, y_test)
    policy = "max_abs" if as_sparse else "standard"
    kwargs = dict(preprocessing={"policy": policy}, condition=condition, seed=31, corruption=profile)
    first = prepare_partitions(X, y[:16], X_test, y_test, **kwargs)
    second = prepare_partitions(X, y[:16], X_test, y_test, **kwargs)
    assert first[3] == second[3]
    assert array_hash(first[0], first[1]) == array_hash(second[0], second[1])
    assert array_hash(X_test, y_test) == raw_test_hash
    assert first[3]["test_unchanged_by_corruption"]
    assert first[3]["training_hash"] == array_hash(first[0], first[1])
    clean = prepare_partitions(X, y[:16], X_test, y_test,
        preprocessing={"policy": policy}, condition="clean", seed=31, corruption={})
    assert array_hash(first[2], y_test) == array_hash(clean[2], y_test)
    if condition != "clean":
        assert first[3]["training_hash"] != clean[3]["training_hash"]
    assert sparse.isspmatrix_csr(first[0]) == as_sparse


def test_sparse_feature_noise_requires_a_memory_cap():
    with pytest.raises(ValueError, match="max_modified_cells"):
        apply_corruption(sparse.eye(20, format="csr"), np.tile([-1, 1], 10), "feature_outlier",
                         seed=1, config={key: value for key, value in OUTLIER.items() if key != "max_modified_cells"})


def test_clean_synthetic_has_no_embedded_corruption_and_reproduces():
    kwargs = dict(n_samples=24, n_features=8, informative_ratio=.25, redundant_ratio=.25,
        correlation_strength=.8, positive_class_fraction=.5, feature_budget_ratio=.25,
        seed=7, research_split="test")
    first = generate_clean_synthetic_instance(**kwargs)
    second = generate_clean_synthetic_instance(**kwargs)
    assert first.data_hash == second.data_hash
    assert first.generation_mode == "clean_base"
    assert first.parameters.label_noise_rate == first.parameters.outlier_sample_rate == 0
    assert first.metadata()["research_split"] == "test"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_invalid_corruption_severity_rejected(bad):
    with pytest.raises(ValueError):
        apply_corruption(np.eye(6), np.tile([-1, 1], 3), "feature_outlier", seed=1,
                         config={**OUTLIER, "scale": bad})
