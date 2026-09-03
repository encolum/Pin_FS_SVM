import numpy as np
import pytest
from scipy import sparse

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


@pytest.mark.parametrize("as_sparse", [False, True])
def test_mixed_masks_disjoint_and_effective_counts_match_actual_changes(as_sparse):
    X = np.array([[0., 2., 0., 4.], [5., 0., 7., 0.], [0., 8., 9., 0.], [3., 0., 0., 1.]])
    y = np.array([-1, 1, -1, 1])
    original = sparse.csr_matrix(X) if as_sparse else X
    result = apply_corruption(original, y, "mixed", seed=8, config={"label_flip_rate": .25,
        "additive_rate": .5, "multiplicative_rate": .5, "additive_std": 1.,
        "multiplicative_std": .5, "max_modified_cells": 8})
    metadata = result.manifest
    masks = [set(map(tuple, metadata[key])) for key in ("additive_cells", "multiplicative_cells")]
    assert masks[0].isdisjoint(masks[1])
    changed = result.X.toarray() if as_sparse else result.X
    eligible = np.count_nonzero(X) if as_sparse else X.size
    assert metadata["eligible_feature_cells"] == eligible
    if as_sparse:
        assert all(X[row, col] != 0 for row, col in masks[0] | masks[1])
        assert np.array_equal(changed[X == 0], X[X == 0])
        assert sparse.isspmatrix_csr(result.X)
        assert result.X.nnz == original.nnz
    for name in ("additive", "multiplicative"):
        effective = set(map(tuple, metadata[f"effective_{name}_cells"]))
        assert metadata[f"{name}_changed_count"] == len(effective)
        assert metadata[f"{name}_effective_rate"] == len(effective) / eligible
        assert all(X[row, col] != changed[row, col] for row, col in effective)
    assert metadata["additive_changed_count"] + metadata["multiplicative_changed_count"] == np.count_nonzero(X != changed)
    assert np.count_nonzero(result.y != y) == metadata["label_changed_count"] == 1


def test_sparse_explicit_zeros_are_not_eligible_and_rounding_keeps_masks_disjoint():
    X = sparse.csr_matrix((np.array([0., 2., 3., 4.]), np.arange(4), [0, 4, 4]), shape=(2, 4))
    result = apply_corruption(X, np.array([-1, 1]), "mixed", seed=2, config={
        "label_flip_rate": 0., "additive_rate": .5, "multiplicative_rate": .5,
        "additive_std": 1., "multiplicative_std": 1., "max_modified_cells": 3})
    assert result.manifest["eligible_feature_cells"] == 3
    assert result.manifest["additive_selected_count"] == 2
    assert result.manifest["multiplicative_selected_count"] == 1
    assert result.X[0, 0] == 0.


@pytest.mark.parametrize("as_sparse", [False, True])
def test_zero_severity_records_no_effective_corruption(as_sparse):
    X = np.eye(6)
    X = sparse.csr_matrix(X) if as_sparse else X
    result = apply_corruption(X, np.tile([-1, 1], 3), "mixed", seed=3, config={
        "label_flip_rate": 0., "additive_rate": .5, "multiplicative_rate": .5,
        "additive_std": 0., "multiplicative_std": 0., "max_modified_cells": 6})
    assert result.manifest["additive_changed_count"] == result.manifest["multiplicative_changed_count"] == 0
    assert result.manifest["modified_sample_indices"] == result.manifest["modified_feature_indices"] == []
    assert result.manifest["generated_output_hash"] == result.manifest["input_hash"]


def test_all_zero_sparse_feature_noise_is_audited_as_ineffective():
    result = apply_corruption(sparse.csr_matrix((4, 100000)), np.array([-1, 1, -1, 1]), "mixed", seed=3,
        config={"label_flip_rate": .25, "additive_rate": .5, "multiplicative_rate": .5,
                "additive_std": 1., "multiplicative_std": 1.})
    assert result.X.nnz == result.manifest["eligible_feature_cells"] == 0
    assert result.manifest["additive_effective_rate"] == result.manifest["multiplicative_effective_rate"] == 0.
    assert result.manifest["label_effective_rate"] == .25


def test_overlapping_rate_request_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        apply_corruption(np.eye(2), np.array([-1, 1]), "mixed", seed=3, config={
            "label_flip_rate": 0., "additive_rate": .6, "multiplicative_rate": .5,
            "additive_std": 1., "multiplicative_std": 1.})


@pytest.mark.parametrize("as_sparse", [False, True])
def test_label_noise_never_changes_features_or_requires_densification(as_sparse):
    X = np.eye(10)
    X = sparse.csr_matrix(X) if as_sparse else X
    y = np.tile([-1, 1], 5)
    result = apply_corruption(X, y, "label_noise", seed=7, config={"label_flip_rate": .3})
    assert result.manifest["condition"] == "label_noise"
    assert np.count_nonzero(result.y != y) == 3
    assert (result.X != X).nnz == 0 if as_sparse else np.array_equal(result.X, X)
    assert result.manifest["modified_feature_indices"] == []


@pytest.mark.parametrize("as_sparse", [False, True])
def test_outliers_use_training_feature_scales_and_preserve_sparse_zeros(as_sparse):
    X = np.array([[0., 2., 5.], [1., 0., 5.], [2., 4., 5.], [3., 8., 5.]])
    y = np.array([-1, 1, -1, 1])
    multiplier = np.array([1., 100., 1.])
    def transform(data):
        return sparse.csr_matrix(data) if as_sparse else data
    profile = {"sample_rate": 1., "feature_rate": 1., "scale": 3., "max_modified_cells": 12}
    first = apply_corruption(transform(X), y, "feature_outlier", seed=9, config=profile)
    second = apply_corruption(transform(X * multiplier), y, "feature_outlier", seed=9, config=profile)
    actual = first.X.toarray() if as_sparse else first.X
    scaled = second.X.toarray() if as_sparse else second.X
    assert np.allclose(scaled - X * multiplier, (actual - X) * multiplier)
    assert np.allclose(first.manifest["feature_scale"], X.std(axis=0))
    assert np.array_equal(actual[:, 2], X[:, 2])  # Zero-variance columns unchanged.
    assert first.manifest["outlier_changed_count"] == np.count_nonzero(actual != X)
    if as_sparse:
        assert np.array_equal(actual[X == 0], X[X == 0])
        assert first.X.nnz == np.count_nonzero(X)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_combined_outlier_scale_is_frozen_before_mixed_noise(as_sparse):
    X, y = synthetic_data()
    original = sparse.csr_matrix(X) if as_sparse else X
    result = apply_corruption(original, y, "combined", seed=19, config={
        "mixed": {"label_flip_rate": 0., "additive_rate": 1., "multiplicative_rate": 0.,
                  "additive_std": 100., "multiplicative_std": 0., "max_modified_cells": 40},
        "feature_outlier": {"sample_rate": 1., "feature_rate": 1., "scale": 2., "max_modified_cells": 40}})
    assert np.allclose(result.manifest["stages"][1]["feature_scale"], X.std(axis=0))


def test_sparse_feature_corruption_never_densifies(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("sparse noise attempted densification")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    X = sparse.eye(20, format="csr")
    apply_corruption(X, np.tile([-1, 1], 10), "feature_outlier", seed=1,
        config={"sample_rate": .5, "feature_rate": .5, "scale": 3., "max_modified_cells": 20})


def test_dense_fortran_order_input_is_actually_corrupted_without_modifying_original():
    X = np.asfortranarray(np.arange(1., 21.).reshape(10, 2))
    original = X.copy()
    result = apply_corruption(X, np.tile([-1, 1], 5), "mixed", seed=8, config={
        "label_flip_rate": 0., "additive_rate": 1., "multiplicative_rate": 0.,
        "additive_std": 1., "multiplicative_std": 0.})
    assert result.manifest["additive_changed_count"] == X.size
    assert np.array_equal(X, original)
