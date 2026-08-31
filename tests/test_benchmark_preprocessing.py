import numpy as np
import pytest
from scipy import sparse
from src.data.benchmark_adapter import load_solver_ready_benchmark
from src.data.preprocessing import fit_preprocessor, transform_partition, estimate_dense_bytes


@pytest.mark.parametrize("policy", ["standard_sparse", "max_abs", "none", "passthrough_upstream_normalized"])
def test_sparse_policies_keep_csr_and_fit_train_only(policy):
    train = sparse.csr_matrix([[0., 1], [2, 0], [0, 3]])
    test = sparse.csr_matrix([[1000., 900]])
    fitted = fit_preprocessor(train, policy=policy)
    parameters = repr(fitted.metadata["parameters"])
    assert sparse.isspmatrix_csr(transform_partition(fitted, train))
    assert sparse.isspmatrix_csr(transform_partition(fitted, test))
    assert repr(fitted.metadata["parameters"]) == parameters
    assert fitted.metadata["fit_samples"] == 3
    assert not fitted.metadata["densified"]
    if policy == "standard_sparse":
        assert fitted.transformer.with_mean is False
    if policy == "max_abs":
        np.testing.assert_array_equal(fitted.transformer.scale_, [2, 3])


def test_dense_training_only_standardization():
    fitted = fit_preprocessor(np.array([[0., 1.], [2, 3]]), policy="standard")
    transform_partition(fitted, np.array([[1e9, 1e9]]))
    assert fitted.metadata["parameters"]["mean_"] == [1., 2.]


def test_densification_is_opt_in_and_guarded_for_every_partition():
    X = sparse.csr_matrix([[0., 1], [2, 0]])
    for kwargs in ({}, {"allow_densify": True}, {"allow_densify": True, "max_dense_bytes": 31}):
        with pytest.raises(ValueError, match="densif"):
            fit_preprocessor(X, policy="standard", **kwargs)
    fitted = fit_preprocessor(X, policy="standard", allow_densify=True, max_dense_bytes=32)
    assert estimate_dense_bytes(X) == 32 and fitted.metadata["densified"]
    assert isinstance(transform_partition(fitted, X), np.ndarray)
    with pytest.raises(ValueError, match="max_dense_bytes"):
        transform_partition(fitted, sparse.vstack([X, X]))


@pytest.mark.parametrize("name", ["basehock", "colon"])
def test_real_sparse_benchmarks_preprocess_without_dense_conversion(name, monkeypatch):
    data = load_solver_ready_benchmark(name, partition_policy="pool")
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected densification")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    fitted = fit_preprocessor(data.X[:20], policy=data.preprocessing_policy)
    assert sparse.isspmatrix_csr(transform_partition(fitted, data.X[20:40]))
