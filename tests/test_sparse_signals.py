from time import perf_counter
import numpy as np
import pytest
from scipy import sparse
from src.search.signals import compute_static_signals, _support_redundancy, TimeBudgetExceeded


def test_sparse_and_dense_signals_agree_including_centering():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20, 9))
    X[np.abs(X) < .8] = 0
    X[:, 0] = 2
    y = np.tile([-1, 1], 10)
    options = dict(B=3, C=1, tau=.5, coefficient_bounds=(-2, 2), seed=1,
                   use_lp=False, use_mutual_information=False, correlation_chunk_size=2)
    dense = compute_static_signals(X, y, **options)
    csr = compute_static_signals(sparse.csr_matrix(X), y, **options)
    assert sparse.isspmatrix_csr(csr.standardized_X)
    for name in dense.values:
        np.testing.assert_allclose(dense.values[name], csr.values[name], atol=1e-9)
    support = np.array([1, 4])
    np.testing.assert_allclose(_support_redundancy(dense.standardized_X, support, chunk_size=2),
                               _support_redundancy(csr.standardized_X, support, chunk_size=2), atol=1e-9)


def test_correlation_converts_only_small_blocks(monkeypatch):
    original = sparse.csr_matrix.toarray
    def bounded(self, *args, **kwargs):
        assert max(self.shape) <= 2
        return original(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", bounded)
    original_csc = sparse.csc_matrix.toarray
    def bounded_csc(self, *args, **kwargs):
        assert max(self.shape) <= 2
        return original_csc(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", bounded_csc)
    data = sparse.random(16, 12, density=.2, random_state=1, format="csr")
    compute_static_signals(data, np.tile([-1, 1], 8), B=3, C=1, tau=.5,
        coefficient_bounds=(-2, 2), seed=1, use_lp=False, correlation_chunk_size=2)


def test_expired_signal_budget_stops_before_work():
    with pytest.raises(TimeBudgetExceeded):
        compute_static_signals(sparse.eye(4, format="csr"), np.array([-1, 1, -1, 1]),
            B=2, C=1, tau=.5, coefficient_bounds=(-2, 2), seed=1, deadline=perf_counter() - 1)
