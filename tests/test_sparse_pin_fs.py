import numpy as np
import pytest
from scipy import sparse
from src.models.base import validate_training_data
from src.models.pin_fs_svm import PinFSSVM
from src.search.restricted_solver import build_pin_fs_problem, solve_restricted_pin_fs
from src.search.kernel_engine import run_kernel_search
from src.search.policies.static_ks import StaticKSPolicy


def fixture():
    X = np.array([[-2, 0, 1], [-1, 1, 0], [1, 0, 0], [2, -1, 1]], dtype=float)
    return X, np.array([-1, -1, 1, 1])


@pytest.mark.parametrize("backend", ["scipy", "cplex"])
def test_sparse_full_kernel_equals_dense_pin_fs(backend, monkeypatch):
    if backend == "cplex":
        pytest.importorskip("cplex")
    X, y = fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected sparse densification")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    parameters = dict(B=2, C=1., tau=.5, coefficient_bounds=(-2., 2.), backend=backend,
                      time_limit=3., mip_gap=0., threads=1)
    dense = solve_restricted_pin_fs(X, y, kernel={0, 1, 2}, **parameters)
    actual = solve_restricted_pin_fs(sparse.csr_matrix(X), y, kernel={0, 1, 2}, **parameters)
    assert actual.objective == pytest.approx(dense.objective, abs=1e-7)
    assert actual.model_build_time >= 0
    built = build_pin_fs_problem(sparse.csr_matrix(X), y, B=2, C=1, tau=.5, lower_bound=-2, upper_bound=2)
    other = build_pin_fs_problem(X, y, B=2, C=1, tau=.5, lower_bound=-2, upper_bound=2)
    assert sparse.isspmatrix_csr(built.constraint_matrix)
    assert (built.constraint_matrix != other.constraint_matrix).nnz == 0


def test_sparse_kernel_engine_and_prediction():
    X, y = fixture()
    X = sparse.csr_matrix(X)
    model = PinFSSVM(B=2, C=1, tau=.5, lower_bound=-2, upper_bound=2, backend="scipy").fit(X, y)
    result = run_kernel_search(X, y, B=2, C=1, tau=.5, coefficient_bounds=(-2, 2),
        policy=StaticKSPolicy(score_name="fisher_score", initial_kernel_size=3, bucket_size=1),
        total_time_limit=3, subproblem_time_limit=2, max_iterations=1, backend="scipy", threads=1,
        final_full_refinement=False, final_refinement_fraction=0, seed=1,
        signal_options={"use_lp": False, "use_correlation": False, "use_mutual_information": False})
    assert result.best_result.objective == pytest.approx(model.diagnostics_.objective_value)
    np.testing.assert_array_equal(model.predict(X), y)
    assert "mutual_information" in result.metadata["skipped_signals"]


@pytest.mark.parametrize("labels", [[-1.2, -1, 1, 1], [-1, -1, 1, np.nan]])
def test_validation_does_not_truncate_invalid_labels(labels):
    with pytest.raises(ValueError):
        validate_training_data(sparse.csr_matrix(fixture()[0]), labels)
