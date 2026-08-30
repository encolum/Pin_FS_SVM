import numpy as np
import pytest

from src.models.corrected.pin_fs_svm import PinFSSVM
from src.search import build_pin_fs_problem, solve_restricted_pin_fs


@pytest.fixture
def small_problem():
    X = np.array([
        [-2.0, 0.1, 1.0],
        [-1.0, -0.2, 0.5],
        [-0.5, 0.2, -1.0],
        [0.5, -0.1, 1.0],
        [1.0, 0.2, -0.5],
        [2.0, -0.2, -1.0],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    return X, y


def _solve(X, y, *, kernel, backend="scipy"):
    return solve_restricted_pin_fs(
        X,
        y,
        kernel=set(kernel),
        B=1,
        C=5.0,
        tau=0.5,
        coefficient_bounds=(-5.0, 5.0),
        backend=backend,
        time_limit=10.0,
        mip_gap=0.0,
        threads=1,
        collect_progress=True,
    )


def test_full_kernel_reproduces_public_pin_fs_estimator(small_problem):
    X, y = small_problem
    model = PinFSSVM(
        B=1,
        C=5.0,
        tau=0.5,
        lower_bound=-5.0,
        upper_bound=5.0,
        backend="scipy",
        mip_gap=0.0,
    ).fit(X, y)
    restricted = _solve(X, y, kernel=range(X.shape[1]))

    assert restricted.objective == pytest.approx(
        model.solver_diagnostics()["objective_value"], abs=1e-7
    )
    assert restricted.coefficients == pytest.approx(model.w_, abs=1e-7)
    assert restricted.intercept == pytest.approx(model.b_, abs=1e-7)


def test_problem_builder_fixes_binary_bounds_outside_kernel(small_problem):
    X, y = small_problem
    problem = build_pin_fs_problem(
        X,
        y,
        B=1,
        C=5.0,
        tau=0.5,
        lower_bound=-5.0,
        upper_bound=5.0,
        allowed_features={0},
    )
    assert problem.upper_bounds[problem.v_slice].tolist() == [1.0, 0.0, 0.0]


def test_restricted_scipy_solve_keeps_outside_features_inactive(small_problem):
    X, y = small_problem
    result = _solve(X, y, kernel={0})

    assert result.kernel == {0}
    assert result.v[1:].tolist() == [0, 0]
    assert result.coefficients[1:] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert result.support.issubset({0})
    assert len(result.progress) == 1


def test_restricted_cplex_solve_honors_fixed_binary_bounds(small_problem):
    pytest.importorskip("docplex")
    pytest.importorskip("cplex")
    X, y = small_problem
    result = _solve(X, y, kernel={0}, backend="cplex")

    assert result.v[1:].tolist() == [0, 0]
    assert result.coefficients[1:] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert result.diagnostics.backend.startswith("docplex-cplex-")


@pytest.mark.parametrize("kernel", [{-1}, {3}, {0.5}])
def test_restricted_solver_rejects_invalid_kernel_indices(small_problem, kernel):
    X, y = small_problem
    with pytest.raises(ValueError, match="feature index|integers"):
        _solve(X, y, kernel=kernel)


def test_kernel_smaller_than_budget_requires_explicit_override(small_problem):
    X, y = small_problem
    with pytest.raises(ValueError, match="kernel has 1 features but B=2"):
        solve_restricted_pin_fs(
            X,
            y,
            kernel={0},
            B=2,
            C=5.0,
            tau=0.5,
            coefficient_bounds=(-5.0, 5.0),
            backend="scipy",
            time_limit=10.0,
            mip_gap=0.0,
            threads=1,
        )
