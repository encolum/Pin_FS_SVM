import numpy as np
import pytest

from src.search import (
    MIPStartData,
    build_pin_fs_problem,
    result_to_mip_start,
    solve_restricted_pin_fs,
    validate_mip_start,
)


@pytest.fixture
def problem_data():
    X = np.array([
        [-2.0, 0.1],
        [-1.0, -0.2],
        [-0.5, 0.2],
        [0.5, -0.1],
        [1.0, 0.2],
        [2.0, -0.2],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    problem = build_pin_fs_problem(
        X,
        y,
        B=1,
        C=5.0,
        tau=0.5,
        lower_bound=-5.0,
        upper_bound=5.0,
    )
    return X, y, problem


def _restricted_solution(X, y, *, backend="scipy", mip_start=None):
    return solve_restricted_pin_fs(
        X,
        y,
        kernel={0, 1} if mip_start is not None else {0},
        B=1,
        C=5.0,
        tau=0.5,
        coefficient_bounds=(-5.0, 5.0),
        backend=backend,
        time_limit=10.0,
        mip_gap=0.0,
        threads=1,
        mip_start=mip_start,
        collect_progress=True,
    )


def test_restricted_solution_becomes_feasible_full_model_start(problem_data):
    X, y, full_problem = problem_data
    restricted = _restricted_solution(X, y)
    mip_start = result_to_mip_start(restricted, full_problem)

    values = validate_mip_start(mip_start, full_problem, check_constraints=True)
    assert values.shape == full_problem.c.shape
    assert values[full_problem.v_slice].tolist() == [1.0, 0.0]


def test_malformed_mip_starts_fail_loudly(problem_data):
    _, _, problem = problem_data
    with pytest.raises(ValueError, match="length"):
        validate_mip_start(MIPStartData(np.zeros(problem.number_of_variables - 1)), problem)

    nonbinary = np.zeros(problem.number_of_variables)
    nonbinary[problem.v_slice.start] = 0.5
    with pytest.raises(ValueError, match="not binary"):
        validate_mip_start(MIPStartData(nonbinary), problem)

    fixed_problem = build_pin_fs_problem(
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
        np.array([-1, 1]),
        B=1,
        C=1.0,
        tau=0.5,
        lower_bound=-2.0,
        upper_bound=2.0,
        allowed_features={0},
    )
    outside_active = np.zeros(fixed_problem.number_of_variables)
    outside_active[fixed_problem.v_slice.start + 1] = 1.0
    with pytest.raises(ValueError, match="variable bounds"):
        validate_mip_start(MIPStartData(outside_active), fixed_problem)


def test_cplex_reports_accepted_and_rejected_mip_starts(problem_data):
    pytest.importorskip("docplex")
    pytest.importorskip("cplex")
    X, y, full_problem = problem_data
    restricted = _restricted_solution(X, y)
    valid_start = result_to_mip_start(restricted, full_problem)

    accepted = _restricted_solution(X, y, backend="cplex", mip_start=valid_start)
    assert accepted.mip_start_status == "accepted"

    infeasible_start = MIPStartData(np.zeros(full_problem.number_of_variables))
    rejected = _restricted_solution(X, y, backend="cplex", mip_start=infeasible_start)
    assert rejected.mip_start_status == "rejected"
