import numpy as np
import pytest

from src.search import (
    SolverProgressRecord,
    first_incumbent_time,
    solve_restricted_pin_fs,
    time_to_target_gap,
    validate_progress_trajectory,
)
from src.search.progress import primal_integral


def test_progress_helpers_find_incumbent_and_target_gap_times():
    records = [
        SolverProgressRecord(0.0, None, 0.0, None, 0, 0),
        SolverProgressRecord(0.4, 12.0, 4.0, 0.66, 1, 1),
        SolverProgressRecord(0.9, 10.0, 8.0, 0.20, 5, 2),
        SolverProgressRecord(1.2, 10.0, 9.5, 0.05, 8, 2),
    ]
    validate_progress_trajectory(records)
    assert first_incumbent_time(records) == pytest.approx(0.4)
    assert time_to_target_gap(records, 0.20) == pytest.approx(0.9)
    assert time_to_target_gap(records, 0.01) is None
    assert primal_integral(records, horizon=1.2, reference_objective=10.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "records, message",
    [
        (
            [
                SolverProgressRecord(1.0, 10.0, 5.0, 0.5, 1, 1),
                SolverProgressRecord(0.5, 9.0, 6.0, 0.3, 2, 2),
            ],
            "timestamps",
        ),
        (
            [
                SolverProgressRecord(0.5, 9.0, 5.0, 0.4, 1, 1),
                SolverProgressRecord(1.0, 10.0, 6.0, 0.4, 2, 2),
            ],
            "incumbent objectives",
        ),
    ],
)
def test_invalid_progress_trajectory_is_rejected(records, message):
    with pytest.raises(ValueError, match=message):
        validate_progress_trajectory(records)


def test_cplex_progress_trajectory_is_collected_and_validated():
    pytest.importorskip("docplex")
    pytest.importorskip("cplex")
    X = np.array([
        [-2.0, 0.1, 1.0],
        [-1.0, -0.2, 0.5],
        [-0.5, 0.2, -1.0],
        [0.5, -0.1, 1.0],
        [1.0, 0.2, -0.5],
        [2.0, -0.2, -1.0],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    result = solve_restricted_pin_fs(
        X,
        y,
        kernel={0, 1, 2},
        B=1,
        C=5.0,
        tau=0.5,
        coefficient_bounds=(-5.0, 5.0),
        backend="cplex",
        time_limit=10.0,
        mip_gap=0.0,
        threads=1,
        collect_progress=True,
    )

    assert result.progress
    validate_progress_trajectory(result.progress)
    assert first_incumbent_time(result.progress) is not None
    assert result.progress[-1].incumbent_objective == pytest.approx(result.objective)
    assert time_to_target_gap(result.progress, 0.0) is not None
