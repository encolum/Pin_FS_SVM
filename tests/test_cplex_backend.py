import numpy as np
import pytest

pytest.importorskip("docplex")
pytest.importorskip("cplex")

from src.models.corrected.pin_fs_svm import PinFSSVM


def test_cplex_pin_fs_backend_matches_formulation_on_small_instance():
    X = np.array([
        [-2.0, 0.1], [-1.0, -0.2], [-0.5, 0.2],
        [0.5, -0.1], [1.0, 0.2], [2.0, -0.2],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    model = PinFSSVM(
        B=1,
        C=5.0,
        tau=0.5,
        lower_bound=-5.0,
        upper_bound=5.0,
        backend="cplex",
        threads=1,
        mip_gap=0.0,
    ).fit(X, y)
    assert model.solver_diagnostics()["backend"].startswith("docplex-cplex-")
    assert model.solver_diagnostics()["status"] == "optimal"
    assert max(model.formulation_residuals(X, y).values()) <= 1e-7
