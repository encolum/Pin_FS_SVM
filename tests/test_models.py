import numpy as np
import pytest

from src.models.pin_fs_svm import PinFSSVM


def test_free_coefficient_can_be_negative():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([1, 1, -1, -1])
    model = PinFSSVM(B=1, C=10.0, tau=0.5, lower_bound=-5, upper_bound=5).fit(X, y)
    assert model.w_[0] < -1e-6


def test_free_intercept_can_be_negative():
    X = np.array([[0.0], [1.0], [3.0], [4.0]])
    y = np.array([-1, -1, 1, 1])
    model = PinFSSVM(B=1, C=10.0, tau=0.5, lower_bound=-5, upper_bound=5).fit(X, y)
    assert model.b_ < -1e-6


def test_zero_score_prediction_is_positive_not_zero():
    model = PinFSSVM(B=1, C=1.0, tau=0.5, lower_bound=-2, upper_bound=2)
    model.w_ = np.array([0.0])
    model.b_ = 0.0
    assert model.predict(np.array([[0.0]])).tolist() == [1]


def test_pin_fs_constraints_and_budget():
    X = np.array([
        [-2.0, 0.1], [-1.0, -0.2], [-0.5, 0.2],
        [0.5, -0.1], [1.0, 0.2], [2.0, -0.2],
    ])
    y = np.array([-1, -1, -1, 1, 1, 1])
    model = PinFSSVM(
        B=1, C=5.0, tau=0.5, lower_bound=-5.0, upper_bound=5.0, mip_gap=0.0
    ).fit(X, y)
    assert int(model.v_.sum()) <= 1
    assert np.allclose(model.w_[model.v_ == 0], 0.0, atol=1e-7)
    assert np.all(model.z_ + 1e-7 >= np.abs(model.w_))
    assert np.allclose(model.z_, np.abs(model.w_), atol=1e-6)
    assert max(model.formulation_residuals(X, y).values()) <= 1e-6
    assert set(model.predict(X)).issubset({-1, 1})


def test_nonpositive_tau_is_rejected():
    with pytest.raises(ValueError):
        PinFSSVM(B=1, C=1, tau=0, lower_bound=-2, upper_bound=2)
