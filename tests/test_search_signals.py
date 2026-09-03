import numpy as np

from src.data.synthetic import generate_synthetic_instance
from src.search.signals import LPRelaxationCache, compute_static_signals, solve_pin_fs_relaxation


def _data():
    return generate_synthetic_instance(
        n_samples=20,
        n_features=6,
        informative_ratio=0.34,
        redundant_ratio=0.33,
        correlation_strength=0.9,
        positive_class_fraction=0.5,
        feature_budget_ratio=0.34,
        seed=4,
    )


def test_lp_relaxation_is_cached_and_within_selector_bounds():
    data = _data()
    cache = LPRelaxationCache()
    kwargs = dict(
        B=data.feature_budget,
        C=1.0,
        tau=0.5,
        coefficient_bounds=(-4.0, 4.0),
        backend="scipy",
        time_limit=2.0,
        cache=cache,
    )
    first = solve_pin_fs_relaxation(data.X, data.y, **kwargs)
    second = solve_pin_fs_relaxation(data.X, data.y, **kwargs)
    assert np.all((first.v_lp >= -1e-8) & (first.v_lp <= 1 + 1e-8))
    assert second.from_cache is True
    assert second.runtime == 0.0
    assert second.objective == first.objective


def test_signal_normalization_parameters_are_persistable_and_deterministic():
    data = _data()
    kwargs = dict(
        B=data.feature_budget,
        C=1.0,
        tau=0.5,
        coefficient_bounds=(-4.0, 4.0),
        seed=9,
        use_lp=False,
    )
    first = compute_static_signals(data.X, data.y, **kwargs)
    second = compute_static_signals(data.X, data.y, **kwargs)
    assert first.normalization.to_dict() == second.normalization.to_dict()
    for values in first.values.values():
        assert np.all(values >= 0)
        assert np.all(values <= 1)
