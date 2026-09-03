import numpy as np

from src.data.synthetic import (
    generate_synthetic_instance,
    save_synthetic_instance,
)


def _generate(seed=42):
    return generate_synthetic_instance(
        n_samples=200,
        n_features=20,
        informative_ratio=0.2,
        redundant_ratio=0.2,
        correlation_strength=0.95,
        positive_class_fraction=0.35,
        feature_budget_ratio=0.15,
        seed=seed,
    )


def test_synthetic_generator_has_exact_dimensions_labels_and_seed_determinism():
    first = _generate()
    second = _generate()
    different = _generate(seed=43)
    assert first.X.shape == (200, 20)
    assert first.y.shape == (200,)
    assert set(np.unique(first.y)) == {-1, 1}
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert first.data_hash == second.data_hash
    assert first.data_hash != different.data_hash
    assert first.feature_budget == 3


def test_redundant_features_follow_recorded_informative_sources():
    instance = _generate()
    for redundant, source in instance.redundant_sources.items():
        correlation = abs(np.corrcoef(instance.X[:, redundant], instance.X[:, source])[0, 1])
        assert correlation > 0.85


def test_synthetic_instance_is_saved(tmp_path):
    original = _generate()
    array_path, metadata_path = save_synthetic_instance(original, tmp_path, instance_id="saved")
    with np.load(array_path, allow_pickle=False) as saved:
        assert np.array_equal(saved["X"], original.X)
        assert np.array_equal(saved["y"], original.y)
    assert metadata_path.is_file()
