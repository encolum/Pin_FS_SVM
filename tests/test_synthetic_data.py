import numpy as np

from src.data.synthetic import (
    generate_synthetic_instance,
    load_synthetic_instance,
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
        label_noise_rate=0.05,
        outlier_sample_rate=0.0,
        outlier_feature_rate=0.0,
        outlier_scale=0.0,
        feature_budget_ratio=0.15,
        seed=seed,
        split="train",
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


def test_synthetic_parameters_and_hash_round_trip(tmp_path):
    original = _generate()
    save_synthetic_instance(original, tmp_path, instance_id="roundtrip")
    loaded = load_synthetic_instance(tmp_path, instance_id="roundtrip")
    assert loaded.data_hash == original.data_hash
    assert np.array_equal(loaded.X, original.X)
    assert np.array_equal(loaded.y, original.y)
    assert loaded.redundant_sources == original.redundant_sources
