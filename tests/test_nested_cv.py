import numpy as np

from src.data.corruptions import apply_corruption as real_apply_corruption
from src.data.preprocessing import fit_transform_training
from src.evaluation.nested_cv import _selection_tie_key, run_nested_cv


def test_nested_indices_are_disjoint_and_outer_predictions_complete():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 4))
    y = np.where(X[:, 0] - 0.5 * X[:, 1] >= 0, 1, -1)
    config = {
        "outer_folds": 3,
        "inner_folds": 2,
        "C_grid": [1.0],
        "tau_grid": [0.5],
        "B_grid": [2],
        "solver": {},
        "coefficient_bounds": {"lower": -2, "upper": 2},
        "corruption": {},
    }
    result = run_nested_cv(
        X, y, dataset="synthetic", condition="clean", model_name="l2_svm", config=config, base_seed=42
    )
    predicted_indices = []
    for split, fold in zip(result["split_audit"], result["folds"]):
        outer_test = set(split["outer_test_indices"])
        for inner in split["inner"]:
            assert outer_test.isdisjoint(inner["train_indices"])
            assert outer_test.isdisjoint(inner["validation_indices"])
            assert set(inner["train_indices"]).isdisjoint(inner["validation_indices"])
        predicted_indices.extend(row["sample_index"] for row in fold["predictions"])
        assert set(row["y_pred"] for row in fold["predictions"]).issubset({-1, 1})
        assert len(fold["inner_corruption_manifests"]) == 2
    assert sorted(predicted_indices) == list(range(60))


def test_scaler_is_fitted_on_training_only():
    train = np.array([[0.0], [2.0]])
    test = np.array([[100.0]])
    transformed_train, transformed, scaler = fit_transform_training(train, test)
    assert scaler.mean_[0] == 1.0
    assert transformed_train.mean() == 0.0
    assert transformed[0][0, 0] == 99.0


def test_corruption_receives_standardized_training_partitions(monkeypatch):
    rng = np.random.default_rng(9)
    X = rng.normal(loc=[20.0, -50.0], scale=[3.0, 7.0], size=(60, 2))
    y = np.where(X[:, 0] + X[:, 1] / 5 >= 10, 1, -1)
    observed = []

    def checked_corruption(X_partition, y_partition, condition, *, seed, config):
        if condition == "mixed":
            observed.append(X_partition.copy())
            assert np.allclose(X_partition.mean(axis=0), 0.0, atol=1e-12)
            assert np.allclose(X_partition.std(axis=0), 1.0, atol=1e-12)
        return real_apply_corruption(
            X_partition, y_partition, condition, seed=seed, config=config
        )

    monkeypatch.setattr("src.evaluation.nested_cv.apply_corruption", checked_corruption)
    config = {
        "outer_folds": 3,
        "inner_folds": 2,
        "C_grid": [1.0],
        "tau_grid": [0.5],
        "B_grid": [1],
        "solver": {},
        "coefficient_bounds": {"lower": -2, "upper": 2},
        "corruption": {
            "mixed": {
                "label_flip_rate": 0.0,
                "additive_rate": 0.0,
                "multiplicative_rate": 0.0,
                "additive_std": 0.1,
                "multiplicative_std": 0.1,
            }
        },
    }
    run_nested_cv(
        X, y, dataset="synthetic", condition="mixed", model_name="l2_svm", config=config, base_seed=42
    )
    assert len(observed) == 9  # 3 outer fits plus 2 inner fits per outer fold


def test_selection_tie_key_prefers_fewer_features_then_smaller_budget():
    assert _selection_tie_key({"B": 3, "C": 1.0}, 2.0) < _selection_tie_key(
        {"B": 1, "C": 1.0}, 3.0
    )
    assert _selection_tie_key({"B": 1, "C": 1.0}, 2.0) < _selection_tie_key(
        {"B": 3, "C": 1.0}, 2.0
    )
    assert _selection_tie_key({"C": 2.0}, 2.0) < _selection_tie_key({"C": 10.0}, 2.0)


def test_parallel_candidate_evaluation_matches_sequential():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(48, 3))
    y = np.where(X[:, 0] - 0.2 * X[:, 1] >= 0, 1, -1)
    base = {
        "outer_folds": 2,
        "inner_folds": 2,
        "C_grid": [0.5, 1.0],
        "tau_grid": [0.5],
        "B_grid": [2],
        "solver": {"backend": "scipy", "threads": 1},
        "coefficient_bounds": {"lower": -2, "upper": 2},
        "corruption": {},
    }
    sequential = run_nested_cv(
        X,
        y,
        dataset="synthetic",
        condition="clean",
        model_name="l2_svm",
        config={**base, "parallelism": {"max_workers": 1}},
        base_seed=42,
    )
    parallel = run_nested_cv(
        X,
        y,
        dataset="synthetic",
        condition="clean",
        model_name="l2_svm",
        config={**base, "parallelism": {"max_workers": 2}},
        base_seed=42,
    )
    assert [fold["best_parameters"] for fold in sequential["folds"]] == [
        fold["best_parameters"] for fold in parallel["folds"]
    ]
    assert [fold["metrics"] for fold in sequential["folds"]] == [
        fold["metrics"] for fold in parallel["folds"]
    ]
