import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.evaluation.metrics import classification_metrics


def test_four_metrics_match_sklearn_and_manual_gmean():
    y_true = np.array([-1, -1, -1, 1, 1, 1, 1])
    y_pred = np.array([-1, -1, 1, -1, 1, 1, 1])
    metrics = classification_metrics(y_true, y_pred)
    assert list(metrics) == ["balanced_accuracy", "weighted_f1", "accuracy", "gmean"]
    assert metrics["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert metrics["weighted_f1"] == pytest.approx(f1_score(y_true, y_pred, average="weighted"))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["gmean"] == pytest.approx(np.sqrt((3 / 4) * (2 / 3)))


def test_metrics_reject_zero_prediction_label():
    with pytest.raises(ValueError):
        classification_metrics(np.array([-1, 1]), np.array([0, 1]))


def test_metrics_fail_when_evaluation_fold_is_missing_a_class():
    with pytest.raises(ValueError, match="both classes"):
        classification_metrics(np.array([1, 1]), np.array([1, 1]))
