import numpy as np
import pytest

from creditwatch.model.evaluate import evaluate_predictions, select_threshold


def test_evaluate_predictions_returns_expected_metrics():
    result = evaluate_predictions(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), threshold=0.5
    )

    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["ks"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.085)
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert result["f1"] == pytest.approx(1.0)


def test_select_threshold_uses_validation_scores_to_maximize_f1():
    threshold = select_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))

    assert threshold == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("y_true", "scores", "threshold"),
    [
        ([0, 1], [0.2], 0.5),
        ([0, 2], [0.2, 0.8], 0.5),
        ([0, 1], [0.2, float("nan")], 0.5),
        ([0, 1], [0.2, 0.8], 1.1),
        ([0, 0], [0.2, 0.8], 0.5),
    ],
)
def test_evaluate_predictions_rejects_invalid_inputs(y_true, scores, threshold):
    with pytest.raises(ValueError):
        evaluate_predictions(y_true, scores, threshold)


def test_select_threshold_requires_both_classes():
    with pytest.raises(ValueError, match="duas classes"):
        select_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
