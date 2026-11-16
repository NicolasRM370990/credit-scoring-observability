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


import pandas as pd

from creditwatch.model.evaluate import evaluate_labeled_batches


class FakeModel:
    def predict_proba(self, features):
        scores = features["feature"].to_numpy(dtype=float)
        return np.column_stack((1 - scores, scores))


def test_evaluate_labeled_batches_associates_drift_by_batch():
    reference = pd.DataFrame(
        {
            "feature": [0.1, 0.2, 0.8, 0.9],
            "target": [0, 0, 1, 1],
        }
    )

    batches = {
        "batch_01": pd.DataFrame(
            {
                "feature": [0.1, 0.2, 0.8, 0.9],
                "target": [0, 0, 1, 1],
            }
        ),
        "batch_02": pd.DataFrame(
            {
                "feature": [0.2, 0.3, 0.7, 0.8],
                "target": [0, 0, 1, 1],
            }
        ),
    }

    drift_by_batch = {
        "batch_01": {
            "drift_detected": False,
            "drift_share": 0.0,
            "score_psi": 0.003,
            "score_ks": 0.02,
            "mean_score_ref": 0.5,
            "mean_score_cur": 0.51,
        },
        "batch_02": {
            "drift_detected": True,
            "drift_share": 0.4,
            "score_psi": 0.15,
            "score_ks": 0.12,
            "mean_score_ref": 0.5,
            "mean_score_cur": 0.65,
        },
    }

    results = evaluate_labeled_batches(
        model=FakeModel(),
        reference=reference,
        batches=batches,
        features=["feature"],
        target="target",
        threshold=0.5,
        drift_by_batch=drift_by_batch,
    )

    assert results[0]["batch_id"] == "reference"
    assert results[0]["drift_detected"] is False

    assert results[1]["drift_detected"] is False
    assert results[1]["drift_share"] == pytest.approx(0.0)
    assert results[1]["score_psi"] == pytest.approx(0.003)
    assert results[1]["score_ks"] == pytest.approx(0.02)
    assert results[1]["mean_score_ref"] == pytest.approx(0.5)
    assert results[1]["mean_score_cur"] == pytest.approx(0.51)

    assert results[2]["drift_detected"] is True
    assert results[2]["drift_share"] == pytest.approx(0.4)
    assert results[2]["score_psi"] == pytest.approx(0.15)
    assert results[2]["score_ks"] == pytest.approx(0.12)
    assert results[2]["mean_score_ref"] == pytest.approx(0.5)
    assert results[2]["mean_score_cur"] == pytest.approx(0.65)

    assert results[1]["positive_rate"] == pytest.approx(0.5)
    assert results[2]["roc_auc"] == pytest.approx(1.0)
    assert results[2]["pr_auc"] == pytest.approx(1.0)
