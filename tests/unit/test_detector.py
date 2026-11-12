import pandas as pd
import pytest

from creditwatch.drift import detector
from creditwatch.interfaces import DriftThresholds, RunContext


@pytest.fixture
def thresholds():
    return DriftThresholds(
        psi_warn=0.10,
        psi_critical=0.25,
        ks_stat_min=0.10,
        drift_share_alert=0.30,
    )


@pytest.fixture
def reference():
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
            "score": [0.1, 0.2, 0.3, 0.4],
        }
    )


@pytest.fixture
def current():
    return pd.DataFrame(
        {
            "feature_a": [2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
            "score": [0.2, 0.3, 0.4, 0.5],
        }
    )


@pytest.mark.parametrize(
    ("psi_value", "ks_stat", "expected_drifted", "expected_severity"),
    [
        (0.01, 0.01, False, "OK"),
        (0.15, 0.01, True, "WARNING"),
        (0.01, 0.15, True, "WARNING"),
        (0.25, 0.01, True, "CRITICAL"),
        (0.30, 0.20, True, "CRITICAL"),
    ],
)
def test_classify_severity(
    psi_value,
    ks_stat,
    expected_drifted,
    expected_severity,
    thresholds,
):
    drifted, severity = detector._classify_severity(
        psi_value,
        ks_stat,
        thresholds,
    )

    assert drifted is expected_drifted
    assert severity == expected_severity


def test_detect_drift_returns_expected_result(
    reference,
    current,
    thresholds,
    monkeypatch,
):
    psi_values = {
        "feature_a": 0.30,
        "feature_b": 0.01,
        "score": 0.20,
    }

    ks_values = {
        "feature_a": (0.20, 0.001),
        "feature_b": (0.01, 0.90),
        "score": (0.15, 0.01),
    }

    monkeypatch.setattr(
        detector,
        "psi",
        lambda ref, cur: psi_values[ref.name],
    )

    monkeypatch.setattr(
        detector,
        "ks_test",
        lambda ref, cur: ks_values[ref.name],
    )

    monkeypatch.setattr(
        detector,
        "run_evidently_drift",
        lambda reference, current, columns: {column: False for column in columns},
    )

    result = detector.detect_drift(
        reference=reference,
        current=current,
        cfg=thresholds,
        ctx=RunContext.new(
            batch_id="batch_test",
            scenario="unit_test",
        ),
    )

    assert result.batch_id == "batch_test"
    assert result.scenario == "unit_test"
    assert result.reference_rows == 4
    assert result.current_rows == 4

    assert result.drift_share == pytest.approx(0.5)
    assert result.dataset_drift is True

    assert len(result.features) == 2

    feature_a = next(item for item in result.features if item.name == "feature_a")
    feature_b = next(item for item in result.features if item.name == "feature_b")

    assert feature_a.drifted is True
    assert feature_a.severity == "CRITICAL"

    assert feature_b.drifted is False
    assert feature_b.severity == "OK"

    assert result.prediction_drift is not None
    assert result.prediction_drift.score_psi == pytest.approx(0.20)
    assert result.prediction_drift.score_ks == pytest.approx(0.15)


def test_detect_drift_does_not_modify_input_data(
    reference,
    current,
    thresholds,
    monkeypatch,
):
    reference_before = reference.copy(deep=True)
    current_before = current.copy(deep=True)

    monkeypatch.setattr(
        detector,
        "psi",
        lambda ref, cur: 0.01,
    )

    monkeypatch.setattr(
        detector,
        "ks_test",
        lambda ref, cur: (0.01, 0.90),
    )

    monkeypatch.setattr(
        detector,
        "run_evidently_drift",
        lambda reference, current, columns: {column: False for column in columns},
    )

    detector.detect_drift(
        reference,
        current,
        thresholds,
        RunContext.new(batch_id="batch_test"),
    )

    pd.testing.assert_frame_equal(reference, reference_before)
    pd.testing.assert_frame_equal(current, current_before)


@pytest.mark.parametrize(
    "invalid_thresholds",
    [
        {
            "psi_warn": 0.30,
            "psi_critical": 0.20,
            "ks_stat_min": 0.10,
            "drift_share_alert": 0.30,
        },
        {
            "psi_warn": 0.10,
            "psi_critical": 0.25,
            "ks_stat_min": 1.10,
            "drift_share_alert": 0.30,
        },
        {
            "psi_warn": 0.10,
            "psi_critical": 0.25,
            "ks_stat_min": 0.10,
            "drift_share_alert": 1.10,
        },
    ],
)
def test_invalid_thresholds_raise_value_error(
    reference,
    current,
    invalid_thresholds,
):
    cfg = DriftThresholds(**invalid_thresholds)

    with pytest.raises(ValueError):
        detector.detect_drift(
            reference,
            current,
            cfg,
            RunContext.new(batch_id="batch_test"),
        )


def test_empty_reference_raises_value_error(
    reference,
    current,
    thresholds,
):
    empty_reference = reference.iloc[0:0]

    with pytest.raises(ValueError, match="vazios"):
        detector.detect_drift(
            empty_reference,
            current,
            thresholds,
            RunContext.new(batch_id="batch_test"),
        )


def test_missing_required_column_raises_value_error(
    reference,
    current,
    thresholds,
):
    invalid_current = current.drop(columns=["feature_a"])

    with pytest.raises(ValueError, match="ausentes"):
        detector.detect_drift(
            reference,
            invalid_current,
            thresholds,
            RunContext.new(batch_id="batch_test"),
        )


def test_missing_score_column_raises_value_error(
    reference,
    current,
    thresholds,
):
    invalid_current = current.drop(columns=["score"])

    with pytest.raises(ValueError, match="ausentes"):
        detector.detect_drift(
            reference,
            invalid_current,
            thresholds,
            RunContext.new(batch_id="batch_test"),
        )
