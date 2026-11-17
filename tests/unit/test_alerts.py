"""Testes unitários das regras de alerta A01-A06."""

import json

import pytest

from creditwatch.interfaces import (
    Alert,
    DriftResult,
    DriftThresholds,
    FeatureDrift,
    PredictionDrift,
    RunMetrics,
    ValidationResult,
)
from creditwatch.observability import alerts


@pytest.fixture
def metrics():
    return RunMetrics(
        run_id="run_test",
        batch_id="batch_test",
        scenario="unit_test",
        pipeline_status="ok",
        rows_in=3000,
        rows_valid=3000,
        latency_s={"total": 10.0},
        drift_share=0.01,
        psi={"feature_a": 0.01},
        ks={"feature_a": 0.01},
        score_mean=0.5,
        score_psi=0.01,
    )


@pytest.fixture
def validation_passed():
    return ValidationResult(
        run_id="run_test",
        batch_id="batch_test",
        status="PASSED",
        n_rows=3000,
        n_rows_failed=0,
        checks=[],
        duration_s=0.1,
    )


@pytest.fixture
def validation_blocked():
    return ValidationResult(
        run_id="run_test",
        batch_id="batch_test",
        status="BLOCKED",
        n_rows=3000,
        n_rows_failed=10,
        checks=[],
        duration_s=0.1,
    )


@pytest.fixture
def drift_result():
    return DriftResult(
        run_id="run_test",
        batch_id="batch_test",
        scenario="unit_test",
        reference_rows=3000,
        current_rows=3000,
        dataset_drift=False,
        drift_share=0.01,
        features=[
            FeatureDrift(
                name="feature_a",
                psi=0.01,
                ks_stat=0.01,
                ks_pvalue=0.90,
                evidently_drifted=False,
                drifted=False,
                severity="OK",
            )
        ],
        prediction_drift=PredictionDrift(
            score_psi=0.01,
            score_ks=0.01,
            mean_score_ref=0.5,
            mean_score_cur=0.5,
        ),
        thresholds=DriftThresholds(
        psi_warn=0.015639,
        psi_critical=0.017247,
        ks_stat_min=0.042700,
        drift_share_alert=0.30,
      ),
    )


@pytest.fixture
def alert_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "alerts.yaml"
    log_path = tmp_path / "artifacts" / "alerts.jsonl"

    config_path.write_text(
        """
rules:
  A01_VALIDATION_BLOCKED:
    severity: CRITICAL
  A02_DATASET_DRIFT:
    severity: WARNING
    metric: drift_share
    threshold: 0.30
  A03_FEATURE_PSI:
    severity: CRITICAL
    metric: psi
    threshold: 0.017247
  A04_LATENCY:
    severity: WARNING
    metric: latency_total_s
    threshold: 120
  A05_VOLUME:
    severity: WARNING
    metric: rows_in
    min: 1000
    max: 10000
  A06_PREDICTION_DRIFT:
    severity: WARNING
    metric: score_psi
    threshold: 0.10
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(alerts, "ALERTS_CONFIG_PATH", config_path)
    monkeypatch.setattr(alerts, "ALERTS_LOG_PATH", log_path)

    return log_path


def test_a01_validation_blocked_generates_critical_alert(
    metrics,
    validation_blocked,
    alert_environment,
):
    metrics.validation_blocked = True

    result = alerts.evaluate_alerts(
        validation_blocked,
        None,
        metrics,
    )

    assert len(result) == 1
    assert result[0].rule_id == "A01_VALIDATION_BLOCKED"
    assert result[0].severity == "CRITICAL"
    assert metrics.n_alerts_critical == 1


def test_a01_does_not_fire_when_validation_passes(
    metrics,
    validation_passed,
    alert_environment,
):
    result = alerts.evaluate_alerts(
        validation_passed,
        None,
        metrics,
    )

    assert not result


def test_a02_dataset_drift_generates_warning(
    metrics,
    drift_result,
    alert_environment,
):
    drift_result.drift_share = 0.35

    result = alerts.evaluate_alerts(
        None,
        drift_result,
        metrics,
    )

    assert len(result) == 1
    assert result[0].rule_id == "A02_DATASET_DRIFT"
    assert result[0].severity == "WARNING"
    assert result[0].value == pytest.approx(0.35)


def test_a02_does_not_fire_below_threshold(
    metrics,
    drift_result,
    alert_environment,
):
    drift_result.drift_share = 0.29

    result = alerts.evaluate_alerts(
        None,
        drift_result,
        metrics,
    )

    assert not result


def test_a03_feature_psi_generates_critical_alert(
    metrics,
    drift_result,
    alert_environment,
):
    drift_result.features[0].psi = 0.02

    result = alerts.evaluate_alerts(
        None,
        drift_result,
        metrics,
    )

    assert len(result) == 1
    assert result[0].rule_id == "A03_FEATURE_PSI"
    assert result[0].severity == "CRITICAL"
    assert result[0].metric == "psi.feature_a"


def test_a03_generates_one_alert_per_critical_feature(
    metrics,
    drift_result,
    alert_environment,
):
    second_feature = FeatureDrift(
        name="feature_b",
        psi=0.03,
        ks_stat=0.20,
        ks_pvalue=0.01,
        evidently_drifted=True,
        drifted=True,
        severity="CRITICAL",
    )
    drift_result.features[0].psi = 0.02
    drift_result.features.append(second_feature)

    result = alerts.evaluate_alerts(
        None,
        drift_result,
        metrics,
    )

    assert len(result) == 2
    assert {item.metric for item in result} == {
        "psi.feature_a",
        "psi.feature_b",
    }
    assert metrics.n_alerts_critical == 2


def test_a03_does_not_fire_below_threshold(
    metrics,
    drift_result,
    alert_environment,
):
    drift_result.features[0].psi = 0.017246

    result = alerts.evaluate_alerts(
        None,
        drift_result,
        metrics,
    )

    assert not result


def test_a04_latency_generates_warning(
    metrics,
    alert_environment,
):
    metrics.latency_s = {"total": 130.0}

    result = alerts.evaluate_alerts(None, None, metrics)

    assert len(result) == 1
    assert result[0].rule_id == "A04_LATENCY"
    assert result[0].severity == "WARNING"


def test_a04_does_not_fire_below_threshold(
    metrics,
    alert_environment,
):
    metrics.latency_s = {"total": 119.0}

    result = alerts.evaluate_alerts(None, None, metrics)

    assert not result


@pytest.mark.parametrize("rows_in", [999, 10001])
def test_a05_volume_generates_warning(
    metrics,
    alert_environment,
    rows_in,
):
    metrics.rows_in = rows_in

    result = alerts.evaluate_alerts(None, None, metrics)

    assert len(result) == 1
    assert result[0].rule_id == "A05_VOLUME"
    assert result[0].severity == "WARNING"


@pytest.mark.parametrize("rows_in", [1000, 3000, 10000])
def test_a05_does_not_fire_inside_volume_range(
    metrics,
    alert_environment,
    rows_in,
):
    metrics.rows_in = rows_in

    result = alerts.evaluate_alerts(None, None, metrics)

    assert not result


def test_a06_prediction_drift_generates_warning(
    metrics,
    alert_environment,
):
    metrics.score_psi = 0.15

    result = alerts.evaluate_alerts(None, None, metrics)

    assert len(result) == 1
    assert result[0].rule_id == "A06_PREDICTION_DRIFT"
    assert result[0].severity == "WARNING"


def test_a06_does_not_fire_below_threshold(
    metrics,
    alert_environment,
):
    metrics.score_psi = 0.09

    result = alerts.evaluate_alerts(None, None, metrics)

    assert not result


def test_alert_counts_are_updated(
    metrics,
    validation_blocked,
    drift_result,
    alert_environment,
):
    metrics.validation_blocked = True
    drift_result.drift_share = 0.35
    drift_result.features[0].psi = 0.30
    metrics.score_psi = 0.15

    result = alerts.evaluate_alerts(
        validation_blocked,
        drift_result,
        metrics,
    )

    assert len(result) == 4
    assert metrics.n_alerts_warning == 2
    assert metrics.n_alerts_critical == 2


def test_alerts_are_persisted_as_jsonl(
    metrics,
    validation_blocked,
    alert_environment,
):
    metrics.validation_blocked = True

    alerts.evaluate_alerts(
        validation_blocked,
        None,
        metrics,
    )

    assert alert_environment.is_file()

    lines = alert_environment.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["rule_id"] == "A01_VALIDATION_BLOCKED"
    assert record["severity"] == "CRITICAL"
    assert record["run_id"] == metrics.run_id


def test_no_alerts_does_not_create_jsonl(
    metrics,
    validation_passed,
    alert_environment,
):
    result = alerts.evaluate_alerts(
        validation_passed,
        None,
        metrics,
    )

    assert not result
    assert not alert_environment.exists()


def test_returned_items_follow_alert_contract(
    metrics,
    validation_blocked,
    alert_environment,
):
    metrics.validation_blocked = True

    result = alerts.evaluate_alerts(
        validation_blocked,
        None,
        metrics,
    )

    assert all(isinstance(item, Alert) for item in result)
