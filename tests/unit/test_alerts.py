"""E2-10: cada regra A01-A06 dispara só quando deve; entrada normal gera zero alertas."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creditwatch.interfaces import (
    DriftResult,
    DriftThresholds,
    FeatureDrift,
    PredictionDrift,
    RunMetrics,
    ValidationCheck,
    ValidationResult,
)
from creditwatch.observability.alerts import (
    RULE_IDS,
    evaluate_alerts,
    load_alert_rules,
    load_alerts,
)
from creditwatch.observability.metrics import load_run_history, record_run

THRESHOLDS = DriftThresholds(psi_warn=0.10, psi_critical=0.25, ks_stat_min=0.10,
                             drift_share_alert=0.30)


@pytest.fixture
def rules() -> dict:
    return load_alert_rules(Path("config/alerts.yaml"))


@pytest.fixture
def paths(tmp_path) -> dict:
    return {
        "alerts_path": tmp_path / "alerts.jsonl",
        "history_path": tmp_path / "run_history.jsonl",
        "log_path": tmp_path / "pipeline.jsonl",
    }


def _metrics(**overrides) -> RunMetrics:
    base = {
        "run_id": "r1", "batch_id": "b01", "scenario": "no_drift", "model_version": "baseline_v1",
        "pipeline_status": "ok", "rows_in": 3000, "rows_valid": 3000,
        "latency_s": {"total": 4.0}, "drift_share": 0.0,
        "psi": {"monthly_income": 0.02, "age": 0.01}, "ks": {"monthly_income": 0.01},
        "score_mean": 0.07, "score_psi": 0.01,
    }
    base.update(overrides)
    return RunMetrics(**base)


def _feature(name: str, psi: float, drifted: bool) -> FeatureDrift:
    sev = "CRITICAL" if psi >= 0.25 else "WARNING" if psi >= 0.10 else "OK"
    return FeatureDrift(name=name, psi=psi, ks_stat=psi / 2, ks_pvalue=0.0,
                        evidently_drifted=drifted, drifted=drifted, severity=sev)


def _drift(features: list[FeatureDrift], score_psi: float = 0.01) -> DriftResult:
    share = sum(f.drifted for f in features) / len(features)
    return DriftResult(
        run_id="r1", batch_id="b01", scenario="covariate_severe",
        reference_rows=71400, current_rows=3000,
        dataset_drift=share >= THRESHOLDS.drift_share_alert, drift_share=share,
        features=features,
        prediction_drift=PredictionDrift(score_psi=score_psi, score_ks=0.05,
                                         mean_score_ref=0.07, mean_score_cur=0.09),
        thresholds=THRESHOLDS,
    )


def _blocked_validation() -> ValidationResult:
    return ValidationResult(
        run_id="r1", batch_id="b01", status="BLOCKED", n_rows=3000, n_rows_failed=12,
        checks=[ValidationCheck(rule_id="R01_AGE_RANGE", severity="BLOCK", column="age",
                                passed=False, n_failures=12, failure_rate=0.004,
                                examples=[{"row_index": 17}]),
                ValidationCheck(rule_id="R02_INCOME", severity="BLOCK", column="monthly_income",
                                passed=True)],
        duration_s=0.3,
    )


def test_config_declares_all_six_rules(rules):
    assert set(rules) >= set(RULE_IDS)
    assert rules["A01_VALIDATION_BLOCKED"]["severity"] == "CRITICAL"
    assert rules["A03_FEATURE_PSI"]["severity"] == "CRITICAL"


def test_normal_run_produces_zero_alerts(rules, paths):
    drift = _drift([_feature("monthly_income", 0.02, False), _feature("age", 0.01, False)])
    metrics = _metrics()
    alerts = evaluate_alerts(None, drift, metrics, rules=rules, **paths)
    assert alerts == []
    assert metrics.n_alerts_warning == 0 and metrics.n_alerts_critical == 0
    assert not paths["alerts_path"].exists()


def test_a01_blocked_batch_is_critical_and_names_rules(rules, paths):
    metrics = _metrics(pipeline_status="blocked", rows_in=3000, rows_rejected=12,
                       validation_blocked=True, drift_share=None, psi={}, ks={},
                       score_mean=None, score_psi=None)
    alerts = evaluate_alerts(_blocked_validation(), None, metrics, rules=rules, **paths)
    assert [a.rule_id for a in alerts] == ["A01_VALIDATION_BLOCKED"]
    assert alerts[0].severity == "CRITICAL"
    assert "R01_AGE_RANGE" in alerts[0].message and "R02_INCOME" not in alerts[0].message
    assert metrics.n_alerts_critical == 1


def test_a01_fires_when_rejected_before_reading(rules, paths):
    """Lote rejeitado na ingestão: rows_in=0 não deve disparar A05 em cima do A01."""
    metrics = _metrics(pipeline_status="blocked", rows_in=0, rows_rejected=1,
                       validation_blocked=True, drift_share=None, psi={}, ks={},
                       score_mean=None, score_psi=None)
    alerts = evaluate_alerts(None, None, metrics, rules=rules, **paths)
    assert [a.rule_id for a in alerts] == ["A01_VALIDATION_BLOCKED"]


def test_a02_and_a03_on_severe_drift(rules, paths):
    drift = _drift([
        _feature("monthly_income", 0.31, True),
        _feature("revolving_utilization", 0.27, True),
        _feature("debt_ratio", 0.18, True),
        _feature("age", 0.01, False),
    ])
    metrics = _metrics(drift_share=drift.drift_share,
                       psi={f.name: f.psi for f in drift.features})
    alerts = evaluate_alerts(None, drift, metrics, rules=rules, **paths)
    ids = [a.rule_id for a in alerts]
    assert ids.count("A02_DATASET_DRIFT") == 1
    assert ids.count("A03_FEATURE_PSI") == 2  # só as duas acima de 0.25
    a03 = [a for a in alerts if a.rule_id == "A03_FEATURE_PSI"]
    assert {a.metric for a in a03} == {"psi_monthly_income", "psi_revolving_utilization"}
    assert all(a.severity == "CRITICAL" and a.threshold == 0.25 for a in a03)
    assert metrics.n_alerts_warning == 1 and metrics.n_alerts_critical == 2


def test_a03_falls_back_to_metrics_psi_without_drift_result(rules, paths):
    metrics = _metrics(psi={"debt_ratio": 0.40})
    alerts = evaluate_alerts(None, None, metrics, rules=rules, **paths)
    assert [a.rule_id for a in alerts] == ["A03_FEATURE_PSI"]
    assert alerts[0].value == 0.40


def test_a04_latency_above_threshold(rules, paths):
    metrics = _metrics(latency_s={"total": 121.0})
    alerts = evaluate_alerts(None, None, metrics, rules=rules, **paths)
    assert [a.rule_id for a in alerts] == ["A04_LATENCY"]
    assert alerts[0].severity == "WARNING" and alerts[0].threshold == 120


@pytest.mark.parametrize("rows", [999, 10001])
def test_a05_volume_outside_range(rules, paths, rows):
    metrics = _metrics(rows_in=rows, rows_valid=rows)
    alerts = evaluate_alerts(None, None, metrics, rules=rules, **paths)
    assert [a.rule_id for a in alerts] == ["A05_VOLUME"]
    assert alerts[0].value == rows


@pytest.mark.parametrize("rows", [1000, 3000, 10000])
def test_a05_volume_inside_range_is_silent(rules, paths, rows):
    assert evaluate_alerts(None, None, _metrics(rows_in=rows), rules=rules, **paths) == []


def test_a06_prediction_drift(rules, paths):
    drift = _drift([_feature("age", 0.01, False)], score_psi=0.15)
    alerts = evaluate_alerts(None, drift, _metrics(), rules=rules, **paths)
    assert [a.rule_id for a in alerts] == ["A06_PREDICTION_DRIFT"]
    assert alerts[0].metric == "score_psi" and alerts[0].value == 0.15


def test_alerts_are_persisted_and_history_updated(rules, paths):
    metrics = _metrics(latency_s={"total": 500.0})
    record_run(metrics, paths["history_path"])  # pipeline grava antes dos alertas
    assert load_run_history(paths["history_path"])[0].n_alerts_warning == 0

    alerts = evaluate_alerts(None, None, metrics, rules=rules, **paths)
    assert len(alerts) == 1
    stored = load_alerts(paths["alerts_path"])
    assert stored[0].rule_id == "A04_LATENCY" and stored[0].run_id == "r1"
    assert load_run_history(paths["history_path"])[0].n_alerts_warning == 1

    log_rows = [json.loads(l) for l in paths["log_path"].read_text(encoding="utf-8").splitlines()]
    assert log_rows[-1]["stage"] == "alert" and log_rows[-1]["status"] == "warn"
    assert log_rows[-1]["extra"]["rules"] == ["A04_LATENCY"]


def test_messages_never_contain_row_values(rules, paths):
    """Mensagens trazem ids, contagens e estatísticas; nunca valores de clientes."""
    drift = _drift([_feature("monthly_income", 0.31, True)])
    alerts = evaluate_alerts(_blocked_validation(), drift, _metrics(validation_blocked=True),
                             rules=rules, **paths)
    text = " ".join(a.message for a in alerts)
    assert "row_index" not in text and "17" not in text.split("R01_AGE_RANGE")[0]
