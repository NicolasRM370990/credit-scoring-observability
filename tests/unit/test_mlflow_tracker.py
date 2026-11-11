"""E2-11: um run no MLflow por lote, com tags, métricas achatadas e artefatos."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from creditwatch.interfaces import RunMetrics
from creditwatch.observability import mlflow_tracker
from creditwatch.observability.mlflow_tracker import (
    MONITORING_EXPERIMENT,
    TRAINING_EXPERIMENT,
    log_run,
    log_training_run,
    tracking_uri,
)

mlflow = pytest.importorskip("mlflow")
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def store(tmp_path, monkeypatch) -> str:
    """SQLite temporário + artefatos em tmp_path (cwd muda para não sujar o repositório)."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return uri


def _metrics(**overrides) -> RunMetrics:
    base = {
        "run_id": "r_ml", "batch_id": "covariate_severe_b03", "scenario": "covariate_severe",
        "model_version": "baseline_v1", "pipeline_status": "ok", "rows_in": 3000,
        "rows_valid": 3000, "latency_s": {"total": 3.2}, "drift_share": 0.3,
        "psi": {"monthly_income": 0.31, "age": 0.01}, "ks": {"monthly_income": 0.12},
        "score_mean": 0.09, "score_psi": 0.12, "n_alerts_warning": 1, "n_alerts_critical": 1,
    }
    base.update(overrides)
    return RunMetrics(**base)


def test_tracking_uri_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    assert tracking_uri() == mlflow_tracker.DEFAULT_TRACKING_URI
    assert (tmp_path / "mlflow").is_dir()  # pasta do SQLite criada
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///x")
    assert tracking_uri() == "file:///x"
    assert tracking_uri("sqlite:///other/db.sqlite") == "sqlite:///other/db.sqlite"


def test_log_run_creates_run_with_tags_metrics_and_artifacts(store, tmp_path):
    report_dir = tmp_path / "drift"
    report_dir.mkdir()
    (report_dir / "covariate_severe_b03.html").write_text("<html>report</html>", encoding="utf-8")
    alerts_path = tmp_path / "alerts.jsonl"
    alerts_path.write_text(
        json.dumps({"run_id": "r_ml", "batch_id": "covariate_severe_b03", "severity": "CRITICAL",
                    "rule_id": "A03_FEATURE_PSI", "message": "x", "ts": "2026-11-11T00:00:00Z"})
        + "\n", encoding="utf-8")

    ml_run_id = log_run(_metrics(), drift_report_dir=report_dir, alerts_path=alerts_path)

    client = mlflow.MlflowClient(tracking_uri=store)
    run = client.get_run(ml_run_id)
    assert run.info.run_name == "covariate_severe_b03"
    assert client.get_experiment(run.info.experiment_id).name == MONITORING_EXPERIMENT
    tags = run.data.tags
    assert tags["scenario"] == "covariate_severe" and tags["status"] == "ok"
    assert tags["model_version"] == "baseline_v1" and tags["contract_version"] == "1.0.0"
    metrics = run.data.metrics
    assert metrics["psi_monthly_income"] == pytest.approx(0.31)
    assert metrics["latency_total_s"] == pytest.approx(3.2)
    assert metrics["drift_share"] == pytest.approx(0.3)
    assert metrics["n_alerts_critical"] == 1.0
    artifacts = {a.path for a in client.list_artifacts(ml_run_id)}
    assert {"run_metrics.json", "alerts.json", "drift_report"} <= artifacts


def test_one_run_per_batch(store):
    for i in range(3):
        log_run(_metrics(run_id=f"r{i}", batch_id=f"no_drift_b0{i}", scenario="no_drift"))
    client = mlflow.MlflowClient(tracking_uri=store)
    exp = client.get_experiment_by_name(MONITORING_EXPERIMENT)
    runs = client.search_runs([exp.experiment_id])
    assert sorted(r.info.run_name for r in runs) == ["no_drift_b00", "no_drift_b01", "no_drift_b02"]


def test_blocked_run_is_logged_with_status_tag(store):
    ml_run_id = log_run(_metrics(pipeline_status="blocked", rows_in=0, rows_rejected=1,
                                 validation_blocked=True, drift_share=None, psi={}, ks={},
                                 score_mean=None, score_psi=None))
    run = mlflow.MlflowClient(tracking_uri=store).get_run(ml_run_id)
    assert run.data.tags["status"] == "blocked"
    assert run.data.metrics["validation_blocked"] == 1.0
    assert "drift_share" not in run.data.metrics


def test_training_run_from_metadata(store):
    metadata = REPO_ROOT / "models" / "baseline_v1.json"
    ml_run_id = log_training_run(metadata)
    client = mlflow.MlflowClient(tracking_uri=store)
    run = client.get_run(ml_run_id)
    assert client.get_experiment(run.info.experiment_id).name == TRAINING_EXPERIMENT
    assert run.info.run_name == "baseline_v1"
    assert run.data.metrics["test_roc_auc"] > 0.75
    assert run.data.params["model_name"] == "HistGradientBoostingClassifier"
    assert any(a.path == "baseline_v1.json" for a in client.list_artifacts(ml_run_id))
