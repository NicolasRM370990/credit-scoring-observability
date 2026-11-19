"""Testes unitários do registro de métricas no MLflow."""

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from creditwatch.interfaces import RunMetrics
from creditwatch.observability import mlflow_tracker


@pytest.fixture
def metrics():
    return RunMetrics(
        run_id="run_test",
        batch_id="batch_test",
        scenario="no_drift",
        model_version="baseline_v1",
        contract_version="1.0.0",
        pipeline_status="ok",
        rows_in=3000,
        rows_valid=2980,
        rows_rejected=20,
        latency_s={
            "ingest": 0.5,
            "validate": 0.2,
            "score": 1.5,
            "drift": 2.0,
            "total": 4.2,
        },
        drift_share=0.10,
        psi={"monthly_income": 0.15},
        ks={"monthly_income": 0.12},
        score_mean=0.08,
        score_psi=0.05,
        n_alerts_warning=2,
        n_alerts_critical=1,
    )


@pytest.fixture
def mlflow_db(tmp_path, monkeypatch):
    db_path = tmp_path / "mlflow" / "test.db"
    db_path.parent.mkdir(parents=True)

    tracking_uri = f"sqlite:///{db_path.as_posix()}"

    monkeypatch.setattr(
        mlflow_tracker,
        "TRACKING_URI",
        tracking_uri,
    )

    monkeypatch.setattr(
        mlflow_tracker,
        "EXPERIMENT_NAME",
        "credit-scoring-monitoring-test",
    )

    mlflow.set_tracking_uri(tracking_uri)

    yield tracking_uri

    mlflow.end_run()


def test_log_run_creates_experiment(metrics, mlflow_db):
    mlflow_tracker.log_run(metrics)

    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("credit-scoring-monitoring-test")

    assert experiment is not None


def test_log_run_uses_batch_id_as_run_name(metrics, mlflow_db):
    mlflow_tracker.log_run(metrics)

    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("credit-scoring-monitoring-test")
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1
    assert runs[0].info.run_name == "batch_test"


def test_log_run_persists_contract_tags(metrics, mlflow_db):
    mlflow_tracker.log_run(metrics)

    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("credit-scoring-monitoring-test")
    runs = client.search_runs([experiment.experiment_id])

    tags = runs[0].data.tags

    assert tags["scenario"] == "no_drift"
    assert tags["status"] == "ok"
    assert tags["model_version"] == "baseline_v1"
    assert tags["contract_version"] == "1.0.0"


def test_log_run_flattens_and_persists_metrics(metrics, mlflow_db):
    mlflow_tracker.log_run(metrics)

    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("credit-scoring-monitoring-test")
    runs = client.search_runs([experiment.experiment_id])

    recorded = runs[0].data.metrics

    assert recorded["latency_ingest_s"] == pytest.approx(0.5)
    assert recorded["latency_total_s"] == pytest.approx(4.2)
    assert recorded["psi_monthly_income"] == pytest.approx(0.15)
    assert recorded["ks_monthly_income"] == pytest.approx(0.12)
    assert recorded["drift_share"] == pytest.approx(0.10)
    assert recorded["score_mean"] == pytest.approx(0.08)
    assert recorded["score_psi"] == pytest.approx(0.05)
    assert recorded["rows_in"] == 3000
    assert recorded["n_alerts_warning"] == 2
    assert recorded["n_alerts_critical"] == 1


def test_log_run_does_not_modify_metrics(metrics, mlflow_db):
    original = metrics.model_dump(mode="json")

    mlflow_tracker.log_run(metrics)

    assert metrics.model_dump(mode="json") == original


def test_log_run_persists_artifacts(metrics, mlflow_db, tmp_path):
    html_file = tmp_path / "batch_test.html"
    json_file = tmp_path / "batch_test.json"

    html_file.write_text("<html>drift report</html>", encoding="utf-8")
    json_file.write_text('{"batch_id": "batch_test"}', encoding="utf-8")

    mlflow_tracker.log_run(
        metrics,
        artifacts={
            "drift": html_file,
            "reports": json_file,
        },
    )

    client = MlflowClient(tracking_uri=mlflow_db)
    experiment = client.get_experiment_by_name("credit-scoring-monitoring-test")
    runs = client.search_runs([experiment.experiment_id])

    assert len(runs) == 1

    run_id = runs[0].info.run_id

    drift_artifacts = client.list_artifacts(run_id, "drift")
    report_artifacts = client.list_artifacts(run_id, "reports")

    assert [item.path for item in drift_artifacts] == ["drift/batch_test.html"]
    assert [item.path for item in report_artifacts] == ["reports/batch_test.json"]

    downloaded_html = client.download_artifacts(
        run_id,
        "drift/batch_test.html",
        str(tmp_path / "downloaded"),
    )

    assert Path(downloaded_html).read_text(encoding="utf-8") == ("<html>drift report</html>")


def test_log_run_rejects_missing_artifact(metrics, mlflow_db, tmp_path):
    missing_file = tmp_path / "missing.html"

    with pytest.raises(FileNotFoundError, match="Artifact não encontrado"):
        mlflow_tracker.log_run(
            metrics,
            artifacts={"drift": missing_file},
        )
