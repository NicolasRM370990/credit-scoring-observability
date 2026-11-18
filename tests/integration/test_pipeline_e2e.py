"""Testes E2E do pipeline CreditWatch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from creditwatch.observability import alerts as alerts_module
from creditwatch.observability import metrics as metrics_module
from creditwatch.observability import mlflow_tracker
from creditwatch.pipeline import run as pipeline

pytestmark = pytest.mark.functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated_pipeline_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Prepara um ambiente temporário com dados e modelo reais."""

    # Copia configurações do projeto.
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "config")

    # Copia os dados necessários para executar o pipeline.
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    for filename in ("production_pool.parquet", "reference_train.parquet"):
        shutil.copy2(
            PROJECT_ROOT / "data" / "processed" / filename,
            processed_dir / filename,
        )

    # Copia modelo e metadados.
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    for filename in ("baseline_v1.joblib", "baseline_v1.json"):
        shutil.copy2(
            PROJECT_ROOT / "models" / filename,
            models_dir / filename,
        )

    # Cria banco MLflow isolado.
    mlflow_dir = tmp_path / "mlflow"
    mlflow_dir.mkdir()
    tracking_db = mlflow_dir / "creditwatch.db"
    tracking_uri = f"sqlite:///{tracking_db.as_posix()}"

    # Redireciona o diretório de trabalho.
    monkeypatch.chdir(tmp_path)

    # Redireciona os caminhos de persistência.
    monkeypatch.setattr(
        pipeline,
        "LOG_PATH",
        Path("artifacts/logs/pipeline.jsonl"),
    )
    monkeypatch.setattr(
        metrics_module,
        "METRICS_LOG_PATH",
        Path("artifacts/metrics/run_history.jsonl"),
    )
    monkeypatch.setattr(
        alerts_module,
        "ALERTS_CONFIG_PATH",
        Path("config/alerts.yaml"),
    )
    monkeypatch.setattr(
        alerts_module,
        "ALERTS_LOG_PATH",
        Path("artifacts/alerts/alerts.jsonl"),
    )
    monkeypatch.setattr(
        mlflow_tracker,
        "TRACKING_URI",
        tracking_uri,
    )

    return tmp_path


def _read_jsonl(path: Path) -> list[dict]:
    """Carrega registros JSONL de um arquivo."""
    if not path.is_file():
        return []

    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_e2e_no_drift_generates_pipeline_artifacts(
    isolated_pipeline_workspace: Path,
) -> None:
    """Executa o cenário de controle e valida os artefatos persistidos."""

    # Executa o cenário completo com componentes reais.
    exit_code = pipeline._run_scenario("no_drift")

    assert exit_code == 0

    # Valida o histórico de métricas.
    metrics_path = Path("artifacts/metrics/run_history.jsonl")
    metrics = _read_jsonl(metrics_path)

    assert len(metrics) == 4
    assert all(item["pipeline_status"] == "ok" for item in metrics)
    assert all(item["scenario"] == "no_drift" for item in metrics)
    assert all(item["rows_in"] > 0 for item in metrics)

    # Valida os eventos do pipeline.
    events_path = Path("artifacts/logs/pipeline.jsonl")
    events = _read_jsonl(events_path)

    assert events
    assert {event["stage"] for event in events} >= {
        "ingest",
        "validate",
        "score",
        "drift",
        "report",
        "pipeline",
        "alert",
    }

    assert all(event["status"] in {"ok", "warn"} for event in events)

    # Valida os relatórios HTML e JSON.
    report_dir = Path("artifacts/drift")

    for batch_number in range(1, 5):
        batch_id = f"batch_{batch_number:02d}"

        html_path = report_dir / f"{batch_id}.html"
        json_path = report_dir / f"{batch_id}.json"

        assert html_path.is_file()
        assert html_path.stat().st_size > 0

        assert json_path.is_file()
        report = json.loads(json_path.read_text(encoding="utf-8"))
        assert report["batch_id"] == batch_id

    # Valida os arquivos de scoring.
    scored_dir = Path("artifacts/scored/no_drift")

    assert len(list(scored_dir.glob("*.parquet"))) == 4

    # Valida o registro das execuções no MLflow.
    client = MlflowClient(tracking_uri=mlflow_tracker.TRACKING_URI)

    experiment = client.get_experiment_by_name(mlflow_tracker.EXPERIMENT_NAME)

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10,
    )

    assert len(runs) == 4

    for run in runs:
        assert run.data.tags["scenario"] == "no_drift"
        assert run.data.tags["status"] == "ok"

        artifacts = client.list_artifacts(run.info.run_id)
        artifact_paths = {artifact.path for artifact in artifacts}

        assert {"alerts", "drift", "reports"} <= artifact_paths


def test_e2e_severe_drift_persists_alerts_and_reports(
    isolated_pipeline_workspace: Path,
) -> None:
    """Executa drift severo e valida métricas, alertas e artefatos."""

    exit_code = pipeline._run_scenario("covariate_severe")

    assert exit_code == 0

    # Valida as métricas dos seis lotes.
    metrics = _read_jsonl(Path("artifacts/metrics/run_history.jsonl"))

    assert len(metrics) == 6
    assert all(item["pipeline_status"] == "ok" for item in metrics)
    assert all(item["scenario"] == "covariate_severe" for item in metrics)
    assert all(item["drift_share"] > 0 for item in metrics)
    assert max(item["score_psi"] for item in metrics) > 0.10

    # Valida a persistência dos alertas.
    alerts = _read_jsonl(Path("artifacts/alerts/alerts.jsonl"))

    assert alerts
    alert_types = {alert["rule_id"] for alert in alerts}

    assert "A03_FEATURE_PSI" in alert_types
    assert "A06_PREDICTION_DRIFT" in alert_types

    # Valida os alertas individuais por execução.
    run_alert_dir = Path("artifacts/alerts/runs")

    assert run_alert_dir.is_dir()
    assert len(list(run_alert_dir.glob("*.json"))) == 6

    # Valida os relatórios de cada lote.
    report_dir = Path("artifacts/drift")

    for batch_number in range(1, 7):
        batch_id = f"batch_{batch_number:02d}"

        html_path = report_dir / f"{batch_id}.html"
        json_path = report_dir / f"{batch_id}.json"

        assert html_path.is_file()
        assert html_path.stat().st_size > 0
        assert json_path.is_file()

    # Valida os seis registros no MLflow.
    client = MlflowClient(tracking_uri=mlflow_tracker.TRACKING_URI)

    experiment = client.get_experiment_by_name(mlflow_tracker.EXPERIMENT_NAME)

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10,
    )

    assert len(runs) == 6

    for run in runs:
        assert run.data.tags["scenario"] == "covariate_severe"
        assert run.data.tags["status"] == "ok"

        artifacts = client.list_artifacts(run.info.run_id)
        artifact_paths = {artifact.path for artifact in artifacts}

        assert {"alerts", "drift", "reports"} <= artifact_paths


def test_e2e_invalid_batch_is_blocked_before_scoring(
    isolated_pipeline_workspace: Path,
) -> None:
    """Valida o bloqueio de lote inválido e a persistência dos alertas."""

    exit_code = pipeline._run_scenario("invalid")

    assert exit_code == 0

    # Valida métricas da execução bloqueada.
    metrics = _read_jsonl(Path("artifacts/metrics/run_history.jsonl"))

    assert len(metrics) == 1

    run_metrics = metrics[0]

    assert run_metrics["pipeline_status"] == "blocked"
    assert run_metrics["validation_blocked"] is True
    assert run_metrics["rows_rejected"] >= 1

    # Valida que o scoring não foi executado.
    scored_files = list(Path("artifacts/scored").rglob("*.parquet"))
    assert scored_files == []

    # Valida alertas.
    alerts = _read_jsonl(Path("artifacts/alerts/alerts.jsonl"))

    alert_types = {alert["rule_id"] for alert in alerts}

    assert "A01_VALIDATION_BLOCKED" in alert_types
    assert "A05_VOLUME" in alert_types

    # Valida arquivo de alertas da execução.
    run_alert_files = list(Path("artifacts/alerts/runs").glob("*.json"))

    assert len(run_alert_files) == 1

    run_alerts = json.loads(run_alert_files[0].read_text(encoding="utf-8"))

    assert all(alert["run_id"] == run_metrics["run_id"] for alert in run_alerts)

    # Valida registro no MLflow.
    client = MlflowClient(tracking_uri=mlflow_tracker.TRACKING_URI)

    experiment = client.get_experiment_by_name(mlflow_tracker.EXPERIMENT_NAME)

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=10,
    )

    assert len(runs) == 1
    assert runs[0].data.tags["status"] == "blocked"

    artifacts = client.list_artifacts(runs[0].info.run_id)
    artifact_paths = {artifact.path for artifact in artifacts}

    assert "alerts" in artifact_paths


def test_e2e_forced_report_failure_does_not_record_success(
    isolated_pipeline_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simula falha na geração do relatório e valida o estado parcial."""

    components = pipeline._load_components()

    def fail_report(*args, **kwargs):
        raise RuntimeError("Falha forçada na geração do relatório")

    components["report"] = fail_report

    monkeypatch.setattr(
        pipeline,
        "_load_components",
        lambda: components,
    )

    # A falha deve interromper a execução.
    with pytest.raises(
        RuntimeError,
        match="Falha forçada na geração do relatório",
    ):
        pipeline._run_scenario("no_drift")

    # O scoring já ocorreu antes da falha.
    scored_path = Path("artifacts/scored/no_drift/batch_01.parquet")

    assert scored_path.is_file()

    # Não deve existir registro de métricas de sucesso.
    metrics_path = Path("artifacts/metrics/run_history.jsonl")

    assert not metrics_path.exists()

    # O log deve registrar as etapas concluídas até o drift.
    events = _read_jsonl(Path("artifacts/logs/pipeline.jsonl"))

    completed_stages = {event["stage"] for event in events if event["status"] == "ok"}

    assert {"ingest", "validate", "score", "drift"} <= completed_stages
    assert "report" not in completed_stages
    assert "pipeline" not in completed_stages

    # A falha não deve produzir relatórios.
    report_dir = Path("artifacts/drift")

    assert not report_dir.exists()


def test_e2e_rerun_preserves_previous_execution_history(
    isolated_pipeline_workspace: Path,
) -> None:
    """Valida reexecução sem perda do histórico anterior."""

    # Primeira execução.
    first_exit_code = pipeline._run_scenario("no_drift")

    assert first_exit_code == 0

    metrics_path = Path("artifacts/metrics/run_history.jsonl")
    events_path = Path("artifacts/logs/pipeline.jsonl")

    first_metrics = _read_jsonl(metrics_path)
    first_events = _read_jsonl(events_path)

    assert len(first_metrics) == 4

    first_run_ids = {item["run_id"] for item in first_metrics}

    assert len(first_run_ids) == 4

    # Segunda execução no mesmo ambiente.
    second_exit_code = pipeline._run_scenario("no_drift")

    assert second_exit_code == 0

    # O histórico anterior deve continuar preservado.
    all_metrics = _read_jsonl(metrics_path)
    all_events = _read_jsonl(events_path)

    assert len(all_metrics) == 8
    assert len(all_events) > len(first_events)

    # Os registros da primeira execução não podem desaparecer.
    assert all_metrics[:4] == first_metrics

    second_metrics = all_metrics[4:]
    second_run_ids = {item["run_id"] for item in second_metrics}

    assert len(second_run_ids) == 4
    assert first_run_ids.isdisjoint(second_run_ids)

    # Cada execução deve gerar seus próprios alertas individuais.
    run_alert_dir = Path("artifacts/alerts/runs")
    run_alert_files = list(run_alert_dir.glob("*.json"))

    assert len(run_alert_files) == 8

    # O MLflow deve manter as oito execuções.
    client = MlflowClient(tracking_uri=mlflow_tracker.TRACKING_URI)

    experiment = client.get_experiment_by_name(mlflow_tracker.EXPERIMENT_NAME)

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=20,
    )

    assert len(runs) == 8

    mlflow_run_ids = {run.data.tags["run_id"] for run in runs}

    assert mlflow_run_ids == first_run_ids | second_run_ids

    # Os relatórios e lotes devem continuar disponíveis.
    assert len(list(Path("artifacts/drift").glob("*.html"))) == 4
    assert len(list(Path("artifacts/drift").glob("*.json"))) == 4
    assert len(list(Path("artifacts/scored/no_drift").glob("*.parquet"))) == 4

    manifest_path = Path("data/production/no_drift/MANIFEST.json")

    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["scenario"] == "no_drift"
    assert len(manifest["batches"]) == 4
