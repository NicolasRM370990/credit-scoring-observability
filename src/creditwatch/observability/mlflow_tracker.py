"""Registro de métricas de monitoramento no MLflow."""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from creditwatch.interfaces import RunMetrics

TRACKING_URI = "sqlite:///mlflow/creditwatch.db"
EXPERIMENT_NAME = "credit-scoring-monitoring"


def _flatten_metrics(metrics: RunMetrics) -> dict[str, float]:
    """Converte os campos de métricas em um dicionário plano."""
    flattened: dict[str, float] = {
        "rows_in": float(metrics.rows_in),
        "rows_valid": float(metrics.rows_valid),
        "rows_rejected": float(metrics.rows_rejected),
        "drift_share": metrics.drift_share,
        "score_mean": metrics.score_mean,
        "score_psi": metrics.score_psi,
        "n_alerts_warning": float(metrics.n_alerts_warning),
        "n_alerts_critical": float(metrics.n_alerts_critical),
    }

    for stage, value in metrics.latency_s.items():
        flattened[f"latency_{stage}_s"] = value

    for feature, value in metrics.psi.items():
        flattened[f"psi_{feature}"] = value

    for feature, value in metrics.ks.items():
        flattened[f"ks_{feature}"] = value

    return {key: float(value) for key, value in flattened.items() if value is not None}


def log_run(
    metrics: RunMetrics,
    artifacts: dict[str, Path] | None = None,
) -> None:
    """Registra métricas e artifacts de uma execução no MLflow."""
    if not isinstance(metrics, RunMetrics):
        raise TypeError("metrics deve ser uma instância de RunMetrics.")

    artifacts = artifacts or {}

    for artifact_name, artifact_path in artifacts.items():
        if not isinstance(artifact_path, Path):
            raise TypeError(f"O artifact '{artifact_name}' deve ser informado como Path.")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact não encontrado: {artifact_path}")

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    tags = {
        "scenario": metrics.scenario or "unknown",
        "status": metrics.pipeline_status,
        "model_version": metrics.model_version or "unknown",
        "contract_version": metrics.contract_version,
        "run_id": metrics.run_id,
        "batch_id": metrics.batch_id,
    }

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=metrics.batch_id,
        tags=tags,
    ):
        mlflow.log_metrics(_flatten_metrics(metrics))

        for artifact_name, artifact_path in artifacts.items():
            mlflow.log_artifact(
                str(artifact_path),
                artifact_path=artifact_name,
            )
