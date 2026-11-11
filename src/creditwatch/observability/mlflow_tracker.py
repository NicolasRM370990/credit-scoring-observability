"""Rastreamento no MLflow (E2-11): um run por lote processado e um run por treino.

- Experimento `credit-scoring-monitoring`: `log_run(RunMetrics)`; nome do run = `batch_id`;
  tags `scenario`, `status`, `model_version`, `contract_version`, `run_id`;
  métricas achatadas por `metrics.flatten_metrics`; artefatos: JSON da execução,
  relatório HTML do Evidently (se existir em artifacts/drift/<batch_id>.html) e alertas do lote.
- Experimento `credit-scoring-training`: `log_training_run(models/<versão>.json)`.

Backend padrão: SQLite local em `mlflow/mlflow.db` (ignorado pelo Git). Abra a UI com
`make mlflow-ui`. Sobrescreva com a variável de ambiente `MLFLOW_TRACKING_URI`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from creditwatch.interfaces import RunMetrics
from creditwatch.observability.metrics import flatten_metrics

DEFAULT_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
DEFAULT_ARTIFACT_ROOT = Path("mlflow/artifacts")
MONITORING_EXPERIMENT = "credit-scoring-monitoring"
TRAINING_EXPERIMENT = "credit-scoring-training"
DRIFT_REPORT_DIR = Path("artifacts/drift")
ALERTS_PATH = Path("artifacts/alerts/alerts.jsonl")


def tracking_uri(explicit: str | None = None) -> str:
    """URI do tracking: argumento > `MLFLOW_TRACKING_URI` > SQLite local."""
    uri = explicit or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return uri


def _mlflow() -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    return mlflow


def _set_experiment(mlflow: Any, name: str) -> None:
    """Cria o experimento (artefatos em mlflow/artifacts/<nome>) se não existir e o ativa."""
    if mlflow.get_experiment_by_name(name) is None:
        root = Path(os.environ.get("CREDITWATCH_MLFLOW_ARTIFACTS", DEFAULT_ARTIFACT_ROOT))
        location = (root / name).resolve()
        location.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=location.as_uri())
    mlflow.set_experiment(name)


def _alerts_for_run(run_id: str, alerts_path: Path) -> list[dict[str, Any]]:
    if not alerts_path.exists():
        return []
    with alerts_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [row for row in rows if row.get("run_id") == run_id]


def log_run(
    metrics: RunMetrics,
    *,
    tracking_uri_override: str | None = None,
    experiment: str = MONITORING_EXPERIMENT,
    extra_artifacts: list[Path] | None = None,
    drift_report_dir: Path = DRIFT_REPORT_DIR,
    alerts_path: Path = ALERTS_PATH,
) -> str:
    """Registra a execução de um lote no MLflow e devolve o `run_id` do MLflow."""
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri(tracking_uri_override))
    _set_experiment(mlflow, experiment)
    with mlflow.start_run(run_name=metrics.batch_id) as active:
        mlflow.set_tags({
            "scenario": metrics.scenario or "manual",
            "status": metrics.pipeline_status,
            "model_version": metrics.model_version or "unknown",
            "contract_version": metrics.contract_version,
            "run_id": metrics.run_id,
            "batch_id": metrics.batch_id,
        })
        mlflow.log_params({
            "batch_id": metrics.batch_id,
            "scenario": metrics.scenario or "manual",
            "model_version": metrics.model_version or "unknown",
            "contract_version": metrics.contract_version,
            "schema_version": metrics.schema_version,
        })
        mlflow.log_metrics(flatten_metrics(metrics))
        mlflow.log_dict(json.loads(metrics.model_dump_json()), "run_metrics.json")

        report = drift_report_dir / f"{metrics.batch_id}.html"
        if report.is_file():
            mlflow.log_artifact(str(report), artifact_path="drift_report")
        alerts = _alerts_for_run(metrics.run_id, alerts_path)
        if alerts:
            mlflow.log_dict({"alerts": alerts}, "alerts.json")
        for artifact in extra_artifacts or []:
            if Path(artifact).is_file():
                mlflow.log_artifact(str(artifact))
        return active.info.run_id


def log_training_run(
    metadata_path: Path = Path("models/baseline_v1.json"),
    *,
    tracking_uri_override: str | None = None,
    experiment: str = TRAINING_EXPERIMENT,
) -> str:
    """Registra o treino do baseline a partir do metadata gravado por `model/train.py`."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri(tracking_uri_override))
    _set_experiment(mlflow, experiment)
    with mlflow.start_run(run_name=metadata.get("model_version", "baseline")) as active:
        mlflow.set_tags({
            "model_version": metadata.get("model_version", "unknown"),
            "model_name": metadata.get("model_name", "unknown"),
            "dataset": metadata.get("dataset", "unknown"),
        })
        mlflow.log_params({
            "model_name": metadata.get("model_name"),
            "seed": metadata.get("seed"),
            "threshold": metadata.get("threshold"),
            "threshold_method": metadata.get("threshold_method"),
            "n_features": len(metadata.get("features", [])),
            "raw_sha256": metadata.get("raw_sha256"),
        })
        for split, values in metadata.get("metrics", {}).items():
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    mlflow.log_metric(f"{split}_{name}", float(value))
        for split, n_rows in metadata.get("splits", {}).items():
            mlflow.log_metric(f"rows_{split}", float(n_rows))
        mlflow.log_artifact(str(metadata_path))
        return active.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Registra artefatos no MLflow")
    parser.add_argument("--training", action="store_true",
                        help="registra o metadata do baseline no experimento de treino")
    parser.add_argument("--metadata", type=Path, default=Path("models/baseline_v1.json"))
    args = parser.parse_args()
    if args.training:
        run_id = log_training_run(args.metadata)
        print(json.dumps({"experiment": TRAINING_EXPERIMENT, "mlflow_run_id": run_id}))
        return 0
    parser.error("informe --training (os runs de monitoramento são criados pelo pipeline)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
