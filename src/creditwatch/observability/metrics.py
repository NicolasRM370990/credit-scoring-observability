"""Persistência de métricas de execução do CreditWatch."""

from __future__ import annotations

import json
from pathlib import Path

from creditwatch.interfaces import RunMetrics

METRICS_LOG_PATH = Path("artifacts/metrics/run_history.jsonl")


def record_run(metrics: RunMetrics) -> None:
    """Registra uma execução no histórico de métricas em formato JSONL."""
    if not isinstance(metrics, RunMetrics):
        raise TypeError("metrics deve ser uma instância de RunMetrics.")

    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = metrics.model_dump(mode="json")

    with METRICS_LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
