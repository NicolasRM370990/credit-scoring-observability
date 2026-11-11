"""Métricas por execução (E1-10): uma linha `RunMetrics` em run_history.jsonl.

`record_run` faz *upsert* por `run_id`: reexecutar o mesmo lote (ex.: `--demo-video`,
que usa run_id determinístico) não duplica linhas, e o motor de alertas pode
atualizar `n_alerts_*` depois de o pipeline ter gravado a linha.
"""
from __future__ import annotations

import json
from pathlib import Path

from creditwatch.interfaces import RunMetrics

RUN_HISTORY_PATH = Path("artifacts/metrics/run_history.jsonl")


def load_run_history(path: Path | None = None) -> list[RunMetrics]:
    """Lê e valida todas as linhas de run_history.jsonl (lista vazia se não existir)."""
    file = Path(path or RUN_HISTORY_PATH)
    if not file.exists():
        return []
    with file.open(encoding="utf-8") as stream:
        return [RunMetrics.model_validate_json(line) for line in stream if line.strip()]


def record_run(metrics: RunMetrics, path: Path | None = None) -> Path:
    """Grava (ou substitui, se o `run_id` já existir) a linha da execução."""
    file = Path(path or RUN_HISTORY_PATH)
    history = [row for row in load_run_history(file) if row.run_id != metrics.run_id]
    history.append(metrics)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(row.model_dump_json() + "\n")
    return file


def flatten_metrics(metrics: RunMetrics) -> dict[str, float]:
    """Achata `RunMetrics` em métricas numéricas (nomes usados no MLflow e no catálogo).

    `latency_s` vira `latency_<estágio>_s`; `psi`/`ks` viram `psi_<feature>`/`ks_<feature>`.
    Campos `None` são omitidos.
    """
    flat: dict[str, float] = {
        "rows_in": float(metrics.rows_in),
        "rows_valid": float(metrics.rows_valid),
        "rows_rejected": float(metrics.rows_rejected),
        "validation_blocked": float(metrics.validation_blocked),
        "n_alerts_warning": float(metrics.n_alerts_warning),
        "n_alerts_critical": float(metrics.n_alerts_critical),
    }
    for stage_name, seconds in metrics.latency_s.items():
        flat[f"latency_{stage_name}_s"] = float(seconds)
    for feature, value in metrics.psi.items():
        flat[f"psi_{feature}"] = float(value)
    for feature, value in metrics.ks.items():
        flat[f"ks_{feature}"] = float(value)
    for name in ("drift_share", "score_mean", "score_psi"):
        value = getattr(metrics, name)
        if value is not None:
            flat[name] = float(value)
    return flat


def summarize_history(path: Path | None = None) -> dict[str, float | int]:
    """Resumo agregado usado pelo dashboard e pelo README (contagens, não valores de linha)."""
    history = load_run_history(path)
    if not history:
        return {"runs": 0}
    return {
        "runs": len(history),
        "runs_ok": sum(row.pipeline_status == "ok" for row in history),
        "runs_blocked": sum(row.pipeline_status == "blocked" for row in history),
        "runs_error": sum(row.pipeline_status == "error" for row in history),
        "rows_in_total": sum(row.rows_in for row in history),
        "alerts_warning": sum(row.n_alerts_warning for row in history),
        "alerts_critical": sum(row.n_alerts_critical for row in history),
    }


def main() -> int:
    """CLI: imprime o resumo do histórico em JSON."""
    print(json.dumps(summarize_history(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
