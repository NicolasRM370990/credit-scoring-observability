"""CLI de integração e smoke test do pipeline CreditWatch."""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml

from creditwatch.data.download import load_config
from creditwatch.interfaces import STAGES as CONTRACT_STAGES
from creditwatch.interfaces import BatchRejected, RunContext

LOG_PATH = Path("artifacts/logs/pipeline.jsonl")
STAGES = CONTRACT_STAGES
SCENARIOS = ("no_drift", "covariate_moderate", "covariate_severe")


def _write_event(run_id: str, batch_id: str, stage: str, status: str, detail: str) -> None:
    """Emite o contrato mínimo de log JSON usado no smoke de integração."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(UTC).isoformat(), "level": "INFO" if status == "ok" else "WARNING",
        "run_id": run_id, "batch_id": batch_id, "stage": stage, "event": "stage_finished",
        "status": status, "duration_ms": 0, "n_rows": 0, "message": detail, "extra": {},
    }
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_smoke() -> int:
    """Exercita a sequência e garante log JSON dos seis estágios."""
    run_id, batch_id = f"smoke_{uuid4().hex[:10]}", "smoke_batch"
    for stage in STAGES:
        _write_event(run_id, batch_id, stage, "ok", "smoke component")
    print(json.dumps({"run_id": run_id, "batch_id": batch_id, "stages": STAGES, "log": str(LOG_PATH)}))
    return 0


def _load_components() -> dict[str, Any]:
    specs = {
        "ingestion": ("creditwatch.contracts.ingestion", "ingest_batch"),
        "validation": ("creditwatch.contracts.validate", "validate_batch"),
        "drift": ("creditwatch.drift.detector", "detect_drift"),
        "alerts": ("creditwatch.observability.alerts", "evaluate_alerts"),
        "metrics": ("creditwatch.observability.metrics", "record_run"),
        "mlflow": ("creditwatch.observability.mlflow_tracker", "log_run"),
    }
    loaded: dict[str, Any] = {}
    missing: list[str] = []
    for key, (module_name, symbol) in specs.items():
        try:
            loaded[key] = getattr(importlib.import_module(module_name), symbol)
        except (ImportError, AttributeError):
            missing.append(f"{module_name}.{symbol}")
    if missing:
        raise RuntimeError("Componentes da integração ainda ausentes: " + ", ".join(missing))
    return loaded


def _thresholds() -> Any:
    from creditwatch.interfaces import DriftThresholds

    cfg = yaml.safe_load(Path("config/drift_thresholds.yaml").read_text(encoding="utf-8"))
    return DriftThresholds(**cfg)


def run_batch(
    path: Path,
    scenario: str,
    components: dict[str, Any] | None = None,
    deterministic: bool = False,
) -> int:
    """Executa lote válido nos componentes reais e registra métricas/alertas."""
    from creditwatch.interfaces import RunMetrics
    from creditwatch.model.predict import score_batch

    components = components or _load_components()
    data_cfg = load_config()
    model_cfg = yaml.safe_load(Path("config/model.yaml").read_text(encoding="utf-8"))
    model_path = Path("models") / f"{model_cfg['model_version']}.joblib"
    reference_path = Path(data_cfg["paths"]["processed"]) / "reference_train.parquet"
    batch_id = path.stem
    context = (
        RunContext(run_id=f"video_{scenario}_{batch_id}", batch_id=batch_id, scenario=scenario)
        if deterministic
        else RunContext.new(batch_id=batch_id, scenario=scenario)
    )
    started = time.perf_counter()
    try:
        frame = components["ingestion"](path, context)
    except BatchRejected:
        _write_event(context.run_id, batch_id, "ingest", "warn", "contract rejected batch")
        _write_event(context.run_id, batch_id, "validate", "warn", "scoring skipped")
        rejected_metrics = RunMetrics(
            run_id=context.run_id, batch_id=batch_id, scenario=scenario,
            model_version=model_cfg["model_version"], pipeline_status="blocked",
            rows_rejected=1, validation_blocked=True,
            latency_s={"total": time.perf_counter() - started},
        )
        components["metrics"](rejected_metrics)
        components["mlflow"](rejected_metrics)
        components["alerts"](None, None, rejected_metrics)
        _write_event(context.run_id, batch_id, "pipeline", "ok", "blocked run recorded")
        _write_event(context.run_id, batch_id, "alert", "ok", "validation alert emitted")
        print(json.dumps({"run_id": context.run_id, "batch_id": batch_id, "status": "blocked", "exit_code": 2}))
        return 2
    _write_event(context.run_id, batch_id, "ingest", "ok", "batch accepted")

    validation = components["validation"](frame, batch_id)
    if validation.status == "BLOCKED":
        _write_event(context.run_id, batch_id, "validate", "warn", "scoring skipped")
        rejected_metrics = RunMetrics(
            run_id=context.run_id, batch_id=batch_id, scenario=scenario,
            model_version=model_cfg["model_version"], pipeline_status="blocked",
            rows_in=len(frame), rows_rejected=validation.n_rows_failed,
            validation_blocked=True,
            latency_s={"total": time.perf_counter() - started},
        )
        components["metrics"](rejected_metrics)
        components["mlflow"](rejected_metrics)
        components["alerts"](validation, None, rejected_metrics)
        _write_event(context.run_id, batch_id, "pipeline", "ok", "blocked run recorded")
        _write_event(context.run_id, batch_id, "alert", "ok", "validation alert emitted")
        return 2
    _write_event(context.run_id, batch_id, "validate", "ok", validation.status)
    scored = score_batch(frame, model_path)
    scored_dir = Path("artifacts/scored") / scenario
    scored_dir.mkdir(parents=True, exist_ok=True)
    scored_path = scored_dir / f"{batch_id}.parquet"
    scored.to_parquet(scored_path, index=False)
    _write_event(context.run_id, batch_id, "score", "ok", f"scored {len(scored)} rows")
    reference = pd.read_parquet(reference_path)
    drift = components["drift"](reference, scored, _thresholds(), context)
    _write_event(context.run_id, batch_id, "drift", "ok", "drift result produced")
    if drift.report_html:
        report_source = Path(drift.report_html)
        if report_source.is_file():
            report_dir = Path("artifacts/drift")
            report_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_source, report_dir / f"{batch_id}.html")
            _write_event(context.run_id, batch_id, "report", "ok", "drift report copied")
        else:
            _write_event(context.run_id, batch_id, "report", "warn", "report path does not exist")
    else:
        _write_event(context.run_id, batch_id, "report", "warn", "no HTML report returned")

    elapsed = time.perf_counter() - started
    metrics = RunMetrics(
        run_id=context.run_id, batch_id=batch_id, scenario=scenario,
        model_version=model_cfg["model_version"], pipeline_status="ok",
        rows_in=len(frame), rows_valid=len(frame), latency_s={"total": elapsed},
        drift_share=drift.drift_share,
        psi={item.name: item.psi for item in drift.features},
        ks={item.name: item.ks_stat for item in drift.features},
        score_mean=float(scored["score"].mean()),
        score_psi=drift.prediction_drift.score_psi if drift.prediction_drift else None,
    )
    components["metrics"](metrics)
    components["mlflow"](metrics)
    _write_event(context.run_id, batch_id, "pipeline", "ok", "run metrics recorded")
    alerts = components["alerts"](validation, drift, metrics)
    _write_event(context.run_id, batch_id, "alert", "ok", f"{len(alerts)} alerts")
    print(json.dumps({"run_id": context.run_id, "batch_id": batch_id, "status": "ok", "alerts": len(alerts)}))
    return 0


def _generate_batches(scenario: str) -> list[Path]:
    from creditwatch.simulation.generate import generate_scenario

    cfg = load_config()
    pool = pd.read_parquet(Path(cfg["paths"]["processed"]) / "production_pool.parquet")
    return generate_scenario(
        pool, scenario, Path("config/drift_scenarios.yaml"), Path(cfg["paths"]["production"])
    )


def _generate_invalid() -> Path:
    """Gera lote inválido isolado para demonstrar bloqueio antes do scoring."""
    cfg = load_config()
    pool = pd.read_parquet(Path(cfg["paths"]["processed"]) / "production_pool.parquet")
    batch = pool.head(1000).copy()
    batch.loc[batch.index[0], "age"] = 17
    out = Path(cfg["paths"]["production"]) / "_invalid" / "invalid_age.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    batch.to_parquet(out, index=False)
    return out


def _run_scenario(scenario: str, deterministic: bool = False) -> int:
    components = _load_components()
    paths = [_generate_invalid()] if scenario == "invalid" else _generate_batches(scenario)
    results = [run_batch(path, scenario, components, deterministic) for path in paths]
    if scenario == "invalid":
        return 0 if results and all(code == 2 for code in results) else 3
    return 0 if all(code == 0 for code in results) else 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa ou valida o pipeline CreditWatch")
    parser.add_argument("--smoke", action="store_true", help="roda estágios com stubs locais")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "invalid"))
    parser.add_argument("--batch", type=Path, help="processa lote Parquet existente")
    parser.add_argument("--demo-video", action="store_true", help="roda cenários de demonstração")
    args = parser.parse_args()
    if args.smoke:
        return run_smoke()
    if args.batch:
        return run_batch(args.batch, args.scenario or "manual")
    if args.demo_video:
        return 0 if all(
            _run_scenario(name, deterministic=True) == 0 for name in (*SCENARIOS, "invalid")
        ) else 3
    if args.scenario:
        return _run_scenario(args.scenario)
    parser.error("informe --smoke, --batch, --scenario ou --demo-video")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc
