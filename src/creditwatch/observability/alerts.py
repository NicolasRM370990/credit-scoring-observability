"""Motor de alertas (E2-10): regras A01-A06 de config/alerts.yaml -> alerts.jsonl.

Regras:
- A01_VALIDATION_BLOCKED  CRITICAL  lote bloqueado pelo contrato de dados
- A02_DATASET_DRIFT       WARNING   drift_share >= threshold (ou Evidently marcou dataset_drift)
- A03_FEATURE_PSI         CRITICAL  PSI de uma feature >= threshold (um alerta por feature)
- A04_LATENCY             WARNING   latência total da execução > threshold (s)
- A05_VOLUME              WARNING   linhas recebidas fora de [min, max]
- A06_PREDICTION_DRIFT    WARNING   PSI da distribuição de scores > threshold

As mensagens só contêm identificadores, contagens e estatísticas agregadas.
Depois de avaliar, `metrics.n_alerts_*` é atualizado e a linha de run_history é regravada.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from creditwatch.interfaces import Alert, DriftResult, RunContext, RunMetrics, ValidationResult
from creditwatch.observability.logging import get_logger
from creditwatch.observability.metrics import RUN_HISTORY_PATH, load_run_history, record_run

ALERTS_CONFIG_PATH = Path("config/alerts.yaml")
ALERTS_PATH = Path("artifacts/alerts/alerts.jsonl")

RULE_IDS = (
    "A01_VALIDATION_BLOCKED", "A02_DATASET_DRIFT", "A03_FEATURE_PSI",
    "A04_LATENCY", "A05_VOLUME", "A06_PREDICTION_DRIFT",
)


def load_alert_rules(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Lê `config/alerts.yaml` e confere que as seis regras existem."""
    cfg = yaml.safe_load(Path(path or ALERTS_CONFIG_PATH).read_text(encoding="utf-8"))
    rules = cfg.get("rules") if isinstance(cfg, dict) else None
    if not isinstance(rules, dict):
        raise TypeError("config/alerts.yaml precisa de uma seção `rules` (mapeamento)")
    missing = [rule for rule in RULE_IDS if rule not in rules]
    if missing:
        raise ValueError(f"regras ausentes em alerts.yaml: {missing}")
    return rules


def _alert(metrics: RunMetrics, rule_id: str, rule: dict[str, Any], message: str,
           metric: str | None = None, value: float | None = None,
           threshold: float | None = None) -> Alert:
    return Alert(
        run_id=metrics.run_id, batch_id=metrics.batch_id, severity=rule["severity"],
        rule_id=rule_id, message=message, metric=metric, value=value, threshold=threshold,
    )


def _rule_a01(validation: ValidationResult | None, metrics: RunMetrics,
              rule: dict[str, Any]) -> list[Alert]:
    blocked = metrics.validation_blocked or (validation is not None and validation.status == "BLOCKED")
    if not blocked:
        return []
    failed = (
        [c.rule_id for c in validation.checks if not c.passed and c.severity == "BLOCK"]
        if validation else []
    )
    detail = f" regras violadas: {', '.join(failed)};" if failed else ""
    message = (
        f"lote bloqueado pelo contrato de dados;{detail} "
        f"{metrics.rows_rejected} linha(s) rejeitada(s); scoring não executado"
    )
    return [_alert(metrics, "A01_VALIDATION_BLOCKED", rule, message, "validation_blocked",
                   1.0, None)]


def _rule_a02(drift: DriftResult | None, metrics: RunMetrics, rule: dict[str, Any]) -> list[Alert]:
    share = drift.drift_share if drift is not None else metrics.drift_share
    if share is None:
        return []
    threshold = float(rule["threshold"])
    flagged = share >= threshold or (drift is not None and drift.dataset_drift)
    if not flagged:
        return []
    n_drifted = sum(f.drifted for f in drift.features) if drift else None
    detail = f" ({n_drifted}/{len(drift.features)} features)" if drift else ""
    return [_alert(metrics, "A02_DATASET_DRIFT", rule,
                   f"drift de dataset: drift_share={share:.3f}{detail} >= {threshold}",
                   "drift_share", float(share), threshold)]


def _rule_a03(drift: DriftResult | None, metrics: RunMetrics, rule: dict[str, Any]) -> list[Alert]:
    threshold = float(rule["threshold"])
    psi_by_feature = (
        {f.name: f.psi for f in drift.features} if drift is not None else dict(metrics.psi)
    )
    return [
        _alert(metrics, "A03_FEATURE_PSI", rule,
               f"PSI crítico em `{feature}`: {psi:.3f} >= {threshold}",
               f"psi_{feature}", float(psi), threshold)
        for feature, psi in psi_by_feature.items()
        if psi >= threshold
    ]


def _rule_a04(metrics: RunMetrics, rule: dict[str, Any]) -> list[Alert]:
    total = metrics.latency_s.get("total")
    threshold = float(rule["threshold"])
    if total is None or total <= threshold:
        return []
    return [_alert(metrics, "A04_LATENCY", rule,
                   f"latência total {total:.1f}s acima do limite de {threshold:.0f}s",
                   "latency_total_s", float(total), threshold)]


def _rule_a05(metrics: RunMetrics, rule: dict[str, Any]) -> list[Alert]:
    if metrics.rows_in == 0 and metrics.pipeline_status == "blocked":
        return []  # lote rejeitado antes da leitura: A01 já cobre
    low, high = int(rule["min"]), int(rule["max"])
    if low <= metrics.rows_in <= high:
        return []
    bound = low if metrics.rows_in < low else high
    return [_alert(metrics, "A05_VOLUME", rule,
                   f"volume de {metrics.rows_in} linhas fora da faixa [{low}, {high}]",
                   "rows_in", float(metrics.rows_in), float(bound))]


def _rule_a06(drift: DriftResult | None, metrics: RunMetrics, rule: dict[str, Any]) -> list[Alert]:
    score_psi = (
        drift.prediction_drift.score_psi
        if drift is not None and drift.prediction_drift is not None
        else metrics.score_psi
    )
    threshold = float(rule["threshold"])
    if score_psi is None or score_psi <= threshold:
        return []
    extra = ""
    if drift is not None and drift.prediction_drift is not None:
        pdrift = drift.prediction_drift
        extra = f" (score médio {pdrift.mean_score_ref:.3f} -> {pdrift.mean_score_cur:.3f})"
    return [_alert(metrics, "A06_PREDICTION_DRIFT", rule,
                   f"drift na distribuição de scores: PSI={score_psi:.3f} > {threshold}{extra}",
                   "score_psi", float(score_psi), threshold)]


def write_alerts(alerts: list[Alert], path: Path | None = None) -> Path:
    """Acrescenta os alertas em alerts.jsonl (uma linha por alerta)."""
    file = Path(path or ALERTS_PATH)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as stream:
        for alert in alerts:
            stream.write(alert.model_dump_json() + "\n")
    return file


def load_alerts(path: Path | None = None) -> list[Alert]:
    file = Path(path or ALERTS_PATH)
    if not file.exists():
        return []
    with file.open(encoding="utf-8") as stream:
        return [Alert.model_validate_json(line) for line in stream if line.strip()]


def evaluate_alerts(
    validation: ValidationResult | None,
    drift: DriftResult | None,
    metrics: RunMetrics,
    *,
    rules: dict[str, dict[str, Any]] | None = None,
    alerts_path: Path | None = None,
    history_path: Path | None = None,
    log_path: Path | None = None,
) -> list[Alert]:
    """Aplica as regras A01-A06, grava os alertas, atualiza `metrics` e o run_history."""
    rules = rules or load_alert_rules()
    alerts: list[Alert] = []
    alerts += _rule_a01(validation, metrics, rules["A01_VALIDATION_BLOCKED"])
    alerts += _rule_a02(drift, metrics, rules["A02_DATASET_DRIFT"])
    alerts += _rule_a03(drift, metrics, rules["A03_FEATURE_PSI"])
    alerts += _rule_a04(metrics, rules["A04_LATENCY"])
    alerts += _rule_a05(metrics, rules["A05_VOLUME"])
    alerts += _rule_a06(drift, metrics, rules["A06_PREDICTION_DRIFT"])

    metrics.n_alerts_warning = sum(a.severity == "WARNING" for a in alerts)
    metrics.n_alerts_critical = sum(a.severity == "CRITICAL" for a in alerts)

    if alerts:
        write_alerts(alerts, alerts_path)

    history_file = Path(history_path or RUN_HISTORY_PATH)
    if any(row.run_id == metrics.run_id for row in load_run_history(history_file)):
        record_run(metrics, history_file)

    ctx = RunContext(run_id=metrics.run_id, batch_id=metrics.batch_id, scenario=metrics.scenario)
    log = get_logger("alert", ctx, log_path)
    log.event(
        "alerts_evaluated", "warn" if alerts else "ok", n_rows=metrics.rows_in,
        message=f"{len(alerts)} alerta(s)",
        extra={
            "n_warning": metrics.n_alerts_warning,
            "n_critical": metrics.n_alerts_critical,
            "rules": sorted({a.rule_id for a in alerts}),
        },
    )
    return alerts


def main() -> int:
    """CLI: resume alerts.jsonl por regra e severidade."""
    alerts = load_alerts()
    summary: dict[str, int] = {}
    for alert in alerts:
        key = f"{alert.rule_id} [{alert.severity}]"
        summary[key] = summary.get(key, 0) + 1
    print(json.dumps({"total": len(alerts), "by_rule": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
