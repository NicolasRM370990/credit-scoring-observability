"""Avaliação e persistência de alertas do CreditWatch."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from creditwatch.interfaces import Alert, DriftResult, RunMetrics, ValidationResult

ALERTS_CONFIG_PATH = Path("config/alerts.yaml")
ALERTS_LOG_PATH = Path("artifacts/alerts/alerts.jsonl")


def _load_alert_rules() -> dict[str, dict[str, Any]]:
    """Carrega as regras de alerta configuradas em YAML."""
    if not ALERTS_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {ALERTS_CONFIG_PATH}")

    config = yaml.safe_load(ALERTS_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    rules = config.get("rules")

    if not isinstance(rules, dict):
        raise TypeError("Configuração de alertas inválida: esperado 'rules'.")

    return rules


def _create_alert(
    metrics: RunMetrics,
    rule_id: str,
    rule: dict[str, Any],
    message: str,
    metric: str | None = None,
    value: float | None = None,
    threshold: float | None = None,
) -> Alert:
    """Cria um alerta respeitando o contrato de integração."""
    severity = rule.get("severity")

    if severity not in ("WARNING", "CRITICAL"):
        raise ValueError(f"Severidade inválida para a regra {rule_id}: {severity}")

    return Alert(
        ts=datetime.now(UTC),
        run_id=metrics.run_id,
        batch_id=metrics.batch_id,
        severity=severity,
        rule_id=rule_id,
        message=message,
        metric=metric,
        value=value,
        threshold=threshold,
    )


def _evaluate_rules(
    validation: ValidationResult | None,
    drift: DriftResult | None,
    metrics: RunMetrics,
    rules: dict[str, dict[str, Any]],
) -> list[Alert]:
    """Avalia individualmente as regras A01 até A06."""
    alerts: list[Alert] = []

    # A01 — validação bloqueada
    rule_id = "A01_VALIDATION_BLOCKED"
    rule = rules.get(rule_id, {})

    if rule and (
        metrics.validation_blocked or (validation is not None and validation.status == "BLOCKED")
    ):
        alerts.append(
            _create_alert(
                metrics,
                rule_id,
                rule,
                "Lote bloqueado pela validação do contrato de dados.",
            )
        )

    # A02 — drift do dataset
    rule_id = "A02_DATASET_DRIFT"
    rule = rules.get(rule_id, {})

    if rule and drift is not None:
        threshold = float(rule["threshold"])
        value = float(drift.drift_share)

        if value >= threshold:
            alerts.append(
                _create_alert(
                    metrics,
                    rule_id,
                    rule,
                    f"Drift do dataset acima do limite: {value:.4f}.",
                    metric=rule.get("metric", "drift_share"),
                    value=value,
                    threshold=threshold,
                )
            )

    # A03 — PSI crítico por feature
    rule_id = "A03_FEATURE_PSI"
    rule = rules.get(rule_id, {})

    if rule and drift is not None:
        threshold = float(rule["threshold"])

        for feature in drift.features:
            value = float(feature.psi)

            if value >= threshold:
                alerts.append(
                    _create_alert(
                        metrics,
                        rule_id,
                        rule,
                        (f"PSI crítico na feature '{feature.name}': {value:.4f}."),
                        metric=f"psi.{feature.name}",
                        value=value,
                        threshold=threshold,
                    )
                )

    # A04 — latência total
    rule_id = "A04_LATENCY"
    rule = rules.get(rule_id, {})

    if rule:
        threshold = float(rule["threshold"])
        value = float(metrics.latency_s.get("total", 0.0))

        if value >= threshold:
            alerts.append(
                _create_alert(
                    metrics,
                    rule_id,
                    rule,
                    f"Latência total acima do limite: {value:.2f}s.",
                    metric=rule.get("metric", "latency_total_s"),
                    value=value,
                    threshold=threshold,
                )
            )

    # A05 — volume de registros
    rule_id = "A05_VOLUME"
    rule = rules.get(rule_id, {})

    if rule:
        minimum = int(rule["min"])
        maximum = int(rule["max"])
        value = float(metrics.rows_in)

        if value < minimum or value > maximum:
            alerts.append(
                _create_alert(
                    metrics,
                    rule_id,
                    rule,
                    (
                        f"Volume fora do intervalo esperado: {int(value)} "
                        f"(mínimo={minimum}, máximo={maximum})."
                    ),
                    metric=rule.get("metric", "rows_in"),
                    value=value,
                    threshold=float(minimum if value < minimum else maximum),
                )
            )

    # A06 — drift da distribuição dos scores
    rule_id = "A06_PREDICTION_DRIFT"
    rule = rules.get(rule_id, {})

    if rule and metrics.score_psi is not None:
        threshold = float(rule["threshold"])
        value = float(metrics.score_psi)

        if value >= threshold:
            alerts.append(
                _create_alert(
                    metrics,
                    rule_id,
                    rule,
                    f"Drift na distribuição dos scores: {value:.4f}.",
                    metric=rule.get("metric", "score_psi"),
                    value=value,
                    threshold=threshold,
                )
            )

    return alerts


def _persist_alerts(alerts: list[Alert]) -> None:
    """Acrescenta alertas ao arquivo JSONL."""
    if not alerts:
        return

    ALERTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ALERTS_LOG_PATH.open("a", encoding="utf-8") as stream:
        for alert in alerts:
            stream.write(
                json.dumps(
                    alert.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def evaluate_alerts(
    validation: ValidationResult | None,
    drift: DriftResult | None,
    metrics: RunMetrics,
) -> list[Alert]:
    """Avalia regras, atualiza contadores e persiste os alertas."""
    rules = _load_alert_rules()

    alerts = _evaluate_rules(
        validation=validation,
        drift=drift,
        metrics=metrics,
        rules=rules,
    )

    metrics.n_alerts_warning = sum(alert.severity == "WARNING" for alert in alerts)
    metrics.n_alerts_critical = sum(alert.severity == "CRITICAL" for alert in alerts)

    _persist_alerts(alerts)

    return alerts
