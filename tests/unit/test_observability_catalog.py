"""Testes de consistência do catálogo de métricas."""

import re
from pathlib import Path

import yaml

from creditwatch.interfaces import RunMetrics
from creditwatch.observability.mlflow_tracker import _flatten_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "docs" / "observability_metrics.md"
ALERTS_CONFIG_PATH = PROJECT_ROOT / "config" / "alerts.yaml"


def _read_catalog() -> str:
    assert CATALOG_PATH.is_file(), f"Catálogo não encontrado: {CATALOG_PATH}"
    return CATALOG_PATH.read_text(encoding="utf-8")


def test_catalog_documents_all_run_metrics_fields():
    catalog = _read_catalog()

    nested_fields = {
        "latency_s": "`latency_s.",
        "psi": "`psi.",
        "ks": "`ks.",
    }

    for field_name in RunMetrics.model_fields:
        if field_name in nested_fields:
            assert nested_fields[field_name] in catalog, (
                f"Campo aninhado não documentado: {field_name}"
            )
        else:
            assert f"`{field_name}`" in catalog, f"Campo RunMetrics não documentado: {field_name}"


def test_catalog_documents_mlflow_flattened_metrics():
    catalog = _read_catalog()

    metrics = RunMetrics(
        run_id="catalog_test",
        batch_id="catalog_batch",
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

    flattened = _flatten_metrics(metrics)

    for metric_name in flattened:
        if metric_name.startswith("latency_") and metric_name.endswith("_s"):
            documented_name = "`latency_<stage>_s`"
        elif metric_name.startswith("psi_"):
            documented_name = "`psi_<feature>`"
        elif metric_name.startswith("ks_"):
            documented_name = "`ks_<feature>`"
        else:
            documented_name = f"`{metric_name}`"

        assert documented_name in catalog, f"Métrica MLflow não documentada: {metric_name}"


def test_catalog_documents_all_configured_alert_rules():
    catalog = _read_catalog()

    config = yaml.safe_load(ALERTS_CONFIG_PATH.read_text(encoding="utf-8"))

    configured_rules = config["rules"]

    for rule_id in configured_rules:
        assert f"`{rule_id}`" in catalog, f"Regra de alerta não documentada: {rule_id}"


def test_catalog_alert_thresholds_match_configuration():
    catalog = _read_catalog()

    config = yaml.safe_load(ALERTS_CONFIG_PATH.read_text(encoding="utf-8"))

    expected_thresholds = {
        "A02_DATASET_DRIFT": "0.30",
        "A03_FEATURE_PSI": "0.017247",
        "A04_LATENCY": "120",
        "A06_PREDICTION_DRIFT": "0.10",
    }

    for rule_id, expected_value in expected_thresholds.items():
        configured_value = str(config["rules"][rule_id]["threshold"])

        assert configured_value == expected_value
        assert expected_value in catalog, (
            f"Threshold de {rule_id} não documentado: {expected_value}"
        )

    volume_rule = config["rules"]["A05_VOLUME"]

    assert str(volume_rule["min"]) in catalog
    assert str(volume_rule["max"]) in catalog


def test_catalog_documents_alert_severities():
    catalog = _read_catalog()

    config = yaml.safe_load(ALERTS_CONFIG_PATH.read_text(encoding="utf-8"))

    for rule_id, rule in config["rules"].items():
        severity = rule["severity"]

        pattern = rf"\|\s*{re.escape(rule_id)}\s*\|.*?\|\s*{severity}\s*\|"

        assert re.search(pattern, catalog), (
            f"Severidade {severity} da regra {rule_id} não encontrada na tabela do catálogo"
        )
