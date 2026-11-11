"""E2-15: o catálogo documenta todos os campos de RunMetrics com os nomes usados no MLflow."""
from __future__ import annotations

import re
from pathlib import Path

from creditwatch.interfaces import RunMetrics
from creditwatch.observability.alerts import RULE_IDS
from creditwatch.observability.metrics import flatten_metrics

CATALOG = Path("docs/observability_metrics.md")


def _catalog_names() -> set[str]:
    return set(re.findall(r"`([a-z_<>áéíóú]+)`", CATALOG.read_text(encoding="utf-8")))


def test_every_runmetrics_field_is_documented():
    documented = _catalog_names()
    dict_fields = {"latency_s": "latency_<estágio>_s", "psi": "psi_<feature>", "ks": "ks_<feature>"}
    for field in RunMetrics.model_fields:
        expected = dict_fields.get(field, field)
        assert expected in documented, f"campo `{field}` não documentado como `{expected}`"


def test_flattened_metric_names_match_catalog_patterns():
    metrics = RunMetrics(run_id="r", batch_id="b", pipeline_status="ok",
                         latency_s={"total": 1.0, "ingest": 0.1},
                         psi={"monthly_income": 0.1}, ks={"age": 0.0},
                         drift_share=0.1, score_mean=0.1, score_psi=0.1)
    documented = _catalog_names()
    for name in flatten_metrics(metrics):
        pattern = re.sub(r"^latency_.+_s$", "latency_<estágio>_s", name)
        pattern = re.sub(r"^psi_.+$", "psi_<feature>", pattern)
        pattern = re.sub(r"^ks_.+$", "ks_<feature>", pattern)
        assert pattern in documented or name in documented, f"`{name}` fora do catálogo"


def test_all_alert_rules_are_documented():
    text = CATALOG.read_text(encoding="utf-8")
    for rule in RULE_IDS:
        assert rule in text
