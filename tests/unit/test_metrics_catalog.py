"""O catálogo nomeia as mesmas métricas que o MLflow achata."""
from __future__ import annotations

from pathlib import Path

from creditwatch.interfaces import RunMetrics
from creditwatch.observability.mlflow_tracker import _flatten_metrics

CATALOG = Path("docs/observability_metrics.md")


def test_catalog_lists_every_flattened_metric():
    metrics = RunMetrics(
        run_id="run_catalog",
        batch_id="batch_catalog",
        pipeline_status="ok",
        rows_in=10,
        rows_valid=9,
        rows_rejected=1,
        latency_s={"total": 1.0, "score": 0.4},
        drift_share=0.1,
        psi={"monthly_income": 0.2},
        ks={"monthly_income": 0.3},
        score_mean=0.05,
        score_psi=0.02,
        n_alerts_warning=1,
        n_alerts_critical=0,
    )
    text = CATALOG.read_text(encoding="utf-8")
    for name in _flatten_metrics(metrics):
        if name.startswith("psi_"):
            assert "`psi_<feature>`" in text
        elif name.startswith("ks_"):
            assert "`ks_<feature>`" in text
        elif name.startswith("latency_") and name != "latency_total_s":
            assert "latency_<estágio>_s" in text or f"`{name}`" in text
        else:
            assert f"`{name}`" in text, name
    for rule in (
        "A01_VALIDATION_BLOCKED",
        "A02_DATASET_DRIFT",
        "A03_FEATURE_PSI",
        "A04_LATENCY",
        "A05_VOLUME",
        "A06_PREDICTION_DRIFT",
    ):
        assert rule in text
