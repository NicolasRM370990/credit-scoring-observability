"""Dashboard lê os JSONL da main sem regravar métricas ou alertas."""
from __future__ import annotations

import json

from creditwatch.interfaces import Alert, RunMetrics
from creditwatch.observability.dashboard import build_dashboard, print_summary


def _metrics() -> RunMetrics:
    return RunMetrics(
        run_id="run_dash",
        batch_id="batch_dash",
        scenario="covariate_moderate",
        model_version="baseline_v1",
        pipeline_status="ok",
        rows_in=1200,
        rows_valid=1200,
        latency_s={"total": 3.5},
        drift_share=0.2,
        psi={"monthly_income": 0.11},
        score_mean=0.07,
        score_psi=0.04,
        n_alerts_warning=1,
        n_alerts_critical=0,
    )


def test_dashboard_renders_aggregates_only(tmp_path, capsys):
    history = tmp_path / "run_history.jsonl"
    alerts = tmp_path / "alerts.jsonl"
    history.write_text(_metrics().model_dump_json() + "\n", encoding="utf-8")
    alerts.write_text(
        Alert(
            run_id="run_dash",
            batch_id="batch_dash",
            severity="WARNING",
            rule_id="A02_DATASET_DRIFT",
            message="drift_share acima do limite",
            metric="drift_share",
            value=0.2,
            threshold=0.3,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    html_path = build_dashboard(history, alerts, tmp_path / "index.html", cdn=True)
    html = html_path.read_text(encoding="utf-8")

    assert "batch_dash" in html
    assert "A02_DATASET_DRIFT" in html
    assert "plotly" in html
    assert "customer_id" not in html

    print_summary(history, alerts)
    summary = json.loads(capsys.readouterr().out)
    assert summary["runs"] == 1
    assert summary["alerts"] == 1
    assert summary["alerts_warning"] == 1


def test_dashboard_empty_history(tmp_path):
    html = build_dashboard(
        tmp_path / "missing.jsonl", tmp_path / "missing-alerts.jsonl", tmp_path / "index.html",
        cdn=True,
    ).read_text(encoding="utf-8")
    assert "Nenhuma execução registrada" in html
