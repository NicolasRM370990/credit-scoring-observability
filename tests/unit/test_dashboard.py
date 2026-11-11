"""E2-14: dashboard HTML estático a partir de run_history e alerts."""
from __future__ import annotations

from creditwatch.interfaces import Alert, RunMetrics
from creditwatch.observability.alerts import write_alerts
from creditwatch.observability.dashboard import build_dashboard
from creditwatch.observability.metrics import record_run


def _run(run_id: str, batch: str, scenario: str, **overrides) -> RunMetrics:
    base = {
        "run_id": run_id, "batch_id": batch, "scenario": scenario, "model_version": "baseline_v1",
        "pipeline_status": "ok", "rows_in": 3000, "rows_valid": 3000,
        "latency_s": {"total": 2.0}, "drift_share": 0.1,
        "psi": {"monthly_income": 0.05, "age": 0.01}, "score_mean": 0.07, "score_psi": 0.02,
    }
    base.update(overrides)
    return RunMetrics(**base)


def test_dashboard_renders_runs_alerts_and_charts(tmp_path):
    history = tmp_path / "run_history.jsonl"
    alerts = tmp_path / "alerts.jsonl"
    record_run(_run("r1", "no_drift_b01", "no_drift"), history)
    record_run(_run("r2", "covariate_severe_b03", "covariate_severe", drift_share=0.4,
                    psi={"monthly_income": 0.31, "age": 0.01}, n_alerts_critical=1), history)
    record_run(_run("r3", "invalid_age", "invalid", pipeline_status="blocked", rows_in=1000,
                    rows_rejected=1, validation_blocked=True, drift_share=None, psi={},
                    score_mean=None, score_psi=None, n_alerts_critical=1), history)
    write_alerts([Alert(run_id="r2", batch_id="covariate_severe_b03", severity="CRITICAL",
                        rule_id="A03_FEATURE_PSI", message="PSI crítico em `monthly_income`",
                        metric="psi_monthly_income", value=0.31, threshold=0.25)], alerts)

    out = build_dashboard(history, alerts, tmp_path / "dash" / "index.html", cdn=True)
    html = out.read_text(encoding="utf-8")
    assert out.exists() and html.startswith("<!doctype html>")
    for text in ("no_drift_b01", "covariate_severe_b03", "invalid_age", "A03_FEATURE_PSI",
                 "plotly", "monthly_income"):
        assert text in html
    assert html.count("<tr class=") == 3  # uma linha por execução
    assert 'class="blocked"' in html


def test_dashboard_without_data_still_renders(tmp_path):
    out = build_dashboard(tmp_path / "none.jsonl", tmp_path / "none2.jsonl",
                          tmp_path / "index.html", cdn=True)
    html = out.read_text(encoding="utf-8")
    assert "Nenhuma execução registrada" in html and "Nenhum alerta" in html


def test_dashboard_embeds_plotly_for_offline_use(tmp_path):
    history = tmp_path / "run_history.jsonl"
    record_run(_run("r1", "b01", "no_drift"), history)
    out = build_dashboard(history, tmp_path / "alerts.jsonl", tmp_path / "index.html")
    size = out.stat().st_size
    assert size > 1_000_000  # plotly.js embutido
