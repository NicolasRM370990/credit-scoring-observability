"""Logs, alertas e dashboard não carregam valor de linha (LGPD)."""
from __future__ import annotations

import json

from creditwatch.interfaces import RunContext, RunMetrics, ValidationResult
from creditwatch.observability.alerts import evaluate_alerts
from creditwatch.observability.dashboard import build_dashboard
from creditwatch.observability.logging import get_logger

SENTINEL_ID = "CUST_SENTINEL_7731"
SENTINEL_INCOME = "987654.0"


def test_logs_alerts_and_dashboard_do_not_contain_row_values(tmp_path, monkeypatch):
    ctx = RunContext(run_id="run_gov", batch_id="batch_gov")
    log_path = tmp_path / "pipeline.jsonl"
    get_logger("validate", ctx, log_path).event(
        "batch_checked",
        n_rows=4,
        message="validação concluída",
        extra={
            "first_row": {
                "customer_id": SENTINEL_ID,
                "monthly_income": 987654.0,
                "age": 17,
            },
            "n_failed": 1,
        },
    )

    metrics = RunMetrics(
        run_id="run_gov",
        batch_id="batch_gov",
        pipeline_status="blocked",
        rows_in=4,
        rows_rejected=1,
        validation_blocked=True,
    )
    validation = ValidationResult(
        run_id="run_gov",
        batch_id="batch_gov",
        status="BLOCKED",
        n_rows=4,
        n_rows_failed=1,
        checks=[],
        duration_s=0.01,
    )
    alerts_path = tmp_path / "alerts.jsonl"
    history_path = tmp_path / "run_history.jsonl"
    monkeypatch.setattr(
        "creditwatch.observability.alerts.ALERTS_LOG_PATH", alerts_path,
    )
    history_path.write_text(metrics.model_dump_json() + "\n", encoding="utf-8")
    evaluate_alerts(validation, None, metrics)

    html = build_dashboard(history_path, alerts_path, tmp_path / "index.html", cdn=True)
    blob = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (log_path, alerts_path, history_path, html)
    )
    assert SENTINEL_ID not in blob
    assert SENTINEL_INCOME not in blob
    logged = json.loads(log_path.read_text(encoding="utf-8"))
    assert logged["extra"]["first_row"]["customer_id"] == "<redacted: data column>"
    assert logged["extra"]["n_failed"] == 1
    assert "A01_VALIDATION_BLOCKED" in alerts_path.read_text(encoding="utf-8")
