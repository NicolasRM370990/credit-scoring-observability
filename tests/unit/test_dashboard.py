"""Testes unitários do dashboard de observabilidade."""

import json

from creditwatch.observability.dashboard import generate_dashboard


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_generate_dashboard_creates_offline_html(tmp_path):
    run_history = tmp_path / "run_history.jsonl"
    alerts_history = tmp_path / "alerts.jsonl"
    output = tmp_path / "dashboard" / "index.html"

    _write_jsonl(
        run_history,
        [
            {
                "run_id": "run_001",
                "batch_id": "batch_no_drift",
                "scenario": "no_drift",
                "pipeline_status": "ok",
                "rows_in": 3000,
                "latency_s": {"total": 4.2},
                "drift_share": 0.10,
                "psi": {"monthly_income": 0.15},
                "created_at": "2026-11-17T10:00:00+00:00",
            },
            {
                "run_id": "run_002",
                "batch_id": "batch_severe",
                "scenario": "covariate_severe",
                "pipeline_status": "blocked",
                "rows_in": 1200,
                "latency_s": {"total": 8.5},
                "drift_share": 0.45,
                "psi": {"monthly_income": 0.30},
                "created_at": "2026-11-17T11:00:00+00:00",
            },
        ],
    )

    _write_jsonl(
        alerts_history,
        [
            {
                "ts": "2026-11-17T11:00:00+00:00",
                "run_id": "run_002",
                "batch_id": "batch_severe",
                "severity": "CRITICAL",
                "rule_id": "A03_FEATURE_PSI",
                "metric": "psi.monthly_income",
                "value": 0.30,
                "threshold": 0.017247,
                "message": "Feature PSI exceeded threshold",
            },
            {
                "ts": "2026-11-17T11:01:00+00:00",
                "run_id": "run_002",
                "batch_id": "batch_severe",
                "severity": "WARNING",
                "rule_id": "A02_DATASET_DRIFT",
                "metric": "drift_share",
                "value": 0.45,
                "threshold": 0.30,
                "message": "Dataset drift detected",
            },
        ],
    )

    result = generate_dashboard(
        run_history_path=run_history,
        alerts_path=alerts_history,
        output_path=output,
    )

    assert result == output
    assert output.is_file()

    html = output.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in html
    assert "CreditWatch" in html
    assert "no_drift" in html
    assert "covariate_severe" in html
    assert "batch_no_drift" in html
    assert "batch_severe" in html
    assert "A03_FEATURE_PSI" in html
    assert "A02_DATASET_DRIFT" in html

    # O código JavaScript do Plotly deve estar incorporado no HTML.
    assert "Plotly.newPlot" in html
    assert "<script src=" not in html


def test_generate_dashboard_calculates_execution_and_alert_totals(
    tmp_path,
):
    run_history = tmp_path / "run_history.jsonl"
    alerts_history = tmp_path / "alerts.jsonl"
    output = tmp_path / "index.html"

    _write_jsonl(
        run_history,
        [
            {
                "run_id": "run_001",
                "batch_id": "batch_001",
                "scenario": "no_drift",
                "pipeline_status": "ok",
                "rows_in": 3000,
                "latency_s": {"total": 4.2},
                "drift_share": 0.10,
                "psi": {},
            },
            {
                "run_id": "run_002",
                "batch_id": "batch_002",
                "scenario": "invalid",
                "pipeline_status": "blocked",
                "rows_in": 1000,
                "latency_s": {"total": 2.0},
                "drift_share": None,
                "psi": {},
            },
        ],
    )

    _write_jsonl(
        alerts_history,
        [
            {
                "run_id": "run_002",
                "batch_id": "batch_002",
                "severity": "CRITICAL",
                "rule_id": "A01_VALIDATION_BLOCKED",
            },
            {
                "run_id": "run_002",
                "batch_id": "batch_002",
                "severity": "WARNING",
                "rule_id": "A05_VOLUME",
            },
        ],
    )

    generate_dashboard(
        run_history_path=run_history,
        alerts_path=alerts_history,
        output_path=output,
    )

    html = output.read_text(encoding="utf-8")

    assert "2</div>" in html
    assert "4000" in html
    assert "A01_VALIDATION_BLOCKED" in html
    assert "A05_VOLUME" in html


def test_generate_dashboard_handles_missing_input_files(tmp_path):
    output = tmp_path / "dashboard" / "index.html"

    result = generate_dashboard(
        run_history_path=tmp_path / "missing_runs.jsonl",
        alerts_path=tmp_path / "missing_alerts.jsonl",
        output_path=output,
    )

    assert result == output
    assert output.is_file()

    html = output.read_text(encoding="utf-8")

    assert "CreditWatch" in html
    assert "Nenhuma execução encontrada." in html
    assert "Nenhum alerta registrado." in html
