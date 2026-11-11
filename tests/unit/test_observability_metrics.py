"""E1-10: run_history.jsonl com uma linha válida por execução."""
from __future__ import annotations

from creditwatch.interfaces import RunMetrics
from creditwatch.observability.metrics import (
    flatten_metrics,
    load_run_history,
    record_run,
    summarize_history,
)


def _metrics(run_id: str, **overrides) -> RunMetrics:
    base = {
        "run_id": run_id, "batch_id": f"batch_{run_id}", "scenario": "covariate_moderate",
        "model_version": "baseline_v1", "pipeline_status": "ok", "rows_in": 3000,
        "rows_valid": 3000, "latency_s": {"ingest": 0.1, "total": 2.5}, "drift_share": 0.2,
        "psi": {"monthly_income": 0.12, "age": 0.01}, "ks": {"monthly_income": 0.08},
        "score_mean": 0.07, "score_psi": 0.05,
    }
    base.update(overrides)
    return RunMetrics(**base)


def test_three_calls_produce_three_valid_lines(tmp_path):
    path = tmp_path / "run_history.jsonl"
    for i in range(3):
        record_run(_metrics(f"r{i}"), path)
    history = load_run_history(path)
    assert [row.run_id for row in history] == ["r0", "r1", "r2"]
    assert all(isinstance(row, RunMetrics) for row in history)
    assert path.read_text(encoding="utf-8").count("\n") == 3


def test_same_run_id_is_upserted_not_duplicated(tmp_path):
    path = tmp_path / "run_history.jsonl"
    record_run(_metrics("video_no_drift_b01"), path)
    record_run(_metrics("video_no_drift_b01", n_alerts_warning=2), path)
    history = load_run_history(path)
    assert len(history) == 1 and history[0].n_alerts_warning == 2


def test_load_missing_file_returns_empty(tmp_path):
    assert load_run_history(tmp_path / "nope.jsonl") == []
    assert summarize_history(tmp_path / "nope.jsonl") == {"runs": 0}


def test_flatten_uses_mlflow_naming_convention():
    flat = flatten_metrics(_metrics("r1", validation_blocked=True))
    assert flat["latency_total_s"] == 2.5 and flat["latency_ingest_s"] == 0.1
    assert flat["psi_monthly_income"] == 0.12 and flat["ks_monthly_income"] == 0.08
    assert flat["validation_blocked"] == 1.0
    assert flat["drift_share"] == 0.2 and flat["score_psi"] == 0.05
    assert all(isinstance(v, float) for v in flat.values())


def test_flatten_omits_none_fields():
    flat = flatten_metrics(_metrics("r1", drift_share=None, score_mean=None, score_psi=None))
    assert "drift_share" not in flat and "score_mean" not in flat and "score_psi" not in flat


def test_summary_counts_status_and_alerts(tmp_path):
    path = tmp_path / "run_history.jsonl"
    record_run(_metrics("a"), path)
    record_run(_metrics("b", pipeline_status="blocked", rows_in=0, validation_blocked=True,
                        n_alerts_critical=1), path)
    summary = summarize_history(path)
    assert summary["runs"] == 2 and summary["runs_ok"] == 1 and summary["runs_blocked"] == 1
    assert summary["alerts_critical"] == 1 and summary["rows_in_total"] == 3000
