"""E1-09: logging JSON por estágio."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from creditwatch.interfaces import LOG_REQUIRED_FIELDS, RunContext
from creditwatch.observability.logging import (
    REDACTED,
    build_event,
    get_logger,
    read_log,
    sanitize_extra,
    stage,
    stage_timer,
)


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(run_id="r_test", batch_id="b_test", scenario="no_drift")


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_event_has_exactly_the_required_fields(ctx, tmp_path):
    log_path = tmp_path / "pipeline.jsonl"
    get_logger("ingest", ctx, log_path).event("batch_accepted", n_rows=10, message="ok")
    (row,) = _lines(log_path)
    assert tuple(row) == LOG_REQUIRED_FIELDS
    assert row["run_id"] == "r_test" and row["batch_id"] == "b_test"
    assert row["stage"] == "ingest" and row["status"] == "ok" and row["n_rows"] == 10


def test_stage_timer_writes_start_and_finish(ctx, tmp_path):
    log_path = tmp_path / "pipeline.jsonl"
    with stage_timer("validate", ctx, n_rows=5, log_path=log_path):
        pass
    rows = _lines(log_path)
    assert [r["event"] for r in rows] == ["stage_started", "stage_finished"]
    assert rows[1]["duration_ms"] >= 0 and rows[1]["status"] == "ok"


def test_stage_timer_logs_error_and_reraises(ctx, tmp_path):
    log_path = tmp_path / "pipeline.jsonl"
    with pytest.raises(ValueError), stage_timer("score", ctx, log_path=log_path):
        raise ValueError("boom")
    rows = _lines(log_path)
    assert rows[-1]["event"] == "stage_failed" and rows[-1]["status"] == "error"
    assert rows[-1]["level"] == "ERROR"
    assert rows[-1]["extra"] == {"exception": "ValueError"}
    assert "boom" not in json.dumps(rows)  # a mensagem da exceção não vai para o log


def test_stage_decorator_records_rows_of_dataframe(ctx, tmp_path):
    log_path = tmp_path / "pipeline.jsonl"

    @stage("score", log_path=log_path)
    def scorer(frame: pd.DataFrame, ctx: RunContext) -> pd.DataFrame:
        return frame.assign(score=0.5)

    out = scorer(pd.DataFrame({"x": [1, 2, 3]}), ctx=ctx)
    assert len(out) == 3
    rows = _lines(log_path)
    assert rows[-1]["event"] == "stage_finished" and rows[-1]["n_rows"] == 3


def test_stage_decorator_requires_run_context(tmp_path):
    @stage("ingest", log_path=tmp_path / "x.jsonl")
    def no_ctx() -> None:
        return None

    with pytest.raises(TypeError):
        no_ctx()


def test_invalid_stage_or_status_rejected(ctx):
    with pytest.raises(ValueError):
        build_event(ctx, "not_a_stage", "x")
    with pytest.raises(ValueError):
        build_event(ctx, "ingest", "x", status="maybe")


def test_extra_never_carries_tabular_data_or_objects():
    frame = pd.DataFrame({"monthly_income": [12345.0]})
    clean = sanitize_extra({"frame": frame, "series": frame["monthly_income"],
                            "n": 3, "obj": object(), "nested": {"a": 1, "b": frame}})
    assert "12345" not in json.dumps(clean)
    assert clean["n"] == 3
    assert clean["frame"].startswith("<omitted")
    assert clean["obj"] == "<omitted>" and clean["nested"] == {"a": 1, "b": "<omitted: tabular data>"}


def test_extra_redacts_dataset_column_names_at_any_depth():
    row = {"customer_id": "C0000001", "age": 17, "monthly_income": 987654.0}
    clean = sanitize_extra({"first_row": row, "age": 17, "psi_monthly_income": 0.31,
                            "n_failed": 1})
    text = json.dumps(clean)
    assert "987654" not in text and "C0000001" not in text
    assert clean["age"] == REDACTED and clean["first_row"]["age"] == REDACTED
    assert clean["psi_monthly_income"] == 0.31 and clean["n_failed"] == 1  # agregados passam


def test_logger_does_not_duplicate_handlers(ctx, tmp_path):
    log_path = tmp_path / "pipeline.jsonl"
    first = get_logger("drift", ctx, log_path)
    second = get_logger("drift", ctx, log_path)
    assert first.logger is second.logger
    assert sum(isinstance(h, logging.Handler) for h in first.logger.handlers) == 1
    second.event("drift_detected", message="x")
    assert len(read_log(log_path)) == 1
