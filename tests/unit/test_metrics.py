"""Testes unitários da persistência de métricas."""

import json

import pytest

from creditwatch.interfaces import RunMetrics


@pytest.fixture
def metrics():
    return RunMetrics(
        run_id="run_test",
        batch_id="batch_test",
        scenario="no_drift",
        model_version="baseline_v1",
        pipeline_status="ok",
        rows_in=3000,
        rows_valid=2980,
        rows_rejected=20,
        validation_blocked=False,
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


@pytest.fixture
def metrics_log(tmp_path, monkeypatch):
    log_path = tmp_path / "artifacts" / "metrics" / "run_history.jsonl"

    from creditwatch.observability import metrics as metrics_module

    monkeypatch.setattr(
        metrics_module,
        "METRICS_LOG_PATH",
        log_path,
    )

    return log_path


def test_record_run_creates_jsonl_file(metrics, metrics_log):
    from creditwatch.observability.metrics import record_run

    record_run(metrics)

    assert metrics_log.is_file()


def test_record_run_persists_metrics_fields(metrics, metrics_log):
    from creditwatch.observability.metrics import record_run

    record_run(metrics)

    record = json.loads(metrics_log.read_text(encoding="utf-8").splitlines()[0])

    assert record["run_id"] == "run_test"
    assert record["batch_id"] == "batch_test"
    assert record["scenario"] == "no_drift"
    assert record["model_version"] == "baseline_v1"
    assert record["pipeline_status"] == "ok"


def test_record_run_persists_alert_counts(metrics, metrics_log):
    from creditwatch.observability.metrics import record_run

    record_run(metrics)

    record = json.loads(metrics_log.read_text(encoding="utf-8").splitlines()[0])

    assert record["n_alerts_warning"] == 2
    assert record["n_alerts_critical"] == 1


def test_record_run_preserves_nested_metrics(metrics, metrics_log):
    from creditwatch.observability.metrics import record_run

    record_run(metrics)

    record = json.loads(metrics_log.read_text(encoding="utf-8").splitlines()[0])

    assert record["latency_s"]["total"] == pytest.approx(4.2)
    assert record["psi"]["monthly_income"] == pytest.approx(0.15)
    assert record["ks"]["monthly_income"] == pytest.approx(0.12)


def test_record_run_appends_multiple_executions(metrics, metrics_log):
    from creditwatch.observability.metrics import record_run

    second_metrics = metrics.model_copy(
        update={
            "run_id": "run_test_2",
            "batch_id": "batch_test_2",
            "n_alerts_warning": 1,
            "n_alerts_critical": 0,
        }
    )

    record_run(metrics)
    record_run(second_metrics)

    records = [json.loads(line) for line in metrics_log.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 2
    assert records[0]["run_id"] == "run_test"
    assert records[1]["run_id"] == "run_test_2"


def test_record_run_does_not_modify_metrics(metrics, metrics_log):
    from creditwatch.observability.metrics import record_run

    original = metrics.model_dump(mode="json")

    record_run(metrics)

    assert metrics.model_dump(mode="json") == original
