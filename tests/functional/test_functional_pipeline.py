"""Testes funcionais: fluxos de negócio ponta a ponta com dados sintéticos e modelo real."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
from conftest import FEATURES, make_credit_frame

from creditwatch.contracts import validate_batch
from creditwatch.drift.reports import generate_batch_report
from creditwatch.interfaces import (
    BatchRejected,
    DriftResult,
    DriftThresholds,
    FeatureDrift,
    PredictionDrift,
)
from creditwatch.model.evaluate import evaluate_labeled_batches, write_perf_vs_drift
from creditwatch.model.fairness import analyze_age_groups, render_report
from creditwatch.model.predict import score_batch
from creditwatch.observability import alerts as alert_engine
from creditwatch.observability.alerts import Alert
from creditwatch.pipeline import run as pipeline
from creditwatch.simulation.generate import generate_scenario

pytestmark = pytest.mark.functional

ROOT = Path(__file__).resolve().parents[2]


def test_valid_batch_passes_contract_and_is_scored(trained_model):
    model_path, _ = trained_model
    batch = make_credit_frame(1_500).drop(columns="target")

    validation = validate_batch(batch, batch_id="functional_valid")
    scored = score_batch(batch, model_path)

    assert validation.status == "PASSED"
    assert validation.n_rows_failed == 0
    assert len(scored) == len(batch)
    assert scored["score"].between(0, 1).all()
    assert set(scored["prediction"].unique()) <= {0, 1}
    assert (scored["model_version"] == "baseline_v1").all()


def test_model_ranks_better_than_random_on_holdout(trained_model):
    model_path, metadata_path = trained_model
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    holdout = make_credit_frame(3_000, seed=99)

    results = evaluate_labeled_batches(
        joblib.load(model_path), holdout, {}, FEATURES, "target", metadata["threshold"]
    )

    assert results[0]["roc_auc"] > 0.7


def test_scoring_is_deterministic(trained_model):
    model_path, _ = trained_model
    batch = make_credit_frame(500).drop(columns="target")

    first = score_batch(batch, model_path)
    second = score_batch(batch, model_path)

    pd.testing.assert_frame_equal(first, second)


def test_threshold_controls_prediction(trained_model, tmp_path):
    model_path, metadata_path = trained_model
    batch = make_credit_frame(800).drop(columns="target")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rates = {}
    for threshold in (0.0, 1.0):
        custom = tmp_path / f"meta_{threshold}.json"
        custom.write_text(json.dumps({**metadata, "threshold": threshold}), encoding="utf-8")
        rates[threshold] = score_batch(batch, model_path, custom)["prediction"].mean()

    assert rates[0.0] == 1.0
    assert rates[1.0] < 1.0


@pytest.mark.parametrize(
    ("scenario", "expected_batches"),
    [("no_drift", 4), ("covariate_moderate", 4), ("covariate_severe", 6)],
)
def test_generated_drift_batches_pass_contract_and_score(
    scenario, expected_batches, production_pool, trained_model, tmp_path
):
    model_path, _ = trained_model

    paths = generate_scenario(
        production_pool, scenario, ROOT / "config" / "drift_scenarios.yaml", tmp_path
    )

    assert len(paths) == expected_batches
    for path in paths:
        batch = pd.read_parquet(path).drop(columns=["event_date", "target"])
        assert validate_batch(batch, batch_id=path.stem).status != "BLOCKED"
        assert score_batch(batch, model_path)["score"].notna().all()


def test_severe_drift_moves_income_further_than_moderate(production_pool, tmp_path):
    config = ROOT / "config" / "drift_scenarios.yaml"
    moderate = generate_scenario(production_pool, "covariate_moderate", config, tmp_path)
    severe = generate_scenario(production_pool, "covariate_severe", config, tmp_path)
    base = production_pool["monthly_income"].mean()

    moderate_shift = pd.read_parquet(moderate[-1])["monthly_income"].mean() / base
    severe_shift = pd.read_parquet(severe[-1])["monthly_income"].mean() / base

    assert 1.0 < moderate_shift < severe_shift


def test_fairness_report_covers_age_bands(trained_model, tmp_path):
    model_path, _ = trained_model
    frame = make_credit_frame(3_000, seed=5)
    scored = score_batch(frame, model_path)

    result = analyze_age_groups(frame, scored)
    result["threshold"] = 0.3
    render_report(result, tmp_path / "bias.md")

    assert [group["age_band"] for group in result["groups"]] == ["18-29", "30-44", "45-59", "60+"]
    assert sum(group["n"] for group in result["groups"]) == len(frame)
    assert "Resultados por faixa" in (tmp_path / "bias.md").read_text(encoding="utf-8")


def test_perf_vs_drift_report_is_written(trained_model, tmp_path):
    model_path, _ = trained_model
    batches = {"covariate_severe_batch_01": make_credit_frame(1_000, seed=3)}
    results = evaluate_labeled_batches(
        joblib.load(model_path),
        make_credit_frame(1_000, seed=4),
        batches,
        FEATURES,
        "target",
        0.3,
        drift_by_batch={
            "covariate_severe_batch_01": {
                "drift_detected": True,
                "drift_share": 0.4,
                "score_psi": 0.15,
                "score_ks": 0.12,
                "mean_score_ref": 0.07,
                "mean_score_cur": 0.12,
            }
        },
    )
    report = tmp_path / "perf.md"

    write_perf_vs_drift(results, report)

    content = report.read_text(encoding="utf-8")
    assert "| covariate_severe | covariate_severe_batch_01 | sim |" in content


# --------------------------------------------------------------- pipeline run_batch
class Recorder:
    def __init__(self, result=None):
        self.calls: list[tuple] = []
        self.result = result

    def __call__(self, *args):
        self.calls.append(args)
        return self.result if self.result is not None else []


def _fake_drift(reference, current, thresholds, ctx):
    return DriftResult(
        run_id=ctx.run_id,
        batch_id=ctx.batch_id,
        scenario=ctx.scenario,
        reference_rows=len(reference),
        current_rows=len(current),
        dataset_drift=False,
        drift_share=0.0,
        features=[
            FeatureDrift(
                name=name, psi=0.01, ks_stat=0.02, ks_pvalue=0.9, drifted=False, severity="OK"
            )
            for name in FEATURES
        ],
        prediction_drift=PredictionDrift(
            score_psi=0.01, score_ks=0.02, mean_score_ref=0.1, mean_score_cur=0.1
        ),
        thresholds=thresholds,
    )


@pytest.fixture
def pipeline_workspace(tmp_path, monkeypatch, trained_model):
    """Diretório isolado com config, modelo e referência, como a CLI espera."""
    shutil.copytree(ROOT / "config", tmp_path / "config")
    model_path, metadata_path = trained_model
    (tmp_path / "models").mkdir()
    shutil.copy2(model_path, tmp_path / "models" / model_path.name)
    shutil.copy2(metadata_path, tmp_path / "models" / metadata_path.name)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    make_credit_frame(2_000, seed=11).to_parquet(processed / "reference_train.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "artifacts" / "logs" / "pipeline.jsonl")
    return tmp_path


def _components(ingestion):
    return {
        "ingestion": ingestion,
        "validation": validate_batch,
        "drift": _fake_drift,
        "report": generate_batch_report,
        "alerts": Recorder(),
        "metrics": Recorder(),
        "mlflow": Recorder(),
    }


def _log_events(workspace: Path) -> list[dict]:
    log = workspace / "artifacts" / "logs" / "pipeline.jsonl"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_run_batch_happy_path_scores_and_records_metrics(pipeline_workspace):
    batch_path = pipeline_workspace / "batch_ok.parquet"
    make_credit_frame(1_200, seed=21).drop(columns="target").to_parquet(batch_path, index=False)
    components = _components(lambda path, ctx: pd.read_parquet(path))

    code = pipeline.run_batch(batch_path, "no_drift", components)

    assert code == 0
    assert (pipeline_workspace / "artifacts" / "scored" / "no_drift" / "batch_ok.parquet").is_file()
    metrics = components["metrics"].calls[0][0]
    assert metrics.pipeline_status == "ok"
    assert metrics.rows_in == metrics.rows_valid == 1_200
    assert 0 <= metrics.score_mean <= 1
    stages = [event["stage"] for event in _log_events(pipeline_workspace)]
    assert stages == ["ingest", "validate", "score", "drift", "report", "pipeline", "alert"]

    report_dir = pipeline_workspace / "artifacts" / "drift"

    assert (report_dir / "batch_ok.html").is_file()
    assert (report_dir / "batch_ok.json").is_file()

    assert (report_dir / "batch_ok.html").stat().st_size > 0
    assert (report_dir / "batch_ok.json").stat().st_size > 0


def test_run_batch_blocks_invalid_batch_before_scoring(pipeline_workspace):
    batch_path = pipeline_workspace / "batch_bad.parquet"
    bad = make_credit_frame(1_200, seed=22).drop(columns="target")
    bad.loc[0, "age"] = 17
    bad.to_parquet(batch_path, index=False)
    components = _components(lambda path, ctx: pd.read_parquet(path))

    code = pipeline.run_batch(batch_path, "invalid", components)

    assert code == 2
    assert not (pipeline_workspace / "artifacts" / "scored").exists()
    metrics = components["metrics"].calls[0][0]
    assert metrics.pipeline_status == "blocked" and metrics.validation_blocked
    validation = components["alerts"].calls[0][0]
    assert validation.status == "BLOCKED"


def test_run_batch_handles_ingestion_rejection(pipeline_workspace):
    def reject(path, ctx):
        raise BatchRejected("schema inválido")

    components = _components(reject)

    code = pipeline.run_batch(pipeline_workspace / "rejected.parquet", "invalid", components)

    assert code == 2
    assert components["metrics"].calls[0][0].pipeline_status == "blocked"
    assert components["alerts"].calls[0][:2] == (None, None)


def test_thresholds_config_matches_contract_model(pipeline_workspace):
    thresholds = pipeline._thresholds()

    assert isinstance(thresholds, DriftThresholds)
    assert thresholds.psi_warn < thresholds.psi_critical


def test_run_batch_records_alert_counts_after_evaluation(
    pipeline_workspace,
):
    batch_path = pipeline_workspace / "batch_alerts.parquet"
    make_credit_frame(1_200, seed=23).drop(columns="target").to_parquet(batch_path, index=False)

    components = _components(lambda path, ctx: pd.read_parquet(path))

    def fake_alerts(validation, drift, metrics):
        metrics.n_alerts_warning = 2
        metrics.n_alerts_critical = 1
        return [
            object(),
            object(),
            object(),
        ]

    components["alerts"] = Recorder()
    components["alerts"].result = None

    def evaluate_and_update(validation, drift, metrics):
        metrics.n_alerts_warning = 2
        metrics.n_alerts_critical = 1

        return [
            Alert(
                run_id=metrics.run_id,
                batch_id=metrics.batch_id,
                severity="WARNING",
                rule_id="TEST_WARNING",
                message="Test warning alert",
            ),
            Alert(
                run_id=metrics.run_id,
                batch_id=metrics.batch_id,
                severity="CRITICAL",
                rule_id="TEST_CRITICAL",
                message="Test critical alert",
            ),
        ]

    components["alerts"] = evaluate_and_update

    code = pipeline.run_batch(
        batch_path,
        "no_drift",
        components,
    )

    assert code == 0

    recorded_metrics = components["metrics"].calls[0][0]

    assert recorded_metrics.n_alerts_warning == 2
    assert recorded_metrics.n_alerts_critical == 1


def test_run_batch_persists_real_alerts_for_blocked_batch(
    pipeline_workspace,
    monkeypatch,
):
    batch_path = pipeline_workspace / "batch_blocked.parquet"

    batch = make_credit_frame(1_200, seed=24).drop(columns="target")
    batch.loc[0, "age"] = 17
    batch.to_parquet(batch_path, index=False)

    alerts_log = pipeline_workspace / "artifacts" / "alerts" / "alerts.jsonl"

    monkeypatch.setattr(
        alert_engine,
        "ALERTS_LOG_PATH",
        alerts_log,
    )

    components = _components(lambda path, ctx: pd.read_parquet(path))

    # Utiliza o motor real, sem simular a avaliação dos alertas.
    components["alerts"] = alert_engine.evaluate_alerts

    code = pipeline.run_batch(
        batch_path,
        "invalid",
        components,
    )

    assert code == 2

    # Verifica a persistência do alerta.
    assert alerts_log.is_file()

    records = [json.loads(line) for line in alerts_log.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 1

    alert = records[0]

    assert alert["rule_id"] == "A01_VALIDATION_BLOCKED"
    assert alert["severity"] == "CRITICAL"
    assert alert["batch_id"] == "batch_blocked"

    # Verifica os contadores registrados pelo pipeline.
    metrics = components["metrics"].calls[0][0]

    assert metrics.validation_blocked is True
    assert metrics.n_alerts_critical == 1
    assert metrics.n_alerts_warning == 0
