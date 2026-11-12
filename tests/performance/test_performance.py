"""Performance e carga: orçamentos de tempo bem abaixo do alerta A04 (120 s por execução)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from conftest import make_credit_frame

from creditwatch.contracts import validate_batch
from creditwatch.model.predict import score_batch
from creditwatch.simulation.generate import generate_scenario

pytestmark = pytest.mark.performance

ROOT = Path(__file__).resolve().parents[2]


def _elapsed(fn, *args, **kwargs) -> tuple[float, object]:
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - started, result


def test_validate_typical_batch_is_fast():
    batch = make_credit_frame(3_000).drop(columns="target")
    elapsed, result = _elapsed(validate_batch, batch, batch_id="perf_typical")

    assert result.status == "PASSED"
    assert elapsed < 5, f"validação de 3.000 linhas levou {elapsed:.2f}s"


def test_validate_large_batch_scales():
    batch = make_credit_frame(200_000).drop(columns="target")
    elapsed, result = _elapsed(validate_batch, batch, batch_id="perf_large")

    assert result.n_rows == 200_000
    assert elapsed < 30, f"validação de 200.000 linhas levou {elapsed:.2f}s"


def test_scoring_throughput(trained_model):
    model_path, _ = trained_model
    batch = make_credit_frame(100_000).drop(columns="target")
    elapsed, scored = _elapsed(score_batch, batch, model_path)

    assert len(scored) == 100_000
    assert elapsed < 20, f"scoring de 100.000 linhas levou {elapsed:.2f}s"


def test_severe_scenario_generation_time(production_pool, tmp_path):
    elapsed, paths = _elapsed(
        generate_scenario, production_pool, "covariate_severe",
        ROOT / "config" / "drift_scenarios.yaml", tmp_path,
    )

    assert len(paths) == 6
    assert elapsed < 60, f"geração do cenário severo levou {elapsed:.2f}s"


def test_repeated_validation_has_stable_latency():
    batch = make_credit_frame(3_000).drop(columns="target")
    timings = [_elapsed(validate_batch, batch, batch_id=f"perf_{i}")[0] for i in range(5)]

    assert max(timings) < 5
    assert max(timings) < 10 * min(timings)
