"""Fixtures sintéticos compartilhados; testes unitários não dependem de dados locais."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

FEATURES = [
    "revolving_utilization",
    "age",
    "past_due_30_59",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "times_90_days_late",
    "real_estate_loans",
    "past_due_60_89",
    "dependents",
]


@pytest.fixture
def production_pool() -> pd.DataFrame:
    """Pool válido grande o bastante para os cenários de drift configurados."""
    n_rows = 18_000
    row = np.arange(n_rows)
    return pd.DataFrame(
        {
            "customer_id": [f"C{index:07d}" for index in row],
            "revolving_utilization": (row % 280) / 100,
            "age": 18 + row % 80,
            "past_due_30_59": row % 5,
            "debt_ratio": (row % 700) / 100,
            "monthly_income": 1_000 + (row % 240) * 100,
            "open_credit_lines": row % 25,
            "times_90_days_late": row % 4,
            "real_estate_loans": row % 5,
            "past_due_60_89": row % 4,
            "dependents": row % 6,
            "target": (row % 11 == 0).astype("int8"),
        }
    )


def make_credit_frame(n_rows: int, seed: int = 42, prefix: str = "C") -> pd.DataFrame:
    """Lote sintético dentro do contrato, com target dependente das features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "customer_id": [f"{prefix}{index:07d}" for index in range(n_rows)],
            "revolving_utilization": rng.uniform(0, 1.5, n_rows),
            "age": rng.integers(18, 95, n_rows),
            "past_due_30_59": rng.poisson(0.4, n_rows),
            "debt_ratio": rng.uniform(0, 3, n_rows),
            "monthly_income": rng.uniform(500, 25_000, n_rows),
            "open_credit_lines": rng.integers(0, 30, n_rows),
            "times_90_days_late": rng.poisson(0.2, n_rows),
            "real_estate_loans": rng.integers(0, 5, n_rows),
            "past_due_60_89": rng.poisson(0.2, n_rows),
            "dependents": rng.integers(0, 6, n_rows),
        }
    )
    logit = (
        -3.0
        + 1.6 * frame["revolving_utilization"]
        + 0.9 * frame["times_90_days_late"]
        + 0.6 * frame["past_due_30_59"]
        - 0.02 * (frame["age"] - 50)
    )
    frame["target"] = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-logit))).astype("int8")
    return frame


@pytest.fixture
def credit_frame() -> pd.DataFrame:
    return make_credit_frame(2_000)


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory) -> tuple[Path, Path]:
    """Modelo real treinado em dados sintéticos, com metadata no formato do treino."""
    out = tmp_path_factory.mktemp("model")
    train = make_credit_frame(6_000, seed=7)
    estimator = HistGradientBoostingClassifier(max_iter=60, random_state=42)
    estimator.fit(train[FEATURES], train["target"])
    model_path = out / "baseline_v1.joblib"
    metadata_path = out / "baseline_v1.json"
    joblib.dump(estimator, model_path)
    metadata = {
        "model_version": "baseline_v1",
        "features": FEATURES,
        "target": "target",
        "threshold": 0.3,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, metadata_path
