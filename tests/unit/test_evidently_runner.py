
from types import SimpleNamespace

import pandas as pd
import pytest

from creditwatch.drift import evidently_runner
from creditwatch.drift.evidently_runner import run_evidently_drift


def _metric_result(column: str, value: float, threshold: float = 0.1):
    params = {
        "type": "evidently:metric_v2:ValueDrift",
        "column": column,
        "method": "Wasserstein distance (normed)",
        "threshold": threshold,
    }

    return SimpleNamespace(
        metric_value_location=SimpleNamespace(
            metric=SimpleNamespace(params=params)
        ),
        value=value,
    )


def _frames():
    reference = pd.DataFrame({"feature_a": [1, 2, 3]})
    current = pd.DataFrame({"feature_a": [1, 2, 4]})
    return reference, current


def test_returns_false_when_value_is_below_threshold(monkeypatch):
    reference, current = _frames()

    snapshot = SimpleNamespace(
        metric_results={"metric_1": _metric_result("feature_a", 0.05)}
    )

    class FakeReport:
        def __init__(self, metrics):
            pass

        def run(self, current_data, reference_data):
            return snapshot

    monkeypatch.setattr(evidently_runner, "Report", FakeReport)

    result = run_evidently_drift(
        reference, current, ["feature_a"]
    )

    assert result == {"feature_a": False}


def test_returns_true_when_value_reaches_threshold(monkeypatch):
    reference, current = _frames()

    snapshot = SimpleNamespace(
        metric_results={"metric_1": _metric_result("feature_a", 0.1)}
    )

    class FakeReport:
        def __init__(self, metrics):
            pass

        def run(self, current_data, reference_data):
            return snapshot

    monkeypatch.setattr(evidently_runner, "Report", FakeReport)

    result = run_evidently_drift(
        reference, current, ["feature_a"]
    )

    assert result == {"feature_a": True}


def test_ignores_non_value_drift_metrics(monkeypatch):
    reference, current = _frames()

    snapshot = SimpleNamespace(
        metric_results={
            "count": SimpleNamespace(
                metric_value_location=SimpleNamespace(
                    metric=SimpleNamespace(
                        params={"type": "evidently:metric_v2:DriftedColumnsCount"}
                    )
                ),
                value=None,
            ),
            "feature": _metric_result("feature_a", 0.02),
        }
    )

    class FakeReport:
        def __init__(self, metrics):
            pass

        def run(self, current_data, reference_data):
            return snapshot

    monkeypatch.setattr(evidently_runner, "Report", FakeReport)

    result = run_evidently_drift(
        reference, current, ["feature_a"]
    )

    assert result == {"feature_a": False}


@pytest.mark.parametrize(
    ("columns", "threshold", "message"),
    [
        ([], 0.1, "não pode estar vazia"),
        (["feature_a", "feature_a"], 0.1, "duplicatas"),
        (["feature_a"], 0, "threshold"),
    ],
)
def test_rejects_invalid_arguments(columns, threshold, message):
    reference, current = _frames()

    with pytest.raises(ValueError, match=message):
        run_evidently_drift(
            reference, current, columns, threshold
        )


def test_rejects_missing_columns():
    reference, current = _frames()

    with pytest.raises(ValueError, match="Colunas ausentes"):
        run_evidently_drift(
            reference, current, ["feature_missing"]
        )


def test_rejects_non_numeric_columns():
    reference = pd.DataFrame({"feature_a": ["a", "b"]})
    current = pd.DataFrame({"feature_a": ["a", "c"]})

    with pytest.raises(ValueError, match="não numérica"):
        run_evidently_drift(
            reference, current, ["feature_a"]
        )


def test_rejects_empty_dataframes():
    reference = pd.DataFrame({"feature_a": []})
    current = pd.DataFrame({"feature_a": [1.0]})

    with pytest.raises(ValueError, match="não podem estar vazios"):
        run_evidently_drift(
            reference, current, ["feature_a"]
        )


def test_rejects_missing_evidently_result(monkeypatch):
    reference, current = _frames()

    snapshot = SimpleNamespace(metric_results={})

    class FakeReport:
        def __init__(self, metrics):
            pass

        def run(self, current_data, reference_data):
            return snapshot

    monkeypatch.setattr(evidently_runner, "Report", FakeReport)

    with pytest.raises(RuntimeError, match="não retornou resultados"):
        run_evidently_drift(
            reference, current, ["feature_a"]
        )
