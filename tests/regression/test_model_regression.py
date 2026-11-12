"""Regressão do modelo versionado e consistência entre arquivos de configuração."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from creditwatch.interfaces import CONTRACT_VERSION

pytestmark = pytest.mark.regression

ROOT = Path(__file__).resolve().parents[2]


def _yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((ROOT / "models" / "baseline_v1.json").read_text(encoding="utf-8"))


def test_test_auc_meets_configured_minimum(metadata):
    assert metadata["metrics"]["test"]["roc_auc"] >= _yaml("model")["min_auc_test"]


def test_model_beats_dummy_baseline(metadata):
    test, dummy = metadata["metrics"]["test"], metadata["metrics"]["dummy_test"]
    assert test["roc_auc"] > dummy["roc_auc"] + 0.2
    assert test["pr_auc"] > dummy["pr_auc"] * 3
    assert test["brier"] < dummy["brier"]


def test_validation_and_test_metrics_are_close(metadata):
    """Diferença grande entre validação e teste indica overfitting ou vazamento."""
    validation, test = metadata["metrics"]["validation"], metadata["metrics"]["test"]
    assert abs(validation["roc_auc"] - test["roc_auc"]) < 0.03
    assert abs(validation["ks"] - test["ks"]) < 0.05


def test_threshold_is_probability_and_matches_metrics(metadata):
    assert 0 < metadata["threshold"] < 1
    assert metadata["metrics"]["test"]["threshold"] == metadata["threshold"]


def test_confusion_matrix_is_consistent(metadata):
    test = metadata["metrics"]["test"]
    matrix = test["confusion_matrix"]
    assert sum(matrix.values()) == test["n_rows"] == metadata["splits"]["reference_test"]
    assert test["precision"] == pytest.approx(matrix["tp"] / (matrix["tp"] + matrix["fp"]))
    assert test["recall"] == pytest.approx(matrix["tp"] / (matrix["tp"] + matrix["fn"]))


def test_model_features_match_data_config(metadata):
    data = _yaml("data")
    assert metadata["features"] == data["features"]["numerical"] + data["features"]["categorical"]
    assert metadata["target"] == data["target"]
    assert metadata["target"] not in metadata["features"]


def test_model_was_trained_on_pinned_dataset(metadata):
    assert metadata["raw_sha256"] == _yaml("data")["dataset"]["sha256"]
    assert len(metadata["model_sha256"]) == 64


def test_split_sizes_follow_configured_fractions(metadata):
    split = _yaml("data")["split"]
    splits = metadata["splits"]
    reference = sum(splits.values())
    reference_share = 1 - split["production_pool"]
    assert splits["reference_train"] / reference == pytest.approx(
        split["reference_train"] / reference_share, abs=0.01
    )


def test_contract_required_columns_cover_model_features(metadata):
    contract = _yaml("contract")
    required = contract["block_rules"]["R04_SCHEMA"]["required_columns"]
    assert set(metadata["features"]) | {_yaml("data")["id_col"]} == set(required)


def test_contract_nullable_columns_match_special_codes():
    """Colunas que recebem NaN no prepare precisam ser anuláveis no contrato."""
    contract = _yaml("contract")
    nullable = set(contract["block_rules"]["R04_SCHEMA"]["nullable_columns"])
    assert set(_yaml("data")["special_codes"]["columns"]) <= nullable


def test_contract_version_matches_code():
    assert str(_yaml("contract")["version"]) == CONTRACT_VERSION


def test_alert_thresholds_match_drift_thresholds():
    alerts = _yaml("alerts")["rules"]
    drift = _yaml("drift_thresholds")
    assert alerts["A02_DATASET_DRIFT"]["threshold"] == drift["drift_share_alert"]
    assert alerts["A03_FEATURE_PSI"]["threshold"] == drift["psi_critical"]


def test_drift_scenarios_fit_in_production_pool(metadata):
    scenarios = _yaml("drift_scenarios")
    pool_share = _yaml("data")["split"]["production_pool"]
    reference_share = 1 - pool_share
    pool_rows = sum(metadata["splits"].values()) / reference_share * pool_share
    for name, scenario in scenarios["scenarios"].items():
        assert scenario["batches"] * scenarios["batch_size"] <= pool_rows, name


def test_scenario_batch_size_respects_volume_alert():
    volume = _yaml("alerts")["rules"]["A05_VOLUME"]
    assert volume["min"] <= _yaml("drift_scenarios")["batch_size"] <= volume["max"]
