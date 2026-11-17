from itertools import pairwise

import pandas as pd
import yaml

from creditwatch.drift.detector import detect_drift
from creditwatch.interfaces import DriftThresholds, RunContext
from creditwatch.simulation import generate_scenario

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


def load_thresholds():
    with open("config/drift_thresholds.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return DriftThresholds(**config)


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df[FEATURES].copy()
    result["score"] = 0.5
    return result


def test_no_drift_does_not_generate_drift_alert(
    tmp_path,
    production_pool,
):
    generated = generate_scenario(
        pool=production_pool,
        scenario="no_drift",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path,
    )

    reference = add_score(production_pool)
    thresholds = load_thresholds()

    assert len(generated) == 4

    for index, path in enumerate(generated, start=1):
        current_batch = pd.read_parquet(path)
        current = add_score(current_batch)

        result = detect_drift(
            reference=reference,
            current=current,
            cfg=thresholds,
            ctx=RunContext.new(
                batch_id=f"no_drift_batch_{index:02d}",
                scenario="no_drift",
            ),
        )

        critical_features = [
            feature.name for feature in result.features if feature.severity == "CRITICAL"
        ]

        assert not critical_features, f"Features críticas no lote {index}: {critical_features}"

        assert result.dataset_drift is False, (
            f"Dataset drift inesperado no lote {index}: drift_share={result.drift_share}"
        )


def test_covariate_moderate_detects_changed_features(
    tmp_path,
    production_pool,
):
    generated = generate_scenario(
        pool=production_pool,
        scenario="covariate_moderate",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path,
    )

    reference = add_score(production_pool)
    thresholds = load_thresholds()

    altered_features = {
        "monthly_income",
        "revolving_utilization",
    }
    control_features = {"age", "dependents"}

    psi_history = {name: [] for name in altered_features}
    last_features = {}

    assert len(generated) == 4

    for index, path in enumerate(generated, start=1):
        current_batch = pd.read_parquet(path)
        current = add_score(current_batch)

        result = detect_drift(
            reference=reference,
            current=current,
            cfg=thresholds,
            ctx=RunContext.new(
                batch_id=f"covariate_moderate_batch_{index:02d}",
                scenario="covariate_moderate",
            ),
        )

        features = {feature.name: feature for feature in result.features}
        last_features = features

        for name in altered_features:
            psi_history[name].append(features[name].psi)

        # Features de controle não devem ser detectadas como drift.
        for name in control_features:
            assert not features[name].drifted, f"{name} apresentou drift inesperado no lote {index}"

        # PSI médio das features alteradas deve superar o das controles.
        altered_psi_mean = sum(features[name].psi for name in altered_features) / len(
            altered_features
        )

        control_psi_mean = sum(features[name].psi for name in control_features) / len(
            control_features
        )

        assert altered_psi_mean > control_psi_mean, (
            f"Lote {index}: PSI médio das alteradas "
            f"({altered_psi_mean:.4f}) não superou o das controles "
            f"({control_psi_mean:.4f})"
        )

    # As features alteradas devem apresentar aumento de PSI.
    for name, values in psi_history.items():
        assert values[-1] > values[0], f"{name} não apresentou aumento de PSI: {values}"

    # Ao menos uma feature alterada deve ser detectada no último lote.
    assert any(last_features[name].drifted for name in altered_features), (
        "Nenhuma feature alterada apresentou drift no último lote."
    )


def test_covariate_severe_detects_stronger_drift(
    tmp_path,
    production_pool,
):
    generated = generate_scenario(
        pool=production_pool,
        scenario="covariate_severe",
        config_path="config/drift_scenarios.yaml",
        out_dir=tmp_path,
    )

    reference = add_score(production_pool)
    thresholds = load_thresholds()

    altered_features = {
        "monthly_income",
        "revolving_utilization",
        "debt_ratio",
    }
    control_features = {"age", "dependents"}

    psi_history = {name: [] for name in altered_features}
    detected_by_batch = []

    assert len(generated) == 6

    for index, path in enumerate(generated, start=1):
        current_batch = pd.read_parquet(path)
        current = add_score(current_batch)

        result = detect_drift(
            reference=reference,
            current=current,
            cfg=thresholds,
            ctx=RunContext.new(
                batch_id=f"covariate_severe_batch_{index:02d}",
                scenario="covariate_severe",
            ),
        )

        features = {feature.name: feature for feature in result.features}

        for name in altered_features:
            psi_history[name].append(features[name].psi)

        for name in control_features:
            assert not features[name].drifted, f"{name} apresentou drift inesperado no lote {index}"

        detected = {name for name in altered_features if features[name].drifted}
        detected_by_batch.append(detected)

    # PSI deve crescer progressivamente nas features alteradas.
    for name, values in psi_history.items():
        assert all(
            current_value > previous_value for previous_value, current_value in pairwise(values)
        ), f"{name} não apresentou crescimento progressivo de PSI: {values}"

    # Lote 1: detecção parcial, sem atingir todas as features alteradas.
    assert 0 < len(detected_by_batch[0]) < len(altered_features)

    # Lotes seguintes: todas as features alteradas devem ser detectadas.
    for index, detected in enumerate(detected_by_batch[1:], start=2):
        assert detected == altered_features, (
            f"Lote {index} não detectou todas as features alteradas: {detected}"
        )

    # Lotes 2 e 3: ao menos uma feature alterada detectada.
    for index in (1, 2):
        assert detected_by_batch[index], f"Nenhuma feature alterada detectada no lote {index + 1}"

    # Lotes 4, 5 e 6: todas as features alteradas detectadas.
    for index in (3, 4, 5):
        assert detected_by_batch[index] == altered_features, (
            f"Detecção incompleta no lote {index + 1}: {detected_by_batch[index]}"
        )
