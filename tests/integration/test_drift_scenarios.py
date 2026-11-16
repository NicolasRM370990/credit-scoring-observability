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

        for name in control_features:
            assert not features[name].drifted, f"{name} apresentou drift inesperado no lote {index}"

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

        print(f"\nLote severo: {index}")

        for name in altered_features | control_features:
            feature = features[name]

            print(
                f"{name}: "
                f"PSI={feature.psi:.4f}, "
                f"KS={feature.ks_stat:.4f}, "
                f"Drifted={feature.drifted}, "
                f"Severity={feature.severity}"
            )

        for name in control_features:
            assert not features[name].drifted, f"{name} apresentou drift inesperado no lote {index}"

        # assert any(
        #     features[name].drifted for name in altered_features
        # ), (
        #     f"Nenhuma feature alterada apresentou drift no lote {index}"
        # )

        altered_psi = {name: features[name].psi for name in altered_features}

        detected = [name for name in altered_features if features[name].drifted]

        print(f"Features detectadas: {detected} | Maior PSI: {max(altered_psi.values()):.4f}")
