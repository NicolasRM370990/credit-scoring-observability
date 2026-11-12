"""Detector de data drift e prediction drift."""

from __future__ import annotations

import pandas as pd

from creditwatch.drift.evidently_runner import run_evidently_drift
from creditwatch.drift.stats import ks_test, psi
from creditwatch.interfaces import (
    DriftResult,
    DriftThresholds,
    FeatureDrift,
    PredictionDrift,
    RunContext,
)

EXCLUDED_COLUMNS = {
    "customer_id",
    "target",
    "event_date",
    "score",
    "prediction",
    "model_version",
}


def _get_feature_columns(reference: pd.DataFrame) -> list[str]:
    """Identifica features numéricas, excluindo colunas auxiliares."""
    return [
        column
        for column in reference.select_dtypes(include="number").columns
        if column not in EXCLUDED_COLUMNS
    ]


def _validate_thresholds(cfg: DriftThresholds) -> None:
    if not 0 <= cfg.psi_warn < cfg.psi_critical:
        raise ValueError("psi_warn deve ser menor que psi_critical")

    if not 0 <= cfg.ks_stat_min <= 1:
        raise ValueError("ks_stat_min deve estar entre zero e um")

    if not 0 <= cfg.drift_share_alert <= 1:
        raise ValueError("drift_share_alert deve estar entre zero e um")


def _validate_data(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> None:
    if reference.empty or current.empty:
        raise ValueError("Os DataFrames não podem estar vazios")

    required = set(features) | {"score"}

    missing_reference = required - set(reference.columns)
    missing_current = required - set(current.columns)

    if missing_reference or missing_current:
        raise ValueError(
            "Colunas obrigatórias ausentes — "
            f"referência: {sorted(missing_reference)}, "
            f"atual: {sorted(missing_current)}"
        )

    for column in required:
        if not pd.api.types.is_numeric_dtype(reference[column]):
            raise ValueError(f"Coluna não numérica na referência: {column}")

        if not pd.api.types.is_numeric_dtype(current[column]):
            raise ValueError(f"Coluna não numérica no lote atual: {column}")

    for dataframe, label in (
        (reference, "referência"),
        (current, "lote atual"),
    ):    
        for column in required:
            values = dataframe[column]

            if not values.notna().all():
                raise ValueError(
                    f"Valores ausentes na coluna {column} ({label})"
                )

            if not values.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all():
                raise ValueError(
                    f"Valores não finitos na coluna {column} ({label})"
                )

def _classify_severity(
    psi_value: float,
    ks_stat: float,
    cfg: DriftThresholds,
) -> tuple[bool, str]:
    if psi_value >= cfg.psi_critical:
        return True, "CRITICAL"

    if psi_value >= cfg.psi_warn or ks_stat >= cfg.ks_stat_min:
        return True, "WARNING"

    return False, "OK"


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: DriftThresholds,
    ctx: RunContext,
) -> DriftResult:
    """Calcula drift de features e scores e retorna DriftResult."""
    _validate_thresholds(cfg)

    features = _get_feature_columns(reference)

    if not features:
        raise ValueError("Nenhuma feature numérica encontrada")

    _validate_data(reference, current, features)

    evidently_results = run_evidently_drift(
        reference=reference,
        current=current,
        columns=features + ["score"],
    )

    feature_results: list[FeatureDrift] = []

    for feature in features:
        psi_value = psi(reference[feature], current[feature])
        ks_stat, ks_pvalue = ks_test(
            reference[feature],
            current[feature],
        )

        drifted, severity = _classify_severity(
            psi_value,
            ks_stat,
            cfg,
        )

        feature_results.append(
            FeatureDrift(
                name=feature,
                psi=psi_value,
                ks_stat=ks_stat,
                ks_pvalue=ks_pvalue,
                evidently_drifted=evidently_results[feature],
                drifted=drifted,
                severity=severity,
            )
        )

    drift_count = sum(feature.drifted for feature in feature_results)
    drift_share = drift_count / len(feature_results)

    score_psi = psi(reference["score"], current["score"])
    score_ks, _ = ks_test(reference["score"], current["score"])

    prediction_drift = PredictionDrift(
        score_psi=score_psi,
        score_ks=score_ks,
        mean_score_ref=float(reference["score"].mean()),
        mean_score_cur=float(current["score"].mean()),
    )

    return DriftResult(
        run_id=ctx.run_id,
        batch_id=ctx.batch_id or "unknown",
        scenario=ctx.scenario,
        reference_rows=len(reference),
        current_rows=len(current),
        dataset_drift=drift_share >= cfg.drift_share_alert,
        drift_share=drift_share,
        features=feature_results,
        prediction_drift=prediction_drift,
        thresholds=cfg,
    )
