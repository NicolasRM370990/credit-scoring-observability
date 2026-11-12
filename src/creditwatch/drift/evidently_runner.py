"""Execução do detector de drift do Evidently."""

from __future__ import annotations

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def run_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.1,
) -> dict[str, bool]:
    """Retorna o indicador de drift do Evidently para cada coluna numérica."""
    if not columns:
        raise ValueError("A lista de colunas não pode estar vazia")

    if len(columns) != len(set(columns)):
        raise ValueError("A lista de colunas contém duplicatas")

    if not 0 < threshold <= 1:
        raise ValueError("threshold deve estar no intervalo (0, 1]")

    if reference.empty or current.empty:
        raise ValueError("Os DataFrames de referência e atual não podem estar vazios")

    missing_reference = set(columns) - set(reference.columns)
    missing_current = set(columns) - set(current.columns)

    if missing_reference or missing_current:
        raise ValueError(
            "Colunas ausentes — "
            f"referência: {sorted(missing_reference)}, "
            f"atual: {sorted(missing_current)}"
        )

    for column in columns:
        if not pd.api.types.is_numeric_dtype(reference[column]):
            raise ValueError(f"Coluna não numérica na referência: {column}")
        if not pd.api.types.is_numeric_dtype(current[column]):
            raise ValueError(f"Coluna não numérica no lote atual: {column}")

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=columns,
                num_method="wasserstein",
                num_threshold=threshold,
            )
        ]
    )

    snapshot = report.run(
        current_data=current[columns],
        reference_data=reference[columns],
    )

    results: dict[str, bool] = {}

    for metric_result in snapshot.metric_results.values():
        location = getattr(metric_result, "metric_value_location", None)
        metric = getattr(location, "metric", None)
        params = getattr(metric, "params", {}) or {}

        if params.get("type") != "evidently:metric_v2:ValueDrift":
            continue

        column = params.get("column")
        if column not in columns:
            continue

        value = getattr(metric_result, "value", None)
        metric_threshold = params.get("threshold")

        if value is None or metric_threshold is None:
            raise RuntimeError(f"Resultado incompleto do Evidently para a coluna: {column}")

        results[column] = float(value) >= float(metric_threshold)

    missing_results = set(columns) - set(results)
    if missing_results:
        raise RuntimeError(f"Evidently não retornou resultados para: {sorted(missing_results)}")

    return results
