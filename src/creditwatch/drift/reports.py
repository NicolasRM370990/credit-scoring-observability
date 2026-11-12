"""Geração de relatórios de drift."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from creditwatch.interfaces import DriftResult


def generate_batch_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    result: DriftResult,
    output_dir: Path,
) -> DriftResult:
    """Gera relatório HTML do Evidently e resumo JSON de um lote."""
    if reference.empty or current.empty:
        raise ValueError("Os DataFrames não podem estar vazios")

    if not result.features:
        raise ValueError("O resultado não possui features para gerar relatório")

    features = [feature.name for feature in result.features]

    columns = features + (["score"] if "score" in reference.columns else [])

    missing_reference = set(columns) - set(reference.columns)
    missing_current = set(columns) - set(current.columns)

    if missing_reference or missing_current:
        raise ValueError(
            "Colunas ausentes — "
            f"referência: {sorted(missing_reference)}, "
            f"atual: {sorted(missing_current)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=columns,
                num_method="wasserstein",
                num_threshold=0.1,
            )
        ]
    )

    snapshot = report.run(
        current_data=current[columns],
        reference_data=reference[columns],
    )

    html_path = output_dir / f"{result.batch_id}.html"
    json_path = output_dir / f"{result.batch_id}.json"

    snapshot.save_html(str(html_path))

    updated_result = result.model_copy(
        update={"report_html": str(html_path)}
    )

    json_path.write_text(
        updated_result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return updated_result


def generate_psi_chart(
    results: list[DriftResult],
    output_path: Path,
) -> Path:
    """Gera gráfico de PSI por feature e por lote."""
    if not results:
        raise ValueError("A lista de resultados não pode estar vazia")

    batch_ids = [result.batch_id for result in results]

    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("Existem batch_ids duplicados nos resultados")

    features = [feature.name for feature in results[0].features]

    if not features:
        raise ValueError("Os resultados não possuem features")

    expected_features = set(features)

    for result in results:
        current_features = {feature.name for feature in result.features}

        if current_features != expected_features:
            raise ValueError(
                f"Features inconsistentes no lote: {result.batch_id}"
            )

        if any(
            not math.isfinite(feature.psi)
            for feature in result.features
        ):
            raise ValueError(
                f"Valor PSI não finito no lote: {result.batch_id}"
            )

    thresholds = results[0].thresholds

    for result in results[1:]:
        if result.thresholds != thresholds:
            raise ValueError("Os limiares de PSI devem ser iguais em todos os lotes")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_positions = np.arange(len(results))
    feature_count = len(features)
    bar_width = 0.8 / feature_count

    fig, ax = plt.subplots(
        figsize=(max(10, len(results) * 1.5), 6)
    )

    try:
        for index, feature_name in enumerate(features):
            values = [
                next(
                    feature.psi
                    for feature in result.features
                    if feature.name == feature_name
                )
                for result in results
            ]

            offset = (
                index - (feature_count - 1) / 2
            ) * bar_width

            ax.bar(
                batch_positions + offset,
                values,
                width=bar_width,
                label=feature_name,
            )

        ax.axhline(
            thresholds.psi_warn,
            linestyle="--",
            label="PSI Warning",
        )

        ax.axhline(
            thresholds.psi_critical,
            linestyle="--",
            label="PSI Critical",
        )

        ax.set_title("Evolução do PSI por Feature e Lote")
        ax.set_xlabel("Lotes")
        ax.set_ylabel("Population Stability Index (PSI)")
        ax.set_xticks(batch_positions)
        ax.set_xticklabels(batch_ids)
        ax.legend(title="Features e limiares")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    finally:
        plt.close(fig)

    return output_path
