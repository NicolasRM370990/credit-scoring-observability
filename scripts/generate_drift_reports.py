
"""Gera relatórios Evidently e gráfico PSI para um cenário."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from creditwatch.drift.detector import detect_drift
from creditwatch.drift.reports import (
    generate_batch_report,
    generate_psi_chart,
)
from creditwatch.interfaces import DriftThresholds, RunContext
from creditwatch.model.predict import score_batch

ROOT = Path(".")
SCENARIO = "covariate_severe"

REFERENCE_PATH = ROOT / "data/processed/reference_train.parquet"
MODEL_PATH = ROOT / "models/baseline_v1.joblib"
THRESHOLDS_PATH = ROOT / "config/drift_thresholds.yaml"

BATCHES_DIR = ROOT / "artifacts/scored" / SCENARIO
OUTPUT_DIR = ROOT / "artifacts/drift" / SCENARIO


def main() -> None:
    """Gera relatórios para todos os lotes pontuados do cenário."""
    for path in (REFERENCE_PATH, MODEL_PATH, THRESHOLDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    batch_paths = sorted(BATCHES_DIR.glob("batch_*_scored.parquet"))

    if not batch_paths:
        raise FileNotFoundError(
            f"Nenhum lote pontuado encontrado em {BATCHES_DIR}"
        )

    print("Carregando referência...")
    reference_raw = pd.read_parquet(REFERENCE_PATH)

    print("Pontuando referência com baseline_v1...")
    reference = score_batch(reference_raw, MODEL_PATH)

    thresholds_data = yaml.safe_load(
        THRESHOLDS_PATH.read_text(encoding="utf-8")
    )
    thresholds = DriftThresholds(**thresholds_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for batch_path in batch_paths:
        batch_id = batch_path.name.replace("_scored.parquet", "")

        print(f"\nProcessando {batch_id}...")

        current = pd.read_parquet(batch_path)

        context = RunContext(
            run_id=f"report_{SCENARIO}_{batch_id}",
            batch_id=batch_id,
            scenario=SCENARIO,
        )

        drift_result = detect_drift(
            reference=reference,
            current=current,
            cfg=thresholds,
            ctx=context,
        )

        updated_result = generate_batch_report(
            reference=reference,
            current=current,
            result=drift_result,
            output_dir=OUTPUT_DIR,
        )

        results.append(updated_result)

        print(
            f"  Dataset drift: {updated_result.dataset_drift}"
            f"\n  Drift share: {updated_result.drift_share:.2%}"
            f"\n  HTML: {updated_result.report_html}"
            f"\n  JSON: {OUTPUT_DIR / f'{batch_id}.json'}"
        )

    chart_path = generate_psi_chart(
        results=results,
        output_path=OUTPUT_DIR / "psi_evolution.png",
    )

    print("\n=== Geração concluída ===")
    print(f"Lotes processados: {len(results)}")
    print(f"Diretório: {OUTPUT_DIR}")
    print(f"Gráfico PSI: {chart_path}")


if __name__ == "__main__":
    main()
