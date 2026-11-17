"""Calibra thresholds de drift usando comparações sem transformação."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from creditwatch.drift.stats import ks_test, psi

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REFERENCE = ROOT / "data/processed/reference_train.parquet"
DEFAULT_POOL = ROOT / "data/processed/production_pool.parquet"
DEFAULT_DATA_CONFIG = ROOT / "config/data.yaml"
DEFAULT_OUTPUT = ROOT / "docs/evidence/drift_threshold_calibration.csv"

DEFAULT_COMPARISONS = 200
DEFAULT_SAMPLE_SIZE = 3000
DEFAULT_SEED = 42


def load_features(config_path: Path) -> list[str]:
    """Carrega as features numéricas da configuração de dados."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    features = config["features"]["numerical"]

    if not features:
        raise ValueError("Nenhuma feature numérica configurada.")

    return features


def calibrate(
    reference: pd.DataFrame,
    pool: pd.DataFrame,
    features: list[str],
    n_comparisons: int,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Calcula PSI e KS em comparações independentes sem reposição interna."""
    if n_comparisons < 200:
        raise ValueError("A calibração exige pelo menos 200 comparações.")

    if sample_size <= 0:
        raise ValueError("sample_size deve ser maior que zero.")

    if sample_size > len(reference) or sample_size > len(pool):
        raise ValueError("Amostras insuficientes para o tamanho solicitado.")

    for label, frame in (("reference", reference), ("pool", pool)):
        missing = set(features) - set(frame.columns)
        if missing:
            raise ValueError(f"Features ausentes em {label}: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    records = []

    for comparison in range(1, n_comparisons + 1):
        reference_indices = rng.choice(
            len(reference),
            size=sample_size,
            replace=False,
        )
        pool_indices = rng.choice(
            len(pool),
            size=sample_size,
            replace=False,
        )

        reference_sample = reference.iloc[reference_indices]
        pool_sample = pool.iloc[pool_indices]

        comparison_results = []

        for feature in features:
            psi_value = psi(
                reference_sample[feature],
                pool_sample[feature],
            )
            ks_stat, _ = ks_test(
                reference_sample[feature],
                pool_sample[feature],
            )

            comparison_results.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "psi": psi_value,
                    "ks_stat": ks_stat,
                }
            )

        max_psi = max(item["psi"] for item in comparison_results)
        max_ks = max(item["ks_stat"] for item in comparison_results)

        for item in comparison_results:
            item["max_psi_comparison"] = max_psi
            item["max_ks_comparison"] = max_ks
            records.append(item)

    return pd.DataFrame(records)


def summarize(results: pd.DataFrame) -> None:
    """Exibe os percentis das distribuições dos máximos por comparação."""
    maxima = results.drop_duplicates(subset=["comparison"])

    print("\n=== CALIBRAÇÃO DE DRIFT ===")
    print(f"Comparações: {len(maxima)}")
    print(f"Features: {results['feature'].nunique()}")

    print("\nDistribuição do maior PSI por comparação:")
    print(maxima["max_psi_comparison"].describe(percentiles=[0.95, 0.99]))

    print("\nDistribuição do maior KS por comparação:")
    print(maxima["max_ks_comparison"].describe(percentiles=[0.95, 0.99]))

    print("\nThresholds candidatos:")
    print(f"psi_warn (P95): {maxima['max_psi_comparison'].quantile(0.95):.6f}")
    print(f"psi_critical (P99): {maxima['max_psi_comparison'].quantile(0.99):.6f}")
    print(f"ks_stat_min (P95): {maxima['max_ks_comparison'].quantile(0.95):.6f}")

    psi_warn = maxima["max_psi_comparison"].quantile(0.95)
    ks_stat_min = maxima["max_ks_comparison"].quantile(0.95)

    feature_results = results.assign(
        drifted=(results["psi"] >= psi_warn) | (results["ks_stat"] >= ks_stat_min)
    )

    drift_share = feature_results.groupby("comparison")["drifted"].mean()

    print("\nDrift share nas comparações sem transformação:")
    print(drift_share.describe(percentiles=[0.95, 0.99]))
    print(f"Comparações com drift_share >= 0.30: {(drift_share >= 0.30).sum()}")
    print(f"Maior drift_share observado: {drift_share.max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra thresholds de PSI e KS sem drift.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comparisons", type=int, default=DEFAULT_COMPARISONS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()

    for path in (args.reference, args.pool, args.data_config):
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    features = load_features(args.data_config)

    print("Carregando datasets...")
    reference = pd.read_parquet(args.reference)
    pool = pd.read_parquet(args.pool)

    print("Executando calibração...")
    results = calibrate(
        reference=reference,
        pool=pool,
        features=features,
        n_comparisons=args.comparisons,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, float_format="%.10f")

    summarize(results)

    print(f"\nEvidência salva em: {args.output}")
    print(f"Linhas geradas: {len(results)}")


if __name__ == "__main__":
    main()
