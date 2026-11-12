
"""Estatísticas para monitoramento de data drift."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calcula o Population Stability Index entre duas distribuições."""
    if bins < 1:
        raise ValueError("'bins' deve ser maior que zero.")

    if not 0 < epsilon < 1:
        raise ValueError("'epsilon' deve estar entre zero e um.")

    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("As amostras não podem estar vazias.")

    if not np.isfinite(reference_values).all():
        raise ValueError("A referência contém valores não finitos.")

    if not np.isfinite(current_values).all():
        raise ValueError("A amostra atual contém valores não finitos.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))

    if edges.size < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    reference_pct = reference_counts / reference_values.size
    current_pct = current_counts / current_values.size

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def ks_test(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[float, float]:
    """Executa o teste Kolmogorov-Smirnov para duas amostras."""
    reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current_values = pd.to_numeric(current, errors="coerce").dropna().to_numpy()

    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("As amostras não podem estar vazias.")

    if not np.isfinite(reference_values).all():
        raise ValueError("A referência contém valores não finitos.")

    if not np.isfinite(current_values).all():
        raise ValueError("A amostra atual contém valores não finitos.")

    result = ks_2samp(
        reference_values,
        current_values,
        method="asymp",
    )

    return float(result.statistic), float(result.pvalue)
