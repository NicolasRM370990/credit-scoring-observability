from __future__ import annotations

import math
from numbers import Real

import pandas as pd


def apply_scale(
    df: pd.DataFrame,
    feature: str,
    factor: float,
) -> pd.DataFrame:
    """Multiplica uma coluna numérica por um fator, sem alterar o DataFrame original.

    Args:
        df: DataFrame de entrada.
        feature: Nome da coluna que será transformada.
        factor: Fator de multiplicação.

    Returns:
        Uma cópia do DataFrame com a coluna transformada.

    Raises:
        TypeError: Se df não for DataFrame ou factor não for numérico.
        ValueError: Se a coluna não existir, não for numérica ou
            se factor não for finito.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    if feature not in df.columns:
        raise ValueError(f"A coluna '{feature}' não existe no DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise ValueError(f"A coluna '{feature}' deve possuir valores numéricos.")

    if isinstance(factor, bool) or not isinstance(factor, Real):
        raise TypeError("factor deve ser um número real.")

    if not math.isfinite(float(factor)):
        raise ValueError("factor deve ser um número finito.")

    result = df.copy(deep=True)
    result[feature] = result[feature] * factor

    return result
