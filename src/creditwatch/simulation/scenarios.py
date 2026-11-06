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


def apply_shift_add(
    df: pd.DataFrame,
    feature: str,
    value: float,
    clip: tuple[float, float] | list[float] | None = None,
) -> pd.DataFrame:
    """Adiciona um valor a uma coluna numérica, com limites opcionais.

    Args:
        df: DataFrame de entrada.
        feature: Nome da coluna que será transformada.
        value: Valor que será adicionado.
        clip: Limites opcionais (mínimo, máximo).

    Returns:
        Uma cópia do DataFrame com a coluna transformada.

    Raises:
        TypeError: Se os tipos dos argumentos forem inválidos.
        ValueError: Se a coluna, o valor ou os limites forem inválidos.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    if feature not in df.columns:
        raise ValueError(f"A coluna '{feature}' não existe no DataFrame.")

    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise ValueError(f"A coluna '{feature}' deve possuir valores numéricos.")

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value deve ser um número real.")

    if not math.isfinite(float(value)):
        raise ValueError("value deve ser um número finito.")

    if clip is not None:
        if not isinstance(clip, (tuple, list)) or len(clip) != 2:
            raise ValueError("clip deve conter exatamente dois limites.")

        lower, upper = clip

        for limit in clip:
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise TypeError("Os limites de clip devem ser números reais.")

            if not math.isfinite(float(limit)):
                raise ValueError("Os limites de clip devem ser finitos.")

        if lower > upper:
            raise ValueError("O limite mínimo de clip não pode superar o máximo.")

    result = df.copy(deep=True)
    result[feature] = result[feature] + value

    if clip is not None:
        result[feature] = result[feature].clip(lower=lower, upper=upper)

    return result
