from __future__ import annotations

import math
from numbers import Real
from collections.abc import Mapping, Sequence

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


def apply_resample_mix(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
    replace: bool = True,
) -> pd.DataFrame:
    """Reamostra registros do próprio DataFrame.

    Args:
        df: DataFrame de origem.
        sample_size: Quantidade de registros da amostra.
        seed: Semente para garantir reprodutibilidade.
        replace: Permite selecionar o mesmo registro mais de uma vez.

    Returns:
        DataFrame contendo os registros reamostrados.

    Raises:
        TypeError: Se os argumentos possuírem tipos inválidos.
        ValueError: Se o tamanho ou a seed forem inválidos.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size deve ser um número inteiro.")

    if sample_size <= 0:
        raise ValueError("sample_size deve ser maior que zero.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um número inteiro.")

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed deve estar entre 0 e 4294967295.")

    if not isinstance(replace, bool):
        raise TypeError("replace deve ser booleano.")

    if df.empty:
        raise ValueError("Não é possível reamostrar um DataFrame vazio.")

    if not replace and sample_size > len(df):
        raise ValueError("sample_size não pode ser maior que o DataFrame quando replace=False.")

    return df.sample(
        n=sample_size,
        replace=replace,
        random_state=seed,
    ).copy(deep=True)


def apply_transformations(
    df: pd.DataFrame,
    transformations: list[dict],
    seed: int,
) -> pd.DataFrame:
    """Executa uma sequência de transformações sobre um DataFrame.

    Args:
        df: DataFrame de origem.
        transformations: Lista de configurações das transformações.
        seed: Semente utilizada nas operações aleatórias.

    Returns:
        DataFrame com as transformações aplicadas.

    Raises:
        TypeError: Se os argumentos possuírem tipos inválidos.
        ValueError: Se uma operação ou configuração for inválida.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    if not isinstance(transformations, list):
        raise TypeError("transformations deve ser uma lista.")

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed deve ser um número inteiro.")

    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed deve estar entre 0 e 4294967295.")

    result = df.copy(deep=True)

    for index, transformation in enumerate(transformations):
        if not isinstance(transformation, Mapping):
            raise TypeError(f"A transformação na posição {index} deve ser um dicionário.")

        operation = transformation.get("op")
        feature = transformation.get("feature")

        if not isinstance(operation, str):
            raise ValueError(f"A transformação na posição {index} precisa informar 'op'.")

        if operation in {"scale", "shift_add"} and not isinstance(feature, str):
            if not isinstance(feature, str):
                raise ValueError(f"A transformação '{operation}' precisa informar 'feature'.")

        if operation == "scale":
            allowed_keys = {"feature", "op", "factor"}
            if set(transformation) != allowed_keys:
                raise ValueError(
                    "Parâmetros inválidos para scale. Esperados: feature, op e factor."
                )

            result = apply_scale(
                df=result,
                feature=feature,
                factor=transformation["factor"],
            )

        elif operation == "shift_add":
            allowed_keys = {"feature", "op", "value", "clip"}
            if not {"feature", "op", "value"}.issubset(transformation):
                raise ValueError(
                    "Parâmetros inválidos para shift_add. "
                    "Esperados: feature, op e value; clip é opcional."
                )

            if set(transformation) - allowed_keys:
                raise ValueError("Parâmetros desconhecidos para shift_add.")

            result = apply_shift_add(
                df=result,
                feature=feature,
                value=transformation["value"],
                clip=transformation.get("clip"),
            )

        elif operation == "resample_mix":
            allowed_keys = {"op", "sample_size", "replace"}
            if not {"op", "sample_size"}.issubset(transformation):
                raise ValueError(
                    "Parâmetros inválidos para resample_mix. Esperados: op e sample_size."
                )

            if set(transformation) - allowed_keys:
                raise ValueError("Parâmetros desconhecidos para resample_mix.")

            result = apply_resample_mix(
                df=result,
                sample_size=transformation["sample_size"],
                seed=seed,
                replace=transformation.get("replace", True),
            )

        else:
            raise ValueError(f"Operação não suportada: '{operation}'.")

    return result
