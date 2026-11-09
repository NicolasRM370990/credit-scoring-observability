"""Validação de lotes de dados conforme config/contract.yaml."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import pandera.pandas as pa
import yaml

from creditwatch.interfaces import (
    CONTRACT_VERSION,
    RunContext,
    ValidationCheck,
    ValidationResult,
)

CONTRACT_PATH = Path("config/contract.yaml")
MAX_EXAMPLES = 10


def _load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Carrega e verifica a configuração do contrato."""
    if not path.exists():
        raise FileNotFoundError(f"Contrato não encontrado: {path}")

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("O contrato deve ser um objeto YAML.")

    for section in ("version", "block_rules", "warn_rules"):
        if section not in config:
            raise ValueError(f"Seção obrigatória ausente no contrato: {section}")

    return config


def _row_examples(mask: pd.Series) -> list[dict[str, int]]:
    """Retorna somente posições de linhas, sem expor valores dos dados."""
    positions = mask.to_numpy().nonzero()[0][:MAX_EXAMPLES]
    return [{"row_index": int(position)} for position in positions]


def _make_check(
    rule_id: str,
    severity: str,
    mask: pd.Series,
    n_rows: int,
    column: str | None = None,
) -> ValidationCheck:
    """Constrói um resultado padronizado para uma regra baseada em linhas."""
    failures = int(mask.sum())

    return ValidationCheck(
        rule_id=rule_id,
        severity=severity,
        column=column,
        passed=failures == 0,
        n_failures=failures,
        failure_rate=failures / n_rows if n_rows else 0.0,
        examples=_row_examples(mask),
    )


def _numeric_values(df: pd.DataFrame, column: str) -> pd.Series:
    """Converte valores para análise numérica sem modificar o DataFrame."""
    return pd.to_numeric(df[column], errors="coerce")


def _validate_schema(
    df: pd.DataFrame,
    rule: dict[str, Any],
) -> tuple[ValidationCheck, pd.Series]:
    """Valida colunas obrigatórias e nulabilidade usando Pandera."""
    n_rows = len(df)
    required = rule["required_columns"]
    nullable = set(rule.get("nullable_columns", []))
    missing = sorted(set(required) - set(df.columns))

    row_failures = pd.Series(False, index=range(n_rows))

    if missing:
        row_failures[:] = True
    else:
        schema = pa.DataFrameSchema(
            columns={column: pa.Column(nullable=column in nullable) for column in required},
            strict=False,
            coerce=False,
        )

        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            failure_cases = exc.failure_cases

            if "index" not in failure_cases.columns:
                row_failures[:] = True
            else:
                failed_indexes = failure_cases["index"].dropna()

                if failed_indexes.empty:
                    row_failures[:] = True
                else:
                    row_failures = pd.Series(
                        df.index.isin(failed_indexes).tolist(),
                        index=range(n_rows),
                    )

    # target_allowed indica se a coluna target pode estar presente.
    # Quando presente, seus valores devem ser binários (0 ou 1).
    if "target" in df.columns:
        if not rule.get("target_allowed", False):
            row_failures[:] = True
        else:
            target = _numeric_values(df, "target")
            invalid_target = (target.isna() | ~target.isin([0, 1])).reset_index(drop=True)
            row_failures |= invalid_target

    check = ValidationCheck(
        rule_id="R04_SCHEMA",
        severity="BLOCK",
        column=None,
        passed=not bool(row_failures.any()),
        n_failures=int(row_failures.sum()),
        failure_rate=float(row_failures.mean()) if n_rows else 0.0,
        examples=_row_examples(row_failures),
    )

    return check, row_failures


def validate_batch(
    df: pd.DataFrame,
    batch_id: str,
    contract_version: str = CONTRACT_VERSION,
) -> ValidationResult:
    """Executa as regras BLOCK e WARN e devolve ValidationResult."""
    started = perf_counter()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve ser um pandas.DataFrame.")

    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("batch_id deve ser uma string não vazia.")

    if df.columns.has_duplicates:
        raise ValueError("O DataFrame possui nomes de colunas duplicados.")

    config = _load_contract()

    configured_version = str(config["version"])
    if contract_version != configured_version:
        raise ValueError(
            f"Versão incompatível: solicitada {contract_version}, configurada {configured_version}."
        )

    n_rows = len(df)
    checks: list[ValidationCheck] = []
    blocked_rows = pd.Series(False, index=range(n_rows))

    rules = config["block_rules"]

    # R01 - idade dentro do intervalo permitido.
    rule = rules["R01_AGE_RANGE"]
    column = rule["column"]
    if column not in df.columns:
        mask = pd.Series(True, index=range(n_rows))
    else:
        values = _numeric_values(df, column)
        mask = (values.isna() | ~values.between(rule["min"], rule["max"])).reset_index(drop=True)

    check = _make_check("R01_AGE_RANGE", "BLOCK", mask, n_rows, column)
    checks.append(check)
    blocked_rows |= mask

    # R02 - renda obrigatória e dentro do intervalo permitido.
    rule = rules["R02_INCOME"]
    column = rule["column"]
    if column not in df.columns:
        mask = pd.Series(True, index=range(n_rows))
    else:
        values = _numeric_values(df, column)
        mask = (values.isna() | ~values.between(rule["min"], rule["max"])).reset_index(drop=True)

    check = _make_check("R02_INCOME", "BLOCK", mask, n_rows, column)
    checks.append(check)
    blocked_rows |= mask

    # R03 - ausência de duplicidades nas colunas configuradas
    # e, quando solicitado, em linhas completas.
    rule = rules["R03_NO_DUPLICATES"]
    columns = rule["columns"]

    if any(column not in df.columns for column in columns):
        mask = pd.Series(True, index=range(n_rows))
    else:
        mask = df.duplicated(subset=columns, keep=False)

    if rule.get("check_full_row", False):
        mask |= df.duplicated(keep=False)

    mask = mask.reset_index(drop=True)
    check = _make_check("R03_NO_DUPLICATES", "BLOCK", mask, n_rows)
    checks.append(check)
    blocked_rows |= mask

    # R04 - schema, colunas obrigatórias e nulabilidade.
    check, mask = _validate_schema(df, rules["R04_SCHEMA"])
    checks.append(check)
    blocked_rows |= mask

    # R05 - variáveis numéricas não negativas.
    rule = rules["R05_NONNEG"]
    for column in rule["columns"]:
        if column not in df.columns:
            mask = pd.Series(True, index=range(n_rows))
        else:
            original = df[column]
            values = _numeric_values(df, column)
            mask = (
                (original.notna() & values.isna()) | (values.notna() & (values < 0))
            ).reset_index(drop=True)

        check = _make_check(
            "R05_NONNEG",
            "BLOCK",
            mask,
            n_rows,
            column,
        )
        checks.append(check)
        blocked_rows |= mask

    # W01 - taxa de nulos por coluna não pode superar o limite.
    rule = config["warn_rules"]["W01_NULL_RATE"]
    max_null_rate = float(rule["max_null_rate"])

    null_rates = df.isna().mean()
    columns_over_limit = null_rates[null_rates > max_null_rate].index.tolist()

    if columns_over_limit:
        mask = df[columns_over_limit].isna().any(axis=1).reset_index(drop=True)
        failure_rate = float(null_rates[columns_over_limit].max())
        failures = int(mask.sum())
    else:
        mask = pd.Series(False, index=range(n_rows))
        failure_rate = float(null_rates.max()) if len(null_rates) else 0.0
        failures = 0

    checks.append(
        ValidationCheck(
            rule_id="W01_NULL_RATE",
            severity="WARN",
            column=None,
            passed=failures == 0,
            n_failures=failures,
            failure_rate=failure_rate,
            examples=_row_examples(mask),
        )
    )

    # W02 - quantidade mínima de linhas.
    rule = config["warn_rules"]["W02_MIN_VOLUME"]
    min_rows = int(rule["min_rows"])
    volume_failed = n_rows < min_rows

    checks.append(
        ValidationCheck(
            rule_id="W02_MIN_VOLUME",
            severity="WARN",
            column=None,
            passed=not volume_failed,
            n_failures=max(0, min_rows - n_rows),
            failure_rate=(max(0, min_rows - n_rows) / min_rows if min_rows else 0.0),
            examples=[],
        )
    )

    has_block_failure = any(not check.passed and check.severity == "BLOCK" for check in checks)
    has_warning = any(not check.passed and check.severity == "WARN" for check in checks)

    status = "BLOCKED" if has_block_failure else "PASSED_WITH_WARNINGS" if has_warning else "PASSED"

    context = RunContext.new(batch_id=batch_id)

    return ValidationResult(
        run_id=context.run_id,
        batch_id=batch_id,
        contract_version=contract_version,
        status=status,
        n_rows=n_rows,
        n_rows_failed=int(blocked_rows.sum()),
        checks=checks,
        duration_s=perf_counter() - started,
    )
