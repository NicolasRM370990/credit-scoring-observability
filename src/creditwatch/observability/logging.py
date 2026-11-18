"""Logging JSON Lines por estágio (E1-09).

Cada linha de `artifacts/logs/pipeline.jsonl` tem exatamente os campos de
`interfaces.LOG_REQUIRED_FIELDS`. Por princípio de minimização (LGPD), o campo
`extra` aceita apenas escalares: contagens, estatísticas agregadas e índices.
Valores de linhas de dados nunca são gravados.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from creditwatch.interfaces import LOG_REQUIRED_FIELDS, STAGES, RunContext, utc_now

LOG_PATH = Path("artifacts/logs/pipeline.jsonl")
DATA_CONFIG_PATH = Path("config/data.yaml")
_SCALARS = (str, int, float, bool, type(None))
_FALLBACK_COLUMNS = frozenset({
    "customer_id", "target", "revolving_utilization", "age", "past_due_30_59", "debt_ratio",
    "monthly_income", "open_credit_lines", "times_90_days_late", "real_estate_loans",
    "past_due_60_89", "dependents", "score", "prediction",
})
REDACTED = "<redacted: data column>"


@lru_cache(maxsize=1)
def protected_columns() -> frozenset[str]:
    """Nomes de colunas do dataset (config/data.yaml): nunca podem ser chaves em `extra`.

    Agregados por feature devem usar prefixo (`psi_monthly_income`, `null_rate_age`).
    """
    columns = set(_FALLBACK_COLUMNS)
    try:
        cfg = yaml.safe_load(DATA_CONFIG_PATH.read_text(encoding="utf-8"))
        features = cfg.get("features", {})
        columns.update(features.get("numerical", []), features.get("categorical", []))
        columns.update(filter(None, (cfg.get("id_col"), cfg.get("target"))))
    except (OSError, AttributeError, yaml.YAMLError):
        pass
    return frozenset(columns)


def _clean_value(key: str, value: Any) -> Any:
    if key in protected_columns():
        return REDACTED
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return "<omitted: tabular data>"
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, dict):
        return {str(k): _clean_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [v if isinstance(v, _SCALARS) else "<omitted>" for v in value]
    return "<omitted>"


def sanitize_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Mantém só escalares (e dicts/listas de escalares); objetos complexos viram placeholder.

    Chaves com nome de coluna do dataset são redigidas em qualquer nível, para impedir que um
    `row.to_dict()` acabe no log.
    """
    if not extra:
        return {}
    return {str(key): _clean_value(str(key), value) for key, value in extra.items()}


def build_event(
    ctx: RunContext,
    stage: str,
    event: str,
    status: str = "ok",
    *,
    duration_ms: float = 0.0,
    n_rows: int = 0,
    message: str = "",
    extra: dict[str, Any] | None = None,
    level: str = "INFO",
) -> dict[str, Any]:
    """Monta o dicionário do evento no formato do contrato (docs/contracts.md §3)."""
    if stage not in STAGES:
        raise ValueError(f"stage inválido: {stage!r}; use um de {STAGES}")
    if status not in ("ok", "warn", "error"):
        raise ValueError(f"status inválido: {status!r}")
    payload = {
        "ts": utc_now().isoformat(),
        "level": level,
        "run_id": ctx.run_id,
        "batch_id": ctx.batch_id,
        "stage": stage,
        "event": event,
        "status": status,
        "duration_ms": round(float(duration_ms), 3),
        "n_rows": int(n_rows),
        "message": message,
        "extra": sanitize_extra(extra),
    }
    assert tuple(payload) == LOG_REQUIRED_FIELDS
    return payload


class JsonLinesHandler(logging.Handler):
    """Grava `record.payload` como uma linha JSON (append)."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)

    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, "payload", None)
        if payload is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


class StageLogger(logging.LoggerAdapter):
    """Logger de um estágio: `.event(...)` emite uma linha no contrato."""

    def __init__(self, logger: logging.Logger, ctx: RunContext, stage: str) -> None:
        super().__init__(logger, {})
        self.ctx = ctx
        self.stage = stage

    def event(
        self,
        event: str,
        status: str = "ok",
        *,
        duration_ms: float = 0.0,
        n_rows: int = 0,
        message: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        level_name = {"ok": "INFO", "warn": "WARNING", "error": "ERROR"}[status]
        payload = build_event(
            self.ctx, self.stage, event, status,
            duration_ms=duration_ms, n_rows=n_rows, message=message, extra=extra,
            level=level_name,
        )
        self.logger.log(getattr(logging, level_name), message, extra={"payload": payload})
        return payload


def get_logger(stage: str, ctx: RunContext, log_path: Path | None = None) -> StageLogger:
    """Devolve o logger JSON de um estágio para a execução `ctx`."""
    path = Path(log_path or LOG_PATH)
    logger = logging.getLogger(f"creditwatch.{stage}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        if isinstance(handler, JsonLinesHandler) and handler.path != path:
            logger.removeHandler(handler)
    if not any(isinstance(h, JsonLinesHandler) and h.path == path for h in logger.handlers):
        logger.addHandler(JsonLinesHandler(path))
    return StageLogger(logger, ctx, stage)


@contextmanager
def stage_timer(
    stage: str, ctx: RunContext, *, n_rows: int = 0, log_path: Path | None = None
) -> Iterator[StageLogger]:
    """Emite `stage_started` e `stage_finished` (ou `stage_failed`, relançando a exceção)."""
    log = get_logger(stage, ctx, log_path)
    log.event("stage_started", n_rows=n_rows, message=f"{stage} started")
    started = time.perf_counter()
    try:
        yield log
    except Exception as exc:
        log.event(
            "stage_failed", "error",
            duration_ms=(time.perf_counter() - started) * 1000, n_rows=n_rows,
            message=f"{stage} failed: {type(exc).__name__}",
            extra={"exception": type(exc).__name__},
        )
        raise
    log.event(
        "stage_finished", "ok",
        duration_ms=(time.perf_counter() - started) * 1000, n_rows=n_rows,
        message=f"{stage} finished",
    )


def stage(name: str, log_path: Path | None = None) -> Callable:
    """Decorator: registra início, fim, duração e status da função de um estágio.

    A função decorada deve receber o `RunContext` como argumento (posicional ou `ctx=`).
    Se devolver um DataFrame, `n_rows` do evento final é o número de linhas.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = kwargs.get("ctx") or next(
                (arg for arg in args if isinstance(arg, RunContext)), None
            )
            if ctx is None:
                raise TypeError(f"@stage({name!r}) exige um RunContext entre os argumentos")
            log = get_logger(name, ctx, log_path)
            log.event("stage_started", message=f"{name} started")
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                log.event(
                    "stage_failed", "error",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    message=f"{name} failed: {type(exc).__name__}",
                    extra={"exception": type(exc).__name__},
                )
                raise
            n_rows = len(result) if isinstance(result, pd.DataFrame) else 0
            log.event(
                "stage_finished", "ok",
                duration_ms=(time.perf_counter() - started) * 1000, n_rows=n_rows,
                message=f"{name} finished",
            )
            return result

        return wrapper

    return decorator


def read_log(path: Path | None = None) -> list[dict[str, Any]]:
    """Lê o arquivo JSON Lines de log (lista vazia se não existir)."""
    file = Path(path or LOG_PATH)
    if not file.exists():
        return []
    with file.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
