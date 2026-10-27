"""Contratos de integração v1: fonte da verdade entre os módulos.

Mudanças aqui só entram por PR com a label `contract-change`, aprovado por
quem produz e por quem consome cada interface (ver docs/contracts.md).
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"
CONTRACT_VERSION = "1.0.0"

STAGES = ("ingest", "validate", "score", "drift", "alert", "report", "pipeline")
LOG_REQUIRED_FIELDS = (
    "ts", "level", "run_id", "batch_id", "stage", "event",
    "status", "duration_ms", "n_rows", "message", "extra",
)

ValidationStatus = Literal["PASSED", "PASSED_WITH_WARNINGS", "BLOCKED"]
RuleSeverity = Literal["BLOCK", "WARN"]
DriftSeverity = Literal["OK", "WARNING", "CRITICAL"]
AlertSeverity = Literal["WARNING", "CRITICAL"]
RunStatus = Literal["ok", "blocked", "error"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class BatchRejected(Exception):
    """Lote bloqueado pelo contrato de dados (levantada por `ingest_batch`)."""


class RunContext(BaseModel):
    run_id: str
    batch_id: str | None = None
    scenario: str | None = None

    @classmethod
    def new(cls, batch_id: str | None = None, scenario: str | None = None) -> RunContext:
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        return cls(run_id=f"{stamp}_{uuid4().hex[:6]}", batch_id=batch_id, scenario=scenario)


# ---------------------------------------------------------------- validação (P2)
class ValidationCheck(BaseModel):
    rule_id: str                       # ex.: R01_AGE_RANGE, W01_NULL_RATE
    severity: RuleSeverity
    column: str | None = None
    passed: bool
    n_failures: int = 0
    failure_rate: float = 0.0
    examples: list[dict[str, int]] = Field(default_factory=list)  # só {"row_index": n}, nunca valores


class ValidationResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    batch_id: str
    contract_version: str = CONTRACT_VERSION
    status: ValidationStatus
    n_rows: int
    n_rows_failed: int = 0
    checks: list[ValidationCheck]
    duration_s: float
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------- drift (P3)
class FeatureDrift(BaseModel):
    name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    evidently_drifted: bool | None = None
    drifted: bool
    severity: DriftSeverity


class PredictionDrift(BaseModel):
    score_psi: float
    score_ks: float
    mean_score_ref: float
    mean_score_cur: float


class DriftThresholds(BaseModel):
    psi_warn: float
    psi_critical: float
    ks_stat_min: float
    drift_share_alert: float


class DriftResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    batch_id: str
    scenario: str | None = None
    reference_rows: int
    current_rows: int
    dataset_drift: bool
    drift_share: float
    features: list[FeatureDrift]
    prediction_drift: PredictionDrift | None = None
    thresholds: DriftThresholds
    report_html: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------- observabilidade (P4)
class RunMetrics(BaseModel):
    """Uma linha por execução em artifacts/metrics/run_history.jsonl.

    No MLflow, `latency_s`, `psi` e `ks` são achatados:
    latency_<estagio>_s, psi_<feature>, ks_<feature>.
    """
    schema_version: str = SCHEMA_VERSION
    run_id: str
    batch_id: str
    scenario: str | None = None
    model_version: str | None = None
    contract_version: str = CONTRACT_VERSION
    pipeline_status: RunStatus
    rows_in: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    validation_blocked: bool = False
    latency_s: dict[str, float] = Field(default_factory=dict)  # ingest, validate, score, drift, total
    drift_share: float | None = None
    psi: dict[str, float] = Field(default_factory=dict)
    ks: dict[str, float] = Field(default_factory=dict)
    score_mean: float | None = None
    score_psi: float | None = None
    n_alerts_warning: int = 0
    n_alerts_critical: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class Alert(BaseModel):
    ts: datetime = Field(default_factory=utc_now)
    run_id: str
    batch_id: str
    severity: AlertSeverity
    rule_id: str                       # A01_... a A06_...
    message: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None


# ---------------------------------------------------------------- assinaturas
class ValidateBatch(Protocol):            # contracts/validate.py (P2)
    def __call__(self, df: pd.DataFrame, batch_id: str,
                 contract_version: str = CONTRACT_VERSION) -> ValidationResult: ...


class IngestBatch(Protocol):              # contracts/ingestion.py (P2); levanta BatchRejected
    def __call__(self, path: Path, ctx: RunContext) -> pd.DataFrame: ...


class ScoreBatch(Protocol):               # model/predict.py (P1)
    def __call__(self, df: pd.DataFrame, model_path: Path) -> pd.DataFrame: ...


class GenerateScenario(Protocol):         # simulation/generate.py (P3)
    def __call__(self, pool: pd.DataFrame, scenario: str,
                 config_path: Path, out_dir: Path) -> list[Path]: ...


class DetectDrift(Protocol):              # drift/detector.py (P3)
    def __call__(self, reference: pd.DataFrame, current: pd.DataFrame,
                 cfg: DriftThresholds, ctx: RunContext) -> DriftResult: ...


class EvaluateAlerts(Protocol):           # observability/alerts.py (P4)
    def __call__(self, validation: ValidationResult | None, drift: DriftResult | None,
                 metrics: RunMetrics) -> list[Alert]: ...