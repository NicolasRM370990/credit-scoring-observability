import shutil
from pathlib import Path

import pandas as pd
import pytest

from creditwatch.contracts.ingestion import ingest_batch
from creditwatch.interfaces import BatchRejected, RunContext


def prepare_workspace(tmp_path):
    """Disponibiliza a configuração real do contrato no ambiente temporário."""
    project_root = Path(__file__).resolve().parents[2]
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        project_root / "config" / "contract.yaml",
        config_dir / "contract.yaml",
    )


def test_ingest_valid_batch(production_pool, tmp_path, monkeypatch):
    prepare_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    batch_path = tmp_path / "valid_batch.parquet"
    production_pool.to_parquet(batch_path, index=False)

    context = RunContext(
        run_id="test_valid",
        batch_id="valid_batch",
        scenario="no_drift",
    )

    result = ingest_batch(batch_path, context)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(production_pool)
    assert batch_path.exists()
    assert not (tmp_path / "data" / "quarantine").exists()


def test_ingest_invalid_batch_moves_file_to_quarantine(production_pool, tmp_path, monkeypatch):
    prepare_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    batch_path = tmp_path / "invalid_age.parquet"
    invalid = production_pool.copy()
    invalid.loc[invalid.index[0], "age"] = 17
    invalid.to_parquet(batch_path, index=False)

    context = RunContext(
        run_id="test_invalid",
        batch_id="invalid_age",
        scenario="invalid",
    )

    with pytest.raises(BatchRejected):
        ingest_batch(batch_path, context)

    quarantine_path = tmp_path / "data" / "quarantine" / batch_path.name

    assert not batch_path.exists()
    assert quarantine_path.is_file()


def test_ingest_missing_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing.parquet"

    context = RunContext(
        run_id="test_missing",
        batch_id="missing",
    )

    with pytest.raises(FileNotFoundError):
        ingest_batch(missing_path, context)
