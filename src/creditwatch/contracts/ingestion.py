"""Ingestão de lotes com validação do contrato de dados."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from creditwatch.contracts.validate import validate_batch
from creditwatch.interfaces import BatchRejected, RunContext

QUARANTINE_DIR = Path("data/quarantine")


def ingest_batch(path: Path, ctx: RunContext) -> pd.DataFrame:
    """Lê e valida um lote, encaminhando lotes bloqueados para quarentena."""
    if not isinstance(path, Path):
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Arquivo do lote não encontrado: {path}")

    if not isinstance(ctx, RunContext):
        raise TypeError("ctx deve ser uma instância de RunContext.")

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Não foi possível ler o lote Parquet: {path}") from exc

    batch_id = ctx.batch_id or path.stem
    validation = validate_batch(frame, batch_id=batch_id)

    if validation.status == "BLOCKED":
        quarantine_dir = QUARANTINE_DIR
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        destination = quarantine_dir / path.name

        if destination.exists():
            raise FileExistsError(
                f"Já existe um arquivo com esse nome na quarentena: {destination}"
            )

        shutil.move(str(path), str(destination))

        raise BatchRejected(
            f"Lote {batch_id!r} bloqueado pelo contrato: "
            f"{validation.n_rows_failed} linha(s) rejeitada(s). "
            f"Arquivo movido para {destination}."
        )

    return frame
