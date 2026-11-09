
"""Geração de lotes sintéticos para cenários de drift."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from creditwatch.contracts import validate_batch
from creditwatch.simulation.scenarios import (
    apply_resample_mix,
    apply_scenario_transformations,
    load_scenario_config,
)

SUPPORTED_SCENARIOS = {
    "no_drift",
    "covariate_moderate",
    "covariate_severe",
}


def _batch_seed(seed: int, batch_number: int) -> int:
    """Gera uma seed determinística para cada lote."""
    payload = f"{seed}:{batch_number}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _save_manifest(
    output_dir: Path,
    scenario: str,
    config: dict[str, Any],
    batches: list[dict[str, Any]],
) -> Path:
    """Salva o manifesto da geração."""
    manifest = {
        "schema_version": "1.0",
        "scenario": scenario,
        "seed": config["seed"],
        "batch_size": config["batch_size"],
        "generated_at": datetime.now(UTC).isoformat(),
        "batches": batches,
    }

    manifest_path = output_dir / "MANIFEST.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest_path


def generate_scenario(
    pool: pd.DataFrame,
    scenario: str,
    config_path: Path,
    out_dir: Path,
) -> list[Path]:
    """Gera lotes de um cenário e valida cada lote."""
    if not isinstance(pool, pd.DataFrame):
        raise TypeError("pool deve ser um pandas.DataFrame.")

    if pool.empty:
        raise ValueError("O pool de produção está vazio.")

    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario deve ser uma string não vazia.")

    config = load_scenario_config(config_path)
    scenarios = config["scenarios"]

    if scenario not in scenarios:
        raise ValueError(f"Cenário não encontrado: {scenario}")

    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"O cenário '{scenario}' não está disponível nesta etapa."
        )

    scenario_config = scenarios[scenario]

    batches_count = scenario_config.get("batches")
    batch_size = config.get("batch_size")
    seed = config["seed"]

    if isinstance(batches_count, bool) or not isinstance(batches_count, int):
        raise TypeError("batches deve ser um número inteiro.")

    if batches_count <= 0:
        raise ValueError("batches deve ser maior que zero.")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size deve ser um número inteiro.")

    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")

    required_rows = batches_count * batch_size

    if required_rows > len(pool):
        raise ValueError(
            f"Pool insuficiente: necessários {required_rows} registros, "
            f"disponíveis {len(pool)}."
        )

    output_dir = Path(out_dir) / scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_pool = apply_resample_mix(
        df=pool,
        sample_size=required_rows,
        seed=_batch_seed(seed, 0),
        replace=False,
    )

    generated_files: list[Path] = []
    manifest_batches: list[dict[str, Any]] = []

    for batch_number in range(1, batches_count + 1):
        start = (batch_number - 1) * batch_size
        end = start + batch_size

        batch = sampled_pool.iloc[start:end].copy(deep=True)

        batch = apply_scenario_transformations(
            df=batch,
            scenario=scenario,
            config_path=config_path,
        )

        batch_id = f"{scenario}_batch_{batch_number:02d}"

        result = validate_batch(
            batch,
            batch_id=batch_id,
        )

        if result.status == "BLOCKED":
            raise ValueError(
                f"Lote {batch_id} bloqueado pelo contrato: "
                f"{result.model_dump_json()}"
            )

        output_path = output_dir / f"batch_{batch_number:02d}.parquet"
        batch.to_parquet(output_path, index=False)

        generated_files.append(output_path)

        manifest_batches.append(
            {
                "batch_id": batch_id,
                "file": output_path.name,
                "n_rows": len(batch),
                "validation_status": result.status,
                "seed": _batch_seed(seed, 0),
                "transformations": scenario_config.get("transformations", []),
            }
        )

    _save_manifest(
        output_dir=output_dir,
        scenario=scenario,
        config=config,
        batches=manifest_batches,
    )

    return generated_files