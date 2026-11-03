"""Scoring de lotes usando modelo e metadata persistidos pelo treino."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_batch(
    df: pd.DataFrame,
    model_path: Path,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Acrescenta score, prediction e model_version sem alterar o DataFrame de entrada.

    O arquivo de metadata fica ao lado do modelo com a mesma raiz e extensão `.json`.
    ``target`` pode estar presente para avaliação, mas nunca é usado como feature.
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path) if metadata_path else model_path.with_suffix(".json")
    if not model_path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata do modelo não encontrada: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("features")
    version = metadata.get("model_version")
    threshold = metadata.get("threshold")
    if not isinstance(features, list) or not features:
        raise ValueError("Metadata deve declarar uma lista não vazia em 'features'")
    if not isinstance(version, str) or not version:
        raise ValueError("Metadata deve declarar 'model_version'")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Metadata deve declarar um 'threshold' numérico entre 0 e 1")
    if len(features) != len(set(features)):
        raise ValueError("Metadata contém features repetidas")

    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas necessárias para scoring ausentes: {sorted(missing)}")
    output_columns = {"score", "prediction", "model_version"}
    collisions = output_columns & set(df.columns)
    if collisions:
        raise ValueError(f"Lote já contém colunas de saída do scoring: {sorted(collisions)}")
    expected_hash = metadata.get("model_sha256")
    if expected_hash and sha256_of(model_path) != expected_hash:
        raise ValueError("SHA-256 do modelo não corresponde à metadata")

    model = joblib.load(model_path)
    scores = np.asarray(model.predict_proba(df[features])[:, 1], dtype=float)
    if scores.shape != (len(df),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Modelo retornou scores inválidos; esperava probabilidades entre 0 e 1")

    result = df.copy()
    result["score"] = scores
    result["prediction"] = (scores >= threshold).astype("int8")
    result["model_version"] = version
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplica scoring a um arquivo Parquet")
    parser.add_argument("input", type=Path, help="Lote Parquet de entrada")
    parser.add_argument("--model", type=Path, default=Path("models/baseline_v1.joblib"))
    parser.add_argument("--metadata", type=Path, help="Metadata JSON; por padrão usa a raiz do modelo")
    parser.add_argument("--output", type=Path, required=True, help="Parquet scored de saída")
    args = parser.parse_args()

    batch = pd.read_parquet(args.input)
    scored = score_batch(batch, args.model, args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    print(f"{len(scored)} linhas pontuadas em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
