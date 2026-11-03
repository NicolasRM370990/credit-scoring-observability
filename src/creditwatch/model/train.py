"""Treina e registra o baseline HistGradientBoostingClassifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from creditwatch.model.evaluate import evaluate_split, select_threshold

DATA_CONFIG_PATH = Path("config/data.yaml")
MODEL_CONFIG_PATH = Path("config/model.yaml")
MANIFEST_PATH = Path("data/processed/split_manifest.json")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(
    *,
    data_config_path: Path = DATA_CONFIG_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    processed_dir: Path | None = None,
    output_dir: Path = Path("models"),
) -> dict:
    data_cfg = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    processed_dir = Path(processed_dir or data_cfg["paths"]["processed"])
    features = [*data_cfg["features"]["numerical"], *data_cfg["features"]["categorical"]]
    target = data_cfg["target"]

    train_frame = pd.read_parquet(processed_dir / "reference_train.parquet")
    val_frame = pd.read_parquet(processed_dir / "reference_val.parquet")
    test_frame = pd.read_parquet(processed_dir / "reference_test.parquet")
    for name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        missing = set(features + [target]) - set(frame.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no split {name}: {sorted(missing)}")
        if not frame[target].isin([0, 1]).all() or frame[target].nunique() != 2:
            raise ValueError(f"O target no split {name} deve conter as classes 0 e 1")

    estimator = HistGradientBoostingClassifier(**model_cfg["model"]["params"])
    estimator.fit(train_frame[features], train_frame[target].astype(int))
    val_scores = estimator.predict_proba(val_frame[features])[:, 1]
    configured_threshold = model_cfg.get("threshold")
    threshold = (
        float(configured_threshold)
        if configured_threshold is not None
        else select_threshold(val_frame[target], val_scores)
    )
    if not 0 <= threshold <= 1:
        raise ValueError("config/model.yaml: threshold deve estar entre 0 e 1")

    val_metrics = evaluate_split(estimator, val_frame, features, target, threshold)
    test_metrics = evaluate_split(estimator, test_frame, features, target, threshold)
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_frame[features], train_frame[target].astype(int))
    dummy_metrics = evaluate_split(dummy, test_frame, features, target, 0.5)
    min_auc = float(model_cfg.get("min_auc_test", 0.0))
    if test_metrics["roc_auc"] < min_auc:
        raise RuntimeError(
            f"ROC-AUC de teste {test_metrics['roc_auc']:.4f} abaixo do mínimo configurado {min_auc:.4f}"
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    version = model_cfg["model_version"]
    model_path = output_dir / f"{version}.joblib"
    metadata_path = output_dir / f"{version}.json"
    joblib.dump(estimator, model_path)
    metadata = {
        "model_version": version,
        "model_name": model_cfg["model"]["name"],
        "created_at": datetime.now(UTC).isoformat(),
        "features": features,
        "target": target,
        "seed": int(model_cfg.get("seed", data_cfg["seed"])),
        "threshold": threshold,
        "threshold_method": "config/model.yaml" if configured_threshold is not None else "validation_max_f1",
        "threshold_selection_note": (
            "Threshold maximizes F1 on validation; operational costs and calibration were not supplied."
            if configured_threshold is None
            else "Threshold explicitly configured in config/model.yaml."
        ),
        "dataset": data_cfg["dataset"]["name"],
        "raw_sha256": manifest.get("raw_sha256"),
        "splits": {name: len(frame) for name, frame in (
            ("reference_train", train_frame), ("reference_val", val_frame), ("reference_test", test_frame)
        )},
        "metrics": {"validation": val_metrics, "test": test_metrics, "dummy_test": dummy_metrics},
        "min_auc_test": min_auc,
        "model_sha256": sha256_of(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Treina o baseline e salva modelo e metadata")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    metadata = train(output_dir=args.output_dir)
    metrics = metadata["metrics"]
    print(f"Modelo salvo: {args.output_dir / (metadata['model_version'] + '.joblib')}")
    print(f"Threshold: {metadata['threshold']:.6f} ({metadata['threshold_method']})")
    print(f"ROC-AUC teste: {metrics['test']['roc_auc']:.4f}; dummy: {metrics['dummy_test']['roc_auc']:.4f}")
    print(f"PR-AUC teste: {metrics['test']['pr_auc']:.4f}; KS: {metrics['test']['ks']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
