import json

import numpy as np
import pandas as pd
import yaml

from creditwatch.model.train import train


def _split(n, offset):
    x = np.arange(n, dtype=float) / n
    target = (np.arange(n) % 2).astype(int)
    return pd.DataFrame({"feature": x + offset, "target": target})


def test_train_saves_loadable_model_and_metadata(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    output = tmp_path / "models"
    processed.mkdir()
    for name, frame in (
        ("reference_train", _split(80, 0.0)),
        ("reference_val", _split(40, 0.01)),
        ("reference_test", _split(40, 0.02)),
    ):
        frame.to_parquet(processed / f"{name}.parquet", index=False)

    data_cfg = {
        "seed": 42,
        "dataset": {"name": "synthetic"},
        "target": "target",
        "features": {"numerical": ["feature"], "categorical": []},
        "paths": {"processed": str(processed)},
    }
    model_cfg = {
        "model": {"name": "HistGradientBoostingClassifier", "params": {
            "max_iter": 5, "early_stopping": False, "random_state": 42
        }},
        "threshold": None,
        "min_auc_test": 0.0,
        "model_version": "test_baseline",
        "seed": 42,
    }
    data_path, model_path = tmp_path / "data.yaml", tmp_path / "model.yaml"
    data_path.write_text(yaml.safe_dump(data_cfg), encoding="utf-8")
    model_path.write_text(yaml.safe_dump(model_cfg), encoding="utf-8")
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps({"raw_sha256": "abc123"}), encoding="utf-8")
    monkeypatch.setattr("creditwatch.model.train.MANIFEST_PATH", manifest_path)

    metadata = train(
        data_config_path=data_path,
        model_config_path=model_path,
        processed_dir=processed,
        output_dir=output,
    )

    assert (output / "test_baseline.joblib").is_file()
    saved_metadata = json.loads((output / "test_baseline.json").read_text(encoding="utf-8"))
    assert saved_metadata["threshold_method"] == "validation_max_f1"
    assert saved_metadata["raw_sha256"] == "abc123"
    assert 0 <= metadata["threshold"] <= 1
    assert {"roc_auc", "pr_auc", "ks", "brier", "confusion_matrix"} <= set(
        metadata["metrics"]["test"]
    )


def test_train_rejects_split_missing_target(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    for name in ("reference_train", "reference_val", "reference_test"):
        pd.DataFrame({"feature": [0.1, 0.2]}).to_parquet(processed / f"{name}.parquet")
    data_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.yaml"
    data_path.write_text(yaml.safe_dump({
        "seed": 42, "dataset": {"name": "synthetic"}, "target": "target",
        "features": {"numerical": ["feature"], "categorical": []},
        "paths": {"processed": str(processed)},
    }), encoding="utf-8")
    model_path.write_text(yaml.safe_dump({
        "model": {"name": "HistGradientBoostingClassifier", "params": {"max_iter": 2}},
        "threshold": None, "min_auc_test": 0.0, "model_version": "test", "seed": 42,
    }), encoding="utf-8")

    try:
        train(data_config_path=data_path, model_config_path=model_path, processed_dir=processed,
              output_dir=tmp_path / "models")
    except ValueError as exc:
        assert "Colunas ausentes" in str(exc)
    else:
        raise AssertionError("Esperava erro para splits sem target")
