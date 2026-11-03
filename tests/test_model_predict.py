import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from creditwatch.model.predict import score_batch


class StubClassifier:
    def predict_proba(self, frame):
        positive = np.asarray(frame["feature"], dtype=float)
        return np.column_stack((1 - positive, positive))


class InvalidClassifier:
    def predict_proba(self, frame):
        return np.array([[0.2, 1.2] for _ in range(len(frame))])


def _write_model(tmp_path, model=None):
    model_path = tmp_path / "baseline.joblib"
    metadata_path = tmp_path / "baseline.json"
    joblib.dump(model or StubClassifier(), model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path.write_text(
        json.dumps(
            {
                "features": ["feature"],
                "model_version": "test_v1",
                "threshold": 0.5,
                "model_sha256": model_hash,
            }
        ),
        encoding="utf-8",
    )
    return model_path, metadata_path


def test_score_batch_preserves_input_and_adds_contract_outputs(tmp_path):
    model_path, _ = _write_model(tmp_path)
    batch = pd.DataFrame({"customer_id": ["C1", "C2"], "feature": [0.2, 0.8], "target": [0, 1]})
    original = batch.copy(deep=True)

    result = score_batch(batch, model_path)

    pd.testing.assert_frame_equal(batch, original)
    assert result["score"].tolist() == pytest.approx([0.2, 0.8])
    assert result["prediction"].tolist() == [0, 1]
    assert result["model_version"].tolist() == ["test_v1", "test_v1"]
    assert result["target"].tolist() == [0, 1]


def test_score_batch_rejects_missing_features(tmp_path):
    model_path, _ = _write_model(tmp_path)

    with pytest.raises(ValueError, match="feature"):
        score_batch(pd.DataFrame({"customer_id": ["C1"]}), model_path)


def test_score_batch_rejects_output_column_collisions(tmp_path):
    model_path, _ = _write_model(tmp_path)

    with pytest.raises(ValueError, match="colunas de saída"):
        score_batch(pd.DataFrame({"feature": [0.2], "score": [0.1]}), model_path)


def test_score_batch_detects_model_metadata_mismatch(tmp_path):
    model_path, metadata_path = _write_model(tmp_path)
    model_path.write_bytes(model_path.read_bytes() + b"altered")

    with pytest.raises(ValueError, match="SHA-256"):
        score_batch(pd.DataFrame({"feature": [0.2]}), model_path, metadata_path)


def test_score_batch_rejects_non_probability_model_output(tmp_path):
    model_path, _ = _write_model(tmp_path, InvalidClassifier())

    with pytest.raises(ValueError, match="scores inválidos"):
        score_batch(pd.DataFrame({"feature": [0.2]}), model_path)
