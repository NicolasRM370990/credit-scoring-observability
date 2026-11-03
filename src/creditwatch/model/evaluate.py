"""Métricas e seleção de threshold para classificação binária."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

CONFIG_PATH = Path("config/data.yaml")


def ks_statistic(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> float:
    """Retorna o KS binário como a maior distância entre TPR e FPR."""
    y = np.asarray(y_true, dtype=int)
    positives = scores[y == 1]
    negatives = scores[y == 0]
    if not len(positives) or not len(negatives):
        raise ValueError("KS requer pelo menos um exemplo de cada classe")
    thresholds = np.unique(scores)
    tpr = np.array([(positives >= threshold).mean() for threshold in thresholds])
    fpr = np.array([(negatives >= threshold).mean() for threshold in thresholds])
    return float(np.max(np.abs(tpr - fpr)))


def select_threshold(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> float:
    """Escolhe threshold que maximiza F1 na validação (critério baseline)."""
    y = np.asarray(y_true, dtype=int)
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("A seleção de threshold requer exemplos das duas classes")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom else 0.0
        # Em empate, escolher o maior threshold reduz falsos positivos.
        if f1 > best_f1 or (f1 == best_f1 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold


def evaluate_predictions(
    y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Calcula métricas de ranking, calibração e classificação no threshold dado."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("y_true e scores devem ser vetores não vazios do mesmo tamanho")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("target deve conter somente 0 e 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores devem ser probabilidades finitas entre 0 e 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold deve estar entre 0 e 1")
    if len(np.unique(y)) < 2:
        raise ValueError("A avaliação requer exemplos das duas classes")

    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_rows": len(y),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "ks": ks_statistic(y, scores),
        "brier": float(brier_score_loss(y, scores)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_split(model, frame: pd.DataFrame, features: list[str], target: str, threshold: float) -> dict:
    missing = set(features + [target]) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para avaliação: {sorted(missing)}")
    scores = model.predict_proba(frame[features])[:, 1]
    return evaluate_predictions(frame[target], scores, threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia o modelo salvo no split de teste")
    parser.add_argument("--model", type=Path, default=Path("models/baseline_v1.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("models/baseline_v1.json"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/reference_test.parquet"))
    parser.add_argument("--output", type=Path, help="Opcional: salva métricas JSON neste caminho")
    args = parser.parse_args()

    import joblib

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.split)
    result = evaluate_split(
        joblib.load(args.model), frame, metadata["features"], cfg["target"], metadata["threshold"]
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
