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

from creditwatch.drift.detector import detect_drift
from creditwatch.interfaces import DriftThresholds, RunContext
from creditwatch.model.predict import score_batch

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


def evaluate_split(
    model, frame: pd.DataFrame, features: list[str], target: str, threshold: float
) -> dict:
    missing = set(features + [target]) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para avaliação: {sorted(missing)}")
    scores = model.predict_proba(frame[features])[:, 1]
    return evaluate_predictions(frame[target], scores, threshold)


def evaluate_with_labels(
    model, frame: pd.DataFrame, features: list[str], target: str, threshold: float
) -> dict:
    """Avalia um lote de produção quando o target atrasado estiver disponível.

    A função não usa os rótulos no scoring: eles entram somente na avaliação
    posterior. Retorna as mesmas métricas do conjunto de referência.
    """
    if target not in frame.columns:
        raise ValueError(f"Lote rotulado precisa conter a coluna {target!r}")
    return evaluate_split(model, frame, features, target, threshold)


def evaluate_labeled_batches(
    model,
    reference: pd.DataFrame,
    batches: dict[str, pd.DataFrame],
    features: list[str],
    target: str,
    threshold: float,
    drift_by_batch: dict[str, dict] | None = None,
) -> list[dict]:
    """Compara desempenho de referência e lotes rotulados por cenário."""
    results = [
        {
            "batch_id": "reference",
            "scenario": "reference",
            "drift_detected": False,
            **evaluate_split(model, reference, features, target, threshold),
        }
    ]

    for batch_id, frame in batches.items():
        drift_info = (drift_by_batch or {}).get(batch_id, {})

        results.append(
            {
                "batch_id": batch_id,
                "scenario": batch_id.split("_batch_")[0],
                "drift_detected": drift_info.get("drift_detected"),
                "drift_share": drift_info.get("drift_share"),
                "score_psi": drift_info.get("score_psi"),
                "score_ks": drift_info.get("score_ks"),
                "mean_score_ref": drift_info.get("mean_score_ref"),
                "mean_score_cur": drift_info.get("mean_score_cur"),
                **evaluate_with_labels(model, frame, features, target, threshold),
            }
        )

    return results


def write_perf_vs_drift(results: list[dict], output: Path) -> None:
    """Grava tabela Markdown comparando detecção e desempenho rotulado."""

    columns = (
        "scenario",
        "batch_id",
        "drift_detected",
        "drift_share",
        "score_psi",
        "score_ks",
        "mean_score_ref",
        "mean_score_cur",
        "positive_rate",
        "roc_auc",
        "pr_auc",
        "brier",
    )

    headers = (
        "Cenário",
        "Lote",
        "Drift detectado",
        "Drift share",
        "Score PSI",
        "Score KS",
        "Score médio ref.",
        "Score médio atual",
        "Taxa positiva",
        "ROC-AUC",
        "PR-AUC",
        "Brier",
    )

    lines = [
        "# Desempenho rotulado × drift",
        "",
        (
            "As métricas de desempenho exigem rótulos observados. "
            "`drift_detected` vem do resultado do detector; "
            "a comparação não interpreta drift como causa de degradação."
        ),
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in results:
        values = [row.get(column) for column in columns]

        values[2] = "não informado" if values[2] is None else ("sim" if values[2] else "não")

        formatted = [
            "n/a"
            if value is None
            else (
                f"{float(value):.4f}" if isinstance(value, (int, float, np.number)) else str(value)
            )
            for value in values
        ]

        lines.append("| " + " | ".join(formatted) + " |")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia o modelo salvo no split de teste")
    parser.add_argument("--model", type=Path, default=Path("models/baseline_v1.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("models/baseline_v1.json"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/reference_test.parquet"))
    parser.add_argument("--output", type=Path, help="Opcional: salva métricas JSON neste caminho")
    parser.add_argument("--labeled-dir", type=Path, help="Diretório de lotes rotulados de produção")
    parser.add_argument(
        "--perf-report",
        type=Path,
        default=Path("docs/evidence/perf_vs_drift.md"),
        help="Relatório comparativo Markdown para --labeled-dir",
    )
    args = parser.parse_args()

    import joblib

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.split)
    model = joblib.load(args.model)

    if args.labeled_dir:
        batch_frames = {
            path.stem: pd.read_parquet(path) for path in sorted(args.labeled_dir.glob("*.parquet"))
        }

        # Referência independente para detecção de drift.
        reference_drift = pd.read_parquet(Path("data/processed/reference_train.parquet"))

        scored_reference = score_batch(
            reference_drift,
            args.model,
            args.metadata,
        )

        threshold_config = yaml.safe_load(
            Path("config/drift_thresholds.yaml").read_text(encoding="utf-8")
        )
        drift_thresholds = DriftThresholds(**threshold_config)

        drift_by_batch = {}

        for batch_id, batch_frame in batch_frames.items():
            scored_batch = score_batch(
                batch_frame,
                args.model,
                args.metadata,
            )

            context = RunContext(
                run_id=f"evaluate_{batch_id}",
                batch_id=batch_id,
                scenario=batch_id.split("_batch_")[0],
            )

            drift_result = detect_drift(
                reference=scored_reference,
                current=scored_batch,
                cfg=drift_thresholds,
                ctx=context,
            )

            drift_by_batch[batch_id] = {
                "drift_detected": drift_result.dataset_drift,
                "drift_share": drift_result.drift_share,
                "score_psi": drift_result.prediction_drift.score_psi,
                "score_ks": drift_result.prediction_drift.score_ks,
                "mean_score_ref": drift_result.prediction_drift.mean_score_ref,
                "mean_score_cur": drift_result.prediction_drift.mean_score_cur,
            }

        if not batch_frames:
            raise FileNotFoundError(f"Nenhum lote Parquet rotulado em {args.labeled_dir}")
        results = evaluate_labeled_batches(
            model,
            frame,
            batch_frames,
            metadata["features"],
            cfg["target"],
            metadata["threshold"],
            drift_by_batch=drift_by_batch,
        )
        write_perf_vs_drift(results, args.perf_report)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Relatório gravado em {args.perf_report}")
    else:
        result = evaluate_split(
            model, frame, metadata["features"], cfg["target"], metadata["threshold"]
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
