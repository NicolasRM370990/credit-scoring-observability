"""Avaliação descritiva de desempenho e aprovação por faixa etária."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from creditwatch.model.predict import score_batch


def analyze_age_groups(frame: pd.DataFrame, scored: pd.DataFrame) -> dict:
    """Resume taxas e métricas por idade, sem inferir causalidade."""
    if "age" not in frame or "target" not in frame:
        raise ValueError("A análise requer as colunas age e target.")
    if not frame.index.equals(scored.index) or len(frame) != len(scored):
        raise ValueError("frame e scored devem ter as mesmas linhas e índices.")
    if "prediction" not in scored or "score" not in scored:
        raise ValueError("scored precisa conter score e prediction.")

    work = pd.DataFrame({
        "age_band": pd.cut(
            frame["age"], bins=[-float("inf"), 29, 44, 59, float("inf")],
            labels=["18-29", "30-44", "45-59", "60+"],
        ),
        "target": frame["target"].to_numpy(),
        "score": scored["score"].to_numpy(),
        "prediction": scored["prediction"].to_numpy(),
    })
    rows: list[dict] = []
    for band, group in work.groupby("age_band", observed=False):
        if group.empty:
            continue
        if group["target"].nunique() == 2:
            roc_auc = float(roc_auc_score(group["target"], group["score"]))
            pr_auc = float(average_precision_score(group["target"], group["score"]))
        else:
            roc_auc, pr_auc = None, None
        approval_rate = float((group["prediction"] == 0).mean())
        positives = group[group["target"] == 1]
        negatives = group[group["target"] == 0]
        recall = float((positives["prediction"] == 1).mean()) if len(positives) else None
        fpr = float((negatives["prediction"] == 1).mean()) if len(negatives) else None
        rows.append({
            "age_band": str(band), "n": len(group),
            "default_rate": float(group["target"].mean()),
            "approval_rate": approval_rate,
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "recall": recall, "false_positive_rate": fpr,
        })
    approval_rates = [row["approval_rate"] for row in rows]
    recalls = [row["recall"] for row in rows if row["recall"] is not None]
    min_approval = min(approval_rates) if approval_rates else None
    return {
        "groups": rows,
        "approval_rate_ratio": min_approval / max(approval_rates) if approval_rates and max(approval_rates) else None,
        "approval_rate_gap": max(approval_rates) - min_approval if approval_rates else None,
        "equal_opportunity_recall_gap": max(recalls) - min(recalls) if recalls else None,
        "n_rows": len(work),
    }


def render_report(result: dict, output: Path) -> None:
    lines = [
        "# Relatório descritivo de desempenho e viés por faixa etária", "",
        f"Amostra avaliada: {result['n_rows']} linhas do conjunto de teste de referência.", "",
        "Atributos sensíveis como sexo e raça não estão disponíveis neste dataset. Faixa etária é uma análise limitada;",
        "não demonstra ausência de viés nem atende, sozinha, a uma avaliação de equidade em produção.", "",
        f"Threshold registrado na metadata: {result['threshold']:.6f}. Aprovação é definida como prediction = 0.", "",
        "## Resultados por faixa", "",
        "| Faixa | N | Inadimplência | Aprovação | ROC-AUC | PR-AUC | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["groups"]:
        vals = [row["age_band"], row["n"]]
        vals.extend("n/a" if row[key] is None else f"{row[key]:.4f}" for key in (
            "default_rate", "approval_rate", "roc_auc", "pr_auc", "recall", "false_positive_rate"
        ))
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines.extend([
        "", "## Diferenças observadas", "",
        f"- Razão entre menor e maior taxa de aprovação: {result['approval_rate_ratio']:.4f}" if result["approval_rate_ratio"] is not None else "- Razão entre taxas de aprovação: n/a",
        f"- Diferença absoluta nas taxas de aprovação: {result['approval_rate_gap']:.4f}" if result["approval_rate_gap"] is not None else "- Diferença nas taxas de aprovação: n/a",
        f"- Diferença de recall entre grupos: {result['equal_opportunity_recall_gap']:.4f}" if result["equal_opportunity_recall_gap"] is not None else "- Diferença de recall: n/a",
        "", "## Limitações e mitigação", "",
        "Essas métricas são descritivas e dependem da amostra, do threshold e da definição operacional de aprovação.",
        "Não são limiares legais nem provam discriminação ou ausência dela. Mudanças de política, seleção de aprovados e",
        "rótulos atrasados podem distorcer a comparação. Em uso real, avaliar atributos permitidos com revisão jurídica,",
        "análise por subgrupos, revisão humana de decisões contestadas e monitoramento periódico das diferenças.", "",
    ])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run(data_path: Path, model_path: Path, output: Path) -> dict:
    metadata_path = model_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(data_path)
    scored = score_batch(frame, model_path, metadata_path)
    result = analyze_age_groups(frame, scored)
    result["threshold"] = metadata["threshold"]
    render_report(result, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera análise de desempenho por idade")
    parser.add_argument("--data", type=Path, default=Path("data/processed/reference_test.parquet"))
    parser.add_argument("--model", type=Path, default=Path("models/baseline_v1.joblib"))
    parser.add_argument("--output", type=Path, default=Path("docs/evidence/bias_report.md"))
    args = parser.parse_args()
    result = run(args.data, args.model, args.output)
    print(f"Relatório salvo em {args.output} ({result['n_rows']} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
