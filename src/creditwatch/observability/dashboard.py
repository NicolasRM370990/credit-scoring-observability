"""Dashboard HTML estático: saúde do pipeline, drift e alertas em uma página.

Lê os JSONL já gravados por `metrics.record_run` e `alerts.evaluate_alerts`
(`artifacts/metrics/run_history.jsonl` e `artifacts/alerts/alerts.jsonl`) e grava
`artifacts/dashboard/index.html`. Por padrão o plotly.js é embutido para abrir
offline; use `--cdn` para um arquivo menor que depende de internet.

Não altera os módulos de métricas, alertas ou MLflow já entregues na main.
Só contém agregados por lote: nenhum valor de linha de dados aparece no dashboard.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

from creditwatch.interfaces import Alert, RunMetrics
from creditwatch.observability.alerts import ALERTS_LOG_PATH
from creditwatch.observability.metrics import METRICS_LOG_PATH

DASHBOARD_PATH = Path("artifacts/dashboard/index.html")

TEMPLATE = Template(
    """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>CreditWatch · Dashboard de monitoramento</title>
<style>
 body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:24px;color:#1f2937;background:#f9fafb}
 h1{margin-bottom:0} .sub{color:#6b7280;margin-top:4px}
 .cards{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}
 .card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;min-width:150px}
 .card b{display:block;font-size:26px} .card span{color:#6b7280;font-size:13px}
 .ok b{color:#047857} .warn b{color:#b45309} .crit b{color:#b91c1c}
 table{border-collapse:collapse;width:100%;background:#fff;font-size:13px}
 th,td{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left}
 th{background:#f3f4f6} tr.blocked td{background:#fef2f2} tr.error td{background:#fff7ed}
 .sev-CRITICAL{color:#b91c1c;font-weight:600} .sev-WARNING{color:#b45309;font-weight:600}
 section{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin:16px 0}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 @media(max-width:1000px){.grid{grid-template-columns:1fr}}
 .empty{color:#6b7280;font-style:italic}
</style></head><body>
<h1>CreditWatch · Dashboard de monitoramento</h1>
<p class="sub">Gerado em {{ generated_at }} · fonte: run_history.jsonl e alerts.jsonl · modelo {{ model_versions }}</p>
<div class="cards">
 <div class="card"><b>{{ s.runs }}</b><span>execuções</span></div>
 <div class="card ok"><b>{{ s.runs_ok }}</b><span>ok</span></div>
 <div class="card crit"><b>{{ s.runs_blocked }}</b><span>bloqueadas pelo contrato</span></div>
 <div class="card warn"><b>{{ s.runs_error }}</b><span>com erro</span></div>
 <div class="card"><b>{{ "{:,}".format(s.rows_in_total).replace(",", ".") }}</b><span>linhas processadas</span></div>
 <div class="card warn"><b>{{ s.alerts_warning }}</b><span>alertas WARNING</span></div>
 <div class="card crit"><b>{{ s.alerts_critical }}</b><span>alertas CRITICAL</span></div>
</div>
{% if not runs %}<section><p class="empty">Nenhuma execução registrada ainda. Rode <code>make demo</code>.</p></section>{% endif %}
<div class="grid">
 <section><h3>Latência total por lote (s)</h3>{{ fig_latency }}</section>
 <section><h3>Linhas recebidas por lote</h3>{{ fig_rows }}</section>
 <section><h3>drift_share por lote e cenário</h3>{{ fig_drift }}</section>
 <section><h3>PSI por feature e lote</h3>{{ fig_psi }}</section>
</div>
<section><h3>Execuções</h3>
<table><thead><tr><th>criado em (UTC)</th><th>cenário</th><th>lote</th><th>status</th><th>linhas</th>
<th>rejeitadas</th><th>latência (s)</th><th>drift_share</th><th>score médio</th><th>score PSI</th><th>WARN</th><th>CRIT</th></tr></thead>
<tbody>
{% for r in runs %}<tr class="{{ r.pipeline_status }}">
<td>{{ r.created_at.strftime("%Y-%m-%d %H:%M:%S") }}</td><td>{{ r.scenario or "-" }}</td><td>{{ r.batch_id }}</td>
<td>{{ r.pipeline_status }}</td><td>{{ r.rows_in }}</td><td>{{ r.rows_rejected }}</td>
<td>{{ "%.2f"|format(r.latency_s.get("total", 0)) }}</td>
<td>{{ "%.3f"|format(r.drift_share) if r.drift_share is not none else "-" }}</td>
<td>{{ "%.3f"|format(r.score_mean) if r.score_mean is not none else "-" }}</td>
<td>{{ "%.3f"|format(r.score_psi) if r.score_psi is not none else "-" }}</td>
<td>{{ r.n_alerts_warning }}</td><td>{{ r.n_alerts_critical }}</td></tr>
{% endfor %}</tbody></table></section>
<section><h3>Alertas ({{ alerts|length }})</h3>
{% if alerts %}<table><thead><tr><th>ts (UTC)</th><th>lote</th><th>severidade</th><th>regra</th><th>métrica</th><th>valor</th><th>limite</th><th>mensagem</th></tr></thead>
<tbody>{% for a in alerts %}<tr>
<td>{{ a.ts.strftime("%Y-%m-%d %H:%M:%S") }}</td><td>{{ a.batch_id }}</td>
<td class="sev-{{ a.severity }}">{{ a.severity }}</td><td>{{ a.rule_id }}</td><td>{{ a.metric or "-" }}</td>
<td>{{ "%.3f"|format(a.value) if a.value is not none else "-" }}</td>
<td>{{ a.threshold if a.threshold is not none else "-" }}</td><td>{{ a.message }}</td></tr>
{% endfor %}</tbody></table>{% else %}<p class="empty">Nenhum alerta emitido.</p>{% endif %}
</section>
<p class="sub">Métricas documentadas em <code>docs/observability_metrics.md</code>. Thresholds em <code>config/alerts.yaml</code>.</p>
</body></html>
"""
)


def _read_models(path: Path, model: type[RunMetrics | Alert]) -> list[Any]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [model.model_validate(json.loads(line)) for line in stream if line.strip()]


def _summary(runs: list[RunMetrics]) -> dict[str, int]:
    return {
        "runs": len(runs),
        "runs_ok": sum(row.pipeline_status == "ok" for row in runs),
        "runs_blocked": sum(row.pipeline_status == "blocked" for row in runs),
        "runs_error": sum(row.pipeline_status == "error" for row in runs),
        "rows_in_total": sum(row.rows_in for row in runs),
        "alerts_warning": sum(row.n_alerts_warning for row in runs),
        "alerts_critical": sum(row.n_alerts_critical for row in runs),
    }


def _frame(runs: list[RunMetrics]) -> pd.DataFrame:
    rows = [
        {
            "batch_id": run.batch_id,
            "scenario": run.scenario or "manual",
            "status": run.pipeline_status,
            "rows_in": run.rows_in,
            "latency_total_s": run.latency_s.get("total", 0.0),
            "drift_share": run.drift_share,
            "created_at": run.created_at,
            **{f"psi_{name}": value for name, value in run.psi.items()},
        }
        for run in runs
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values("created_at") if not frame.empty else frame


def _figures(frame: pd.DataFrame, include_js: bool | str) -> dict[str, str]:
    import plotly.express as px
    import plotly.graph_objects as go

    empty = '<p class="empty">sem dados</p>'
    if frame.empty:
        return {key: empty for key in ("fig_latency", "fig_rows", "fig_drift", "fig_psi")}

    figs: dict[str, Any] = {
        "fig_latency": px.bar(frame, x="batch_id", y="latency_total_s", color="scenario"),
        "fig_rows": px.bar(
            frame, x="batch_id", y="rows_in", color="status",
            color_discrete_map={"ok": "#047857", "blocked": "#b91c1c", "error": "#b45309"},
        ),
    }
    drifted = frame.dropna(subset=["drift_share"])
    figs["fig_drift"] = (
        px.line(drifted, x="batch_id", y="drift_share", color="scenario", markers=True)
        if not drifted.empty else None
    )
    psi_cols = [column for column in frame.columns if column.startswith("psi_")]
    if psi_cols:
        matrix = frame.set_index("batch_id")[psi_cols].T
        matrix.index = [column.removeprefix("psi_") for column in matrix.index]
        figs["fig_psi"] = go.Figure(go.Heatmap(
            z=matrix.values, x=list(matrix.columns), y=list(matrix.index),
            colorscale="YlOrRd", zmin=0, zmax=0.5, colorbar={"title": "PSI"},
        ))
    else:
        figs["fig_psi"] = None

    html: dict[str, str] = {}
    first = True
    for key, fig in figs.items():
        if fig is None:
            html[key] = empty
            continue
        fig.update_layout(
            margin={"l": 40, "r": 20, "t": 20, "b": 60}, height=320,
            legend={"orientation": "h", "y": -0.3},
        )
        html[key] = fig.to_html(full_html=False, include_plotlyjs=include_js if first else False)
        first = False
    return html


def build_dashboard(
    history_path: Path | None = None,
    alerts_path: Path | None = None,
    output: Path | None = None,
    *,
    cdn: bool = False,
) -> Path:
    """Gera o HTML e devolve o caminho gravado."""
    runs = _read_models(Path(history_path or METRICS_LOG_PATH), RunMetrics)
    alerts = _read_models(Path(alerts_path or ALERTS_LOG_PATH), Alert)
    versions = sorted({run.model_version for run in runs if run.model_version}) or ["-"]
    html = TEMPLATE.render(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        model_versions=", ".join(versions),
        s=_summary(runs),
        runs=sorted(runs, key=lambda run: run.created_at),
        alerts=sorted(alerts, key=lambda alert: alert.ts),
        **_figures(_frame(runs), "cdn" if cdn else True),
    )
    out = Path(output or DASHBOARD_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def print_summary(history_path: Path | None = None, alerts_path: Path | None = None) -> None:
    """Imprime contagens dos JSONL já gravados, sem gerar o HTML."""
    runs = _read_models(Path(history_path or METRICS_LOG_PATH), RunMetrics)
    alerts = _read_models(Path(alerts_path or ALERTS_LOG_PATH), Alert)
    counts = _summary(runs)
    print(json.dumps({**counts, "alerts": len(alerts)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera o dashboard HTML de monitoramento")
    parser.add_argument("--output", type=Path, default=DASHBOARD_PATH)
    parser.add_argument("--cdn", action="store_true", help="não embutir plotly.js (arquivo menor)")
    parser.add_argument("--summary", action="store_true", help="só imprime contagens, sem HTML")
    args = parser.parse_args()
    if args.summary:
        print_summary()
        return 0
    path = build_dashboard(output=args.output, cdn=args.cdn)
    print(f"dashboard gravado em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
