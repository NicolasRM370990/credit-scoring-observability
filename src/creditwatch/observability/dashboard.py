"""Dashboard HTML estático (E2-14): saúde do pipeline, drift e alertas em uma página.

Lê `artifacts/metrics/run_history.jsonl` e `artifacts/alerts/alerts.jsonl` e grava
`artifacts/dashboard/index.html` (Jinja2 + Plotly). Por padrão o plotly.js é embutido
para abrir offline; use `--cdn` para um arquivo pequeno que depende de internet.

Só contém agregados por lote: nenhum valor de linha de dados aparece no dashboard.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

from creditwatch.interfaces import Alert, RunMetrics
from creditwatch.observability.alerts import ALERTS_PATH, load_alerts
from creditwatch.observability.metrics import RUN_HISTORY_PATH, load_run_history, summarize_history

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
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px} @media(max-width:1000px){.grid{grid-template-columns:1fr}}
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


def _frame(runs: list[RunMetrics]) -> pd.DataFrame:
    rows = []
    for r in runs:
        rows.append({
            "batch_id": r.batch_id, "scenario": r.scenario or "manual",
            "status": r.pipeline_status, "rows_in": r.rows_in,
            "latency_total_s": r.latency_s.get("total", 0.0),
            "drift_share": r.drift_share, "created_at": r.created_at, **{f"psi_{k}": v for k, v in r.psi.items()},
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("created_at") if not frame.empty else frame


def _figures(frame: pd.DataFrame, include_js: bool | str) -> dict[str, str]:
    """Quatro gráficos Plotly; o primeiro embute o plotly.js, os demais o reutilizam."""
    import plotly.express as px
    import plotly.graph_objects as go

    empty = '<p class="empty">sem dados</p>'
    if frame.empty:
        return {k: empty for k in ("fig_latency", "fig_rows", "fig_drift", "fig_psi")}

    figs: dict[str, Any] = {}
    figs["fig_latency"] = px.bar(frame, x="batch_id", y="latency_total_s", color="scenario")
    figs["fig_rows"] = px.bar(frame, x="batch_id", y="rows_in", color="status",
                              color_discrete_map={"ok": "#047857", "blocked": "#b91c1c",
                                                  "error": "#b45309"})
    drifted = frame.dropna(subset=["drift_share"])
    figs["fig_drift"] = (
        px.line(drifted, x="batch_id", y="drift_share", color="scenario", markers=True)
        if not drifted.empty else None
    )
    psi_cols = [c for c in frame.columns if c.startswith("psi_")]
    if psi_cols:
        matrix = frame.set_index("batch_id")[psi_cols].T
        matrix.index = [c.removeprefix("psi_") for c in matrix.index]
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
        fig.update_layout(margin={"l": 40, "r": 20, "t": 20, "b": 60}, height=320,
                          legend={"orientation": "h", "y": -0.3})
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
    runs = load_run_history(history_path or RUN_HISTORY_PATH)
    alerts: list[Alert] = load_alerts(alerts_path or ALERTS_PATH)
    runs_sorted = sorted(runs, key=lambda r: r.created_at)
    summary = {"runs": 0, "runs_ok": 0, "runs_blocked": 0, "runs_error": 0,
               "rows_in_total": 0, "alerts_warning": 0, "alerts_critical": 0}
    summary.update(summarize_history(history_path or RUN_HISTORY_PATH))
    versions = sorted({r.model_version for r in runs if r.model_version}) or ["-"]

    html = TEMPLATE.render(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        model_versions=", ".join(versions), s=summary, runs=runs_sorted,
        alerts=sorted(alerts, key=lambda a: a.ts),
        **_figures(_frame(runs), "cdn" if cdn else True),
    )
    out = Path(output or DASHBOARD_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera o dashboard HTML de monitoramento")
    parser.add_argument("--output", type=Path, default=DASHBOARD_PATH)
    parser.add_argument("--cdn", action="store_true", help="não embutir plotly.js (arquivo menor)")
    args = parser.parse_args()
    path = build_dashboard(output=args.output, cdn=args.cdn)
    print(f"dashboard gravado em {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
