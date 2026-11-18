"""Geração do dashboard estático de observabilidade do CreditWatch."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from jinja2 import Template

RUN_HISTORY_PATH = Path("artifacts/metrics/run_history.jsonl")
ALERTS_PATH = Path("artifacts/alerts/alerts.jsonl")
DASHBOARD_PATH = Path("artifacts/dashboard/index.html")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CreditWatch | Dashboard de Observabilidade</title>
    <style>
        :root {
            color-scheme: light;
            --background: #f4f6fa;
            --surface: #ffffff;
            --text: #182230;
            --muted: #667085;
            --border: #e4e7ec;
            --primary: #3157a4;
            --danger: #b42318;
            --warning: #b54708;
            --success: #067647;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 32px;
            background: var(--background);
            color: var(--text);
            font-family: Arial, Helvetica, sans-serif;
        }
        main { max-width: 1440px; margin: auto; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 28px;
        }
        h1 { margin: 0 0 8px; font-size: 28px; }
        h2 { font-size: 18px; margin: 0 0 18px; }
        p { color: var(--muted); }
        .muted { color: var(--muted); font-size: 13px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .card, .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
        }
        .metric-label { color: var(--muted); font-size: 13px; }
        .metric-value {
            font-size: 30px;
            font-weight: 700;
            margin-top: 10px;
        }
        .charts {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .chart { min-width: 0; overflow: hidden; }
        .table-wrap { overflow-x: auto; }
        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            white-space: nowrap;
        }
        th, td {
            padding: 11px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }
        th { color: var(--muted); font-weight: 600; }
        .status {
            display: inline-block;
            border-radius: 20px;
            padding: 4px 9px;
            font-size: 11px;
            font-weight: 700;
        }
        .status-ok { background: #ecfdf3; color: var(--success); }
        .status-blocked { background: #fef3f2; color: var(--danger); }
        .severity-critical { color: var(--danger); font-weight: 700; }
        .severity-warning { color: var(--warning); font-weight: 700; }
        .section { margin-bottom: 20px; }
        footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
        @media (max-width: 900px) {
            body { padding: 16px; }
            .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .charts { grid-template-columns: 1fr; }
            header { align-items: flex-start; flex-direction: column; }
        }
        @media (max-width: 480px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<main>
    <header>
        <div>
            <h1>CreditWatch</h1>
            <p>Dashboard de observabilidade do pipeline de credit scoring</p>
        </div>
        <div class="muted">
            Gerado em {{ generated_at }}<br>
            Fonte: run_history.jsonl e alerts.jsonl
        </div>
    </header>

    <section class="grid">
        <div class="card">
            <div class="metric-label">Execuções registradas</div>
            <div class="metric-value">{{ total_runs }}</div>
        </div>
        <div class="card">
            <div class="metric-label">Lotes bloqueados</div>
            <div class="metric-value">{{ blocked_runs }}</div>
        </div>
        <div class="card">
            <div class="metric-label">Linhas processadas</div>
            <div class="metric-value">{{ total_rows }}</div>
        </div>
        <div class="card">
            <div class="metric-label">Alertas registrados</div>
            <div class="metric-value">{{ total_alerts }}</div>
            <div class="muted">
                {{ critical_alerts }} críticos · {{ warning_alerts }} avisos
            </div>
        </div>
    </section>

    <section class="charts">
        <div class="panel chart">
            <h2>Distribuição de status</h2>
            {{ status_chart | safe }}
        </div>
        <div class="panel chart">
            <h2>Drift share por lote</h2>
            {{ drift_chart | safe }}
        </div>
        <div class="panel chart">
            <h2>PSI máximo por lote</h2>
            {{ psi_chart | safe }}
        </div>
        <div class="panel chart">
            <h2>Duração das execuções</h2>
            {{ duration_chart | safe }}
        </div>
    </section>

    <section class="panel section">
        <h2>Histórico de execuções</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Data</th><th>Cenário</th><th>Lote</th><th>Status</th>
                        <th>Linhas</th><th>Duração (s)</th><th>Drift share</th>
                        <th>PSI máximo</th><th>Alertas</th>
                    </tr>
                </thead>
                <tbody>
                {% for run in runs %}
                    <tr>
                        <td>{{ run.created_at }}</td>
                        <td>{{ run.scenario }}</td>
                        <td>{{ run.batch_id }}</td>
                        <td>
                            <span class="status status-{{ run.status_class }}">
                                {{ run.status }}
                            </span>
                        </td>
                        <td>{{ run.rows_in }}</td>
                        <td>{{ run.duration }}</td>
                        <td>{{ run.drift_share }}</td>
                        <td>{{ run.max_psi }}</td>
                        <td>{{ run.alert_count }}</td>
                    </tr>
                {% else %}
                    <tr><td colspan="9">Nenhuma execução encontrada.</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </section>

    <section class="panel section">
        <h2>Alertas registrados</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Data</th><th>Severidade</th><th>Regra</th>
                        <th>Lote</th><th>Métrica</th><th>Valor</th>
                        <th>Limite</th><th>Mensagem</th>
                    </tr>
                </thead>
                <tbody>
                {% for alert in alerts %}
                    <tr>
                        <td>{{ alert.ts }}</td>
                        <td class="severity-{{ alert.severity_class }}">
                            {{ alert.severity }}
                        </td>
                        <td>{{ alert.rule_id }}</td>
                        <td>{{ alert.batch_id }}</td>
                        <td>{{ alert.metric }}</td>
                        <td>{{ alert.value }}</td>
                        <td>{{ alert.threshold }}</td>
                        <td>{{ alert.message }}</td>
                    </tr>
                {% else %}
                    <tr><td colspan="8">Nenhum alerta registrado.</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </section>

    <footer>
        CreditWatch · Relatório estático gerado localmente ·
        Os dados apresentados correspondem aos registros disponíveis nos arquivos JSONL.
    </footer>
</main>
</body>
</html>
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lê JSONL sem interromper o dashboard por linhas inválidas."""
    if not path.is_file():
        return []

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(item, dict):
                records.append(item)

    return records


def _format_number(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _max_psi(psi_values: Any) -> float | None:
    if not isinstance(psi_values, dict) or not psi_values:
        return None

    values = []
    for value in psi_values.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    return max(values) if values else None


def _prepare_runs(
    records: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normaliza execuções para apresentação no dashboard."""
    alerts_by_run: Counter[str] = Counter()

    for alert in alerts:
        run_id = alert.get("run_id")
        if run_id:
            alerts_by_run[str(run_id)] += 1

    prepared = []

    for record in records:
        status = str(record.get("pipeline_status") or "unknown").lower()
        latency = record.get("latency_s") or {}
        if not isinstance(latency, dict):
            latency = {}

        psi = record.get("psi") or {}
        duration = latency.get("total")
        max_psi = _max_psi(psi)

        prepared.append(
            {
                "run_id": str(record.get("run_id") or ""),
                "batch_id": str(record.get("batch_id") or "—"),
                "scenario": str(record.get("scenario") or "—"),
                "status": status,
                "status_class": "blocked" if status == "blocked" else "ok",
                "created_at": str(record.get("created_at") or "—"),
                "rows_in": int(record.get("rows_in") or 0),
                "duration": _format_number(duration, 2),
                "duration_value": float(duration) if duration is not None else None,
                "drift_share": record.get("drift_share"),
                "max_psi": max_psi,
                "alert_count": alerts_by_run[str(record.get("run_id") or "")],
            }
        )

    def sort_key(item: dict[str, Any]) -> str:
        return item["created_at"]

    return sorted(prepared, key=sort_key)


def _prepare_alerts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []

    for alert in records:
        severity = str(alert.get("severity") or "UNKNOWN").lower()

        prepared.append(
            {
                "ts": str(alert.get("ts") or "—"),
                "severity": severity.upper(),
                "severity_class": severity,
                "rule_id": str(alert.get("rule_id") or "—"),
                "batch_id": str(alert.get("batch_id") or "—"),
                "metric": str(alert.get("metric") or "—"),
                "value": _format_number(alert.get("value")),
                "threshold": _format_number(alert.get("threshold")),
                "message": str(alert.get("message") or "—"),
            }
        )

    return sorted(prepared, key=lambda item: item["ts"], reverse=True)


def _chart_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if include_plotlyjs else False,
        config={"responsive": True, "displayModeBar": False},
    )


def _build_charts(runs: list[dict[str, Any]]) -> dict[str, str]:
    labels = [run["batch_id"] for run in runs]

    status_counts = Counter(run["status"] for run in runs)
    status_fig = go.Figure(
        data=[
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                hole=0.58,
            )
        ]
    )
    status_fig.update_layout(height=300, margin={"t": 10, "b": 10, "l": 10, "r": 10})

    drift_fig = go.Figure()
    drift_fig.add_trace(
        go.Bar(
            x=labels,
            y=[run["drift_share"] for run in runs],
            name="drift_share",
        )
    )
    drift_fig.update_layout(
        height=300,
        margin={"t": 10, "b": 60, "l": 40, "r": 10},
        yaxis_title="Proporção",
        xaxis_title="Lote",
    )

    psi_fig = go.Figure()
    psi_fig.add_trace(
        go.Bar(
            x=labels,
            y=[run["max_psi"] for run in runs],
            name="PSI máximo",
        )
    )
    psi_fig.update_layout(
        height=300,
        margin={"t": 10, "b": 60, "l": 40, "r": 10},
        yaxis_title="PSI",
        xaxis_title="Lote",
    )

    duration_fig = go.Figure()
    duration_fig.add_trace(
        go.Bar(
            x=labels,
            y=[run["duration_value"] for run in runs],
            name="Duração total",
        )
    )
    duration_fig.update_layout(
        height=300,
        margin={"t": 10, "b": 60, "l": 40, "r": 10},
        yaxis_title="Segundos",
        xaxis_title="Lote",
    )

    return {
        "status_chart": _chart_html(status_fig, True),
        "drift_chart": _chart_html(drift_fig, False),
        "psi_chart": _chart_html(psi_fig, False),
        "duration_chart": _chart_html(duration_fig, False),
    }


def generate_dashboard(
    run_history_path: Path = RUN_HISTORY_PATH,
    alerts_path: Path = ALERTS_PATH,
    output_path: Path = DASHBOARD_PATH,
) -> Path:
    """Gera dashboard HTML estático e autocontido."""
    run_records = _read_jsonl(run_history_path)
    alert_records = _read_jsonl(alerts_path)

    runs = _prepare_runs(run_records, alert_records)
    alerts = _prepare_alerts(alert_records)

    total_rows = sum(run["rows_in"] for run in runs)
    blocked_runs = sum(run["status"] == "blocked" for run in runs)
    critical_alerts = sum(alert["severity"] == "CRITICAL" for alert in alerts)
    warning_alerts = sum(alert["severity"] == "WARNING" for alert in alerts)

    context = {
        "generated_at": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S"),
        "total_runs": len(runs),
        "blocked_runs": blocked_runs,
        "total_rows": total_rows,
        "total_alerts": len(alerts),
        "critical_alerts": critical_alerts,
        "warning_alerts": warning_alerts,
        "runs": runs,
        "alerts": alerts,
        **_build_charts(runs),
    }

    html = Template(HTML_TEMPLATE).render(**context)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def main() -> int:
    output = generate_dashboard()
    print(f"Dashboard gerado: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
