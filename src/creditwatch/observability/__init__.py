"""Observabilidade (P4): logs JSON por estágio, métricas por execução, alertas,
MLflow e dashboard HTML.

Módulos:
- `logging`: `get_logger`, `stage_timer`, `@stage` (JSON Lines em artifacts/logs/pipeline.jsonl)
- `metrics`: `record_run`, `load_run_history`, `flatten_metrics`
- `alerts`: `evaluate_alerts` (regras A01-A06 de config/alerts.yaml)
- `mlflow_tracker`: `log_run`, `log_training_run`
- `dashboard`: `build_dashboard` (HTML estático em artifacts/dashboard/index.html)

Os imports são feitos diretamente nos submódulos para que importar o pacote não
carregue o MLflow nem o Plotly.
"""
