"""E4-03 / T15: logs, resultados de validação e alertas não carregam valores de linhas de dados."""
from __future__ import annotations

import json

import pandas as pd

from creditwatch.contracts.validate import validate_batch
from creditwatch.interfaces import RunContext, RunMetrics
from creditwatch.observability.alerts import evaluate_alerts, load_alert_rules
from creditwatch.observability.logging import read_log, stage_timer

# Valores improváveis de aparecerem por acaso em qualquer texto de log
# (a idade inválida 17 provoca o bloqueio, mas não é checada: "17" ocorre em timestamps).
SENTINELS = {"customer_id": "CUST_SENTINEL_7731", "monthly_income": 987654.0,
             "debt_ratio": 123456.789}


def _batch() -> pd.DataFrame:
    frame = pd.DataFrame({
        "customer_id": [f"C{i:07d}" for i in range(1200)],
        "revolving_utilization": 0.3, "age": 40, "past_due_30_59": 0, "debt_ratio": 0.4,
        "monthly_income": 5000.0, "open_credit_lines": 5, "times_90_days_late": 0,
        "real_estate_loans": 1, "past_due_60_89": 0, "dependents": 1,
    })
    # uma linha inválida (idade 17, renda fora do teto) com valores-sentinela
    frame.loc[0, "age"] = 17
    for column, value in SENTINELS.items():
        frame.loc[0, column] = value
    return frame


def test_logs_do_not_contain_row_values(tmp_path):
    frame = _batch()
    ctx = RunContext(run_id="r_gov", batch_id="invalid_sentinel")
    log_path = tmp_path / "pipeline.jsonl"

    with stage_timer("validate", ctx, n_rows=len(frame), log_path=log_path) as log:
        result = validate_batch(frame, ctx.batch_id)
        log.event("validation_finished", "warn", n_rows=result.n_rows,
                  message=result.status,
                  extra={"frame": frame, "first_row": frame.iloc[0].to_dict(),
                         "n_failed": result.n_rows_failed})
    assert result.status == "BLOCKED"

    metrics = RunMetrics(run_id=ctx.run_id, batch_id=ctx.batch_id, pipeline_status="blocked",
                         rows_in=len(frame), rows_rejected=result.n_rows_failed,
                         validation_blocked=True)
    alerts = evaluate_alerts(result, None, metrics, rules=load_alert_rules(),
                             alerts_path=tmp_path / "alerts.jsonl",
                             history_path=tmp_path / "run_history.jsonl", log_path=log_path)
    assert alerts and alerts[0].rule_id == "A01_VALIDATION_BLOCKED"

    corpus = "\n".join([
        json.dumps(read_log(log_path), ensure_ascii=False),
        result.model_dump_json(),
        " ".join(a.model_dump_json() for a in alerts),
    ])
    for value in SENTINELS.values():
        assert str(value) not in corpus, f"valor de linha vazou para os logs: {value}"
    # exemplos de falha trazem só a posição da linha
    for check in result.checks:
        for example in check.examples:
            assert set(example) == {"row_index"}
