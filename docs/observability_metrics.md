# Catálogo de métricas de observabilidade (E2-15)

Fonte da verdade do schema: `RunMetrics` em `src/creditwatch/interfaces.py`.
Cada execução do pipeline (um lote) gera **uma linha** em `artifacts/metrics/run_history.jsonl`,
**um run** no MLflow (experimento `credit-scoring-monitoring`) e zero ou mais linhas em
`artifacts/alerts/alerts.jsonl`. O dashboard (`make dashboard`) lê esses dois arquivos.

O teste `tests/unit/test_metrics_catalog.py` confere que todo campo numérico de `RunMetrics`
aparece nesta tabela com o mesmo nome usado no MLflow (`metrics.flatten_metrics`).

## 1. Métricas de saúde do pipeline

| Nome (run_history / MLflow) | Definição | Unidade | Fonte | Alerta |
|---|---|---|---|---|
| `pipeline_status` | Resultado da execução: `ok`, `blocked` (contrato rejeitou o lote), `error` (exceção) | categoria (tag no MLflow) | `pipeline/run.py` | A01 quando `blocked` |
| `rows_in` | Linhas lidas do lote antes da validação | linhas | `ingest_batch` | A05 fora de `[1000, 10000]` |
| `rows_valid` | Linhas que passaram no contrato e foram pontuadas | linhas | `validate_batch` | — |
| `rows_rejected` | Linhas com falha em regra BLOCK (ou 1 quando o arquivo é rejeitado antes da leitura) | linhas | `ValidationResult.n_rows_failed` | — |
| `validation_blocked` | 1 se o contrato bloqueou o lote, 0 caso contrário | 0/1 | `ValidationResult.status == BLOCKED` | **A01 CRITICAL** |
| `latency_total_s` | Tempo de parede do lote, da ingestão ao registro de métricas | segundos | `time.perf_counter` em `run.py` | A04 > 120 s |
| `latency_<estágio>_s` | Tempo de cada estágio (`ingest`, `validate`, `score`, `drift`), quando medido | segundos | `RunMetrics.latency_s` (dict) | — |

## 2. Métricas de drift do modelo

| Nome | Definição | Unidade | Fonte | Alerta |
|---|---|---|---|---|
| `drift_share` | Fração das features com `drifted = true` | proporção 0–1 | `DriftResult.drift_share` (P3) | A02 ≥ 0,30 |
| `psi_<feature>` | Population Stability Index da feature: referência (treino) × lote atual, bins por quantis da referência | adimensional (0 = idêntico) | `DriftResult.features[].psi` | **A03 CRITICAL ≥ 0,25**; WARNING ≥ 0,10 (convenção de mercado, ver `docs/thresholds.md`) |
| `ks_<feature>` | Estatística de Kolmogorov–Smirnov (distância máxima entre CDFs) da feature | 0–1 | `DriftResult.features[].ks_stat` | informativo; `ks_stat_min = 0,10` entra na decisão de `drifted` |
| `score_mean` | Média dos scores (probabilidade de inadimplência) no lote | proporção 0–1 | `score_batch` | — (comparar com `mean_score_ref`) |
| `score_psi` | PSI da distribuição de scores: referência × lote | adimensional | `DriftResult.prediction_drift.score_psi` | A06 > 0,10 |

## 3. Métricas de alertas

| Nome | Definição | Unidade | Fonte | Alerta |
|---|---|---|---|---|
| `n_alerts_warning` | Alertas WARNING emitidos para a execução | contagem | `evaluate_alerts` | — |
| `n_alerts_critical` | Alertas CRITICAL emitidos para a execução | contagem | `evaluate_alerts` | — |

## 4. Dimensões (tags/params no MLflow, colunas no run_history)

`run_id`, `batch_id`, `scenario`, `model_version`, `contract_version`, `schema_version`, `created_at`.

## 5. Regras de alerta (`config/alerts.yaml`)

| Regra | Severidade | Condição | Ação esperada |
|---|---|---|---|
| `A01_VALIDATION_BLOCKED` | CRITICAL | lote bloqueado pelo contrato | investigar a origem dos dados; lote fica em `data/quarantine/`; nenhuma predição é gerada |
| `A02_DATASET_DRIFT` | WARNING | `drift_share ≥ 0,30` ou Evidently marcou `dataset_drift` | abrir o relatório HTML do lote; avaliar retreino |
| `A03_FEATURE_PSI` | CRITICAL | `psi_<feature> ≥ 0,25` (um alerta por feature) | confirmar com a área de negócio se a mudança é real; considerar recalibração |
| `A04_LATENCY` | WARNING | `latency_total_s > 120` | investigar infraestrutura/volume |
| `A05_VOLUME` | WARNING | `rows_in < 1000` ou `> 10000` | verificar extração/particionamento do lote |
| `A06_PREDICTION_DRIFT` | WARNING | `score_psi > 0,10` | comparar `score_mean` com a referência; checar taxa de aprovação |

Os thresholds são **convenção inicial** (PSI 0,10/0,25 é prática de mercado, não regra
estatística). A calibração com amostras sem drift está prevista em E2-09 (`docs/thresholds.md`).

## 6. Logs (`artifacts/logs/pipeline.jsonl`)

Uma linha JSON por evento com os campos de `LOG_REQUIRED_FIELDS`:
`ts, level, run_id, batch_id, stage, event, status, duration_ms, n_rows, message, extra`.
Estágios: `ingest, validate, score, drift, alert, report, pipeline`. Eventos padrão:
`stage_started`, `stage_finished`, `stage_failed`, `alerts_evaluated`.
**Nenhum valor de linha de dados é gravado** (ver `docs/governance/data_inventory.md` e o teste
`tests/unit/test_governance_logs.py`).

## 7. Onde ver

| Visão | Comando | Saída |
|---|---|---|
| Histórico tabular | `make alerts-summary` | resumo JSON de alertas e execuções |
| MLflow UI | `make mlflow-ui` → http://127.0.0.1:5000 | um run por lote, métricas acima, HTML do Evidently e `alerts.json` como artefatos |
| Dashboard offline | `make dashboard` → `artifacts/dashboard/index.html` | cards de saúde, latência, volume, `drift_share`, heatmap de PSI, tabelas de execuções e alertas |
| Evidências versionadas | `make evidence` | cópia leve em `docs/evidence/` |
