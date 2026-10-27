# Contratos de integração v1

> **Fonte da verdade do código:** `src/creditwatch/interfaces.py` (modelos pydantic).
> Este documento explica o que os modelos não dizem. Versão **1.0** (contrato de dados **1.0.0**).
> **Mudanças:** PR com a label `contract-change`, aprovado por produtor **e** consumidores. Registrar no changelog abaixo.

## 1. Dataset
- Formato **Parquet**, colunas em **snake_case**. Lotes de produção podem ser Parquet ou CSV.
- Colunas e tipos definidos em `config/data.yaml`. **Nenhum módulo escreve nomes de colunas no código.**
- `customer_id`: texto, pseudonimizado, gerado no `prepare`.
- `event_date`, `batch_id`: só em lotes de produção.
- `target`: `1` = inadimplente (atraso de 90 dias ou mais em 2 anos), `0` = adimplente. **Ausente** nos lotes de produção sem rótulo. Presente apenas nos lotes `labeled` do cenário `concept_drift`.
- Splits (fração do total limpo): `reference_train` 60%, `reference_val` 10%, `reference_test` 10%, `production_pool` 20%. Sem `customer_id` repetido entre splits.
- Saída do scoring: colunas do lote + `score` (float 0 a 1, probabilidade de inadimplência), `prediction` (0 ou 1, pelo threshold de `config/model.yaml`) e `model_version`.

## 2. Quem entrega o quê

| Interface | Produtor | Consumidores | Módulo |
|---|---|---|---|
| Splits em `data/processed/` | P1 | P2, P3 | `data/prepare.py` |
| `ValidationResult` | P2 | P1, P4 | `contracts/validate.py` |
| `BatchRejected` e quarentena | P2 | P1, P4 | `contracts/ingestion.py` |
| `score_batch` | P1 | P1 (pipeline), P3 | `model/predict.py` |
| Lotes + `MANIFEST.json` | P3 | P1, P2 | `simulation/generate.py` |
| `DriftResult` | P3 | P1, P4 | `drift/detector.py` |
| Logger, `RunContext` | P4 | todos | `observability/logging.py` |
| `RunMetrics`, `Alert` | P4 | P1 | `observability/metrics.py`, `alerts.py` |

## 3. Log (JSON Lines em `artifacts/logs/pipeline.jsonl`)
Campos mínimos: `ts` (UTC ISO-8601), `level`, `run_id`, `batch_id`, `stage`, `event`, `status` (`ok|warn|error`), `duration_ms`, `n_rows`, `message`, `extra`.
Estágios: `ingest, validate, score, drift, alert, report, pipeline`.
**Nunca registrar valores de linhas de dados.** Só contagens, estatísticas agregadas e índices de linha.

## 4. Métricas e alertas
- `RunMetrics`: uma linha por execução em `artifacts/metrics/run_history.jsonl`. No MLflow, `latency_s`, `psi` e `ks` viram `latency_<estágio>_s`, `psi_<feature>` e `ks_<feature>`.
- MLflow: experimentos `credit-scoring-training` e `credit-scoring-monitoring`. Nome do run = `batch_id`. Tags: `scenario`, `status`, `model_version`, `contract_version`.
- Alertas em `artifacts/alerts/alerts.jsonl`. Regras A01 a A06 em `config/alerts.yaml`.

## 5. Saídas e códigos de saída
- Gerados (não versionados): `artifacts/{logs,metrics,alerts,validation,drift,scored,dashboard}/`.
- Evidências (versionadas, pequenas): `reports/` e `docs/evidence/`, via `make evidence`.
- Exit codes do pipeline e das CLIs: `0` ok, `2` lote bloqueado pelo contrato, `3` erro de execução.

## 6. Cenários de drift
Definidos em `config/drift_scenarios.yaml`: `no_drift` (controle), `covariate_moderate`, `covariate_severe` e `concept_drift`. `age` e `dependents` ficam sem alteração, como variáveis de controle.

## 7. Thresholds
Valores iniciais por **convenção** (`config/drift_thresholds.yaml`, `config/alerts.yaml`). Serão calibrados em E2-09 e documentados em `docs/thresholds.md`.

## Changelog
| Versão | Data | Mudança | PR |
|---|---|---|---|
| 1.0 | 2026-10-26 | Versão inicial | |