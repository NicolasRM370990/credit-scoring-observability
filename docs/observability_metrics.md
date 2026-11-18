
# Catálogo de Métricas — CreditWatch

## 1. Objetivo

Este documento descreve as métricas utilizadas na observabilidade do pipeline de credit scoring.

O catálogo identifica:

- Nome da métrica.
- Definição.
- Unidade.
- Fonte de dados.
- Regra de alerta associada, quando aplicável.

As métricas são persistidas no histórico operacional (`run_history.jsonl`) e, quando aplicável, registradas no MLflow.

## 2. Fontes de dados

| Fonte | Caminho | Descrição |
|---|---|---|
| Run history | `artifacts/metrics/run_history.jsonl` | Histórico de métricas por execução |
| Alert history | `artifacts/alerts/alerts.jsonl` | Histórico de alertas gerados |
| MLflow | `sqlite:///mlflow/creditwatch.db` | Rastreamento de métricas e metadados |
| Alert configuration | `config/alerts.yaml` | Configuração das regras de alerta |

## 3. Métricas operacionais

| Nome | Definição | Unidade | Fonte | Alerta associado |
|---|---|---|---|---|
| `rows_in` | Quantidade de registros recebidos pelo pipeline | registros | RunMetrics / MLflow | A05_VOLUME |
| `rows_valid` | Quantidade de registros aprovados pelo contrato de dados | registros | RunMetrics / MLflow | — |
| `rows_rejected` | Quantidade de registros rejeitados na validação | registros | RunMetrics / MLflow | A01_VALIDATION_BLOCKED |
| `latency_s.ingest` | Tempo de execução da etapa de ingestão | segundos | RunMetrics | A04_LATENCY, indiretamente |
| `latency_s.validate` | Tempo de execução da etapa de validação | segundos | RunMetrics | A04_LATENCY, indiretamente |
| `latency_s.score` | Tempo de execução da etapa de scoring | segundos | RunMetrics | A04_LATENCY, indiretamente |
| `latency_s.drift` | Tempo de execução da etapa de detecção de drift | segundos | RunMetrics | A04_LATENCY, indiretamente |
| `latency_s.total` | Tempo total de execução do pipeline | segundos | RunMetrics | A04_LATENCY |
| `drift_share` | Proporção de features classificadas como drift | proporção (0–1) | RunMetrics | A02_DATASET_DRIFT |
| `score_mean` | Média dos scores produzidos pelo modelo | score | RunMetrics / MLflow | — |
| `score_psi` | Population Stability Index aplicado à distribuição dos scores | índice PSI | RunMetrics / MLflow | A06_PREDICTION_DRIFT |
| `psi.<feature>` | Population Stability Index calculado para uma feature | índice PSI | RunMetrics / MLflow | A03_FEATURE_PSI |
| `ks.<feature>` | Estatística do teste Kolmogorov-Smirnov para uma feature | estatística (0–1) | RunMetrics / MLflow | — |
| `n_alerts_warning` | Quantidade de alertas com severidade WARNING na execução | alertas | RunMetrics / MLflow | — |
| `n_alerts_critical` | Quantidade de alertas com severidade CRITICAL na execução | alertas | RunMetrics / MLflow | — |

Observações:

- Os nomes `latency_s.*`, `psi.<feature>` e `ks.<feature>` representam estruturas aninhadas no JSONL.
- No MLflow, essas estruturas são convertidas para nomes planos.
- O alerta A04 utiliza especificamente `latency_s.total`, registrado no MLflow como `latency_total_s`.
- O alerta A03 é avaliado individualmente por feature.

## 4. Métricas registradas no MLflow

O tracker utiliza `_flatten_metrics()` para transformar os campos de `RunMetrics` em métricas numéricas do MLflow.

| Nome no MLflow | Origem em RunMetrics | Unidade |
|---|---|---|
| `rows_in` | `rows_in` | registros |
| `rows_valid` | `rows_valid` | registros |
| `rows_rejected` | `rows_rejected` | registros |
| `drift_share` | `drift_share` | proporção (0–1) |
| `score_mean` | `score_mean` | score |
| `score_psi` | `score_psi` | índice PSI |
| `n_alerts_warning` | `n_alerts_warning` | alertas |
| `n_alerts_critical` | `n_alerts_critical` | alertas |
| `latency_<stage>_s` | `latency_s.<stage>` | segundos |
| `psi_<feature>` | `psi.<feature>` | índice PSI |
| `ks_<feature>` | `ks.<feature>` | estatística (0–1) |

As métricas com valor `None` são omitidas pelo tracker.

### 4.1. Tags do MLflow

As seguintes informações são registradas como tags, não como métricas numéricas:

| Tag | Origem | Descrição |
|---|---|---|
| `scenario` | Cenário da execução | Cenário de simulação |
| `status` | `pipeline_status` | Resultado da execução |
| `model_version` | `model_version` | Versão do modelo |
| `contract_version` | `contract_version` | Versão do contrato |
| `run_id` | `run_id` | Identificador da execução |
| `batch_id` | `batch_id` | Identificador do lote |

## 5. Catálogo de alertas

As regras são definidas em `config/alerts.yaml` e avaliadas pelo componente de alertas.

| ID | Nome | Condição | Métrica | Severidade |
|---|---|---|---|---|
| A01 | VALIDATION_BLOCKED | Lote bloqueado pela validação | `validation_blocked` | CRITICAL |
| A02 | DATASET_DRIFT | `drift_share >= 0.30` | `drift_share` | WARNING |
| A03 | FEATURE_PSI | PSI de uma feature `>= 0.017247` | `psi.<feature>` | CRITICAL |
| A04 | LATENCY | Latência total `>= 120` segundos | `latency_total_s` | WARNING |
| A05 | VOLUME | Quantidade de registros `< 1000` ou `> 10000` | `rows_in` | WARNING |
| A06 | PREDICTION_DRIFT | `score_psi >= 0.10` | `score_psi` | WARNING |

### 5.1. Detalhamento das regras

#### A01 — Validation Blocked

Disparado quando a validação bloqueia o lote.

- Severidade: CRITICAL.
- Condição: `validation_blocked = true`.
- Objetivo: impedir que dados inválidos avancem para o scoring.

#### A02 — Dataset Drift

Disparado quando a proporção de features com drift atinge o limite configurado.

- Severidade: WARNING.
- Métrica: `drift_share`.
- Limite: `0.30`.

#### A03 — Feature PSI

Disparado quando o PSI de uma feature atinge o limite configurado.

- Severidade: CRITICAL.
- Métrica: `psi.<feature>`.
- Limite: `0.017247`.

#### A04 — Latency

Disparado quando a latência total atinge o limite configurado.

- Severidade: WARNING.
- Métrica: `latency_total_s`.
- Limite: `120` segundos.

#### A05 — Volume

Disparado quando o volume de registros fica fora da faixa permitida.

- Severidade: WARNING.
- Métrica: `rows_in`.
- Limite inferior: `1000`.
- Limite superior: `10000`.

#### A06 — Prediction Drift

Disparado quando o PSI da distribuição dos scores atinge o limite configurado.

- Severidade: WARNING.
- Métrica: `score_psi`.
- Limite: `0.10`.

## 6. Campos de contexto e rastreabilidade

Os campos abaixo são registrados no histórico, mas não são métricas numéricas do MLflow:

| Campo | Definição |
|---|---|
| `schema_version` | Versão do schema de métricas |
| `run_id` | Identificador único da execução |
| `batch_id` | Identificador do lote |
| `scenario` | Cenário executado |
| `model_version` | Versão do modelo |
| `contract_version` | Versão do contrato de dados |
| `pipeline_status` | Estado final do pipeline |
| `validation_blocked` | Indica se a validação bloqueou o lote |
| `created_at` | Data e hora de criação do registro |

Os alertas possuem seu próprio registro, com os campos:

- `ts`
- `run_id`
- `batch_id`
- `severity`
- `rule_id`
- `message`
- `metric`
- `value`
- `threshold`

## 7. Considerações

1. Os limites apresentados correspondem à configuração atual em `config/alerts.yaml`.
2. Alterações nos limites devem ser refletidas neste catálogo.
3. Métricas calculadas por feature são dinâmicas e dependem das features avaliadas em cada execução.
4. Nem todos os campos de `RunMetrics` são registrados como métricas no MLflow.
5. Tags e campos de rastreabilidade não devem ser contabilizados como métricas numéricas.
6. O catálogo deve permanecer consistente com `RunMetrics`, `_flatten_metrics()` e as regras de alerta.
