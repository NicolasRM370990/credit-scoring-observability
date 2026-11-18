# Credit Scoring Observability

Tech Challenge Fase 4 (Pós Tech MLET): camada de **sustentação e confiabilidade** para um modelo de Credit Scoring, com contratos de qualidade de dados, simulação e detecção de drift, observabilidade e governança (LGPD).

> **Status:** em desenvolvimento. O plano de execução do time está em [`docs/PLANO_TRABALHO.md`](docs/PLANO_TRABALHO.md) e os contratos de integração em [`docs/contracts.md`](docs/contracts.md).

## Início rápido

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Git. O Python 3.12 é instalado pelo uv.

```bash
git clone https://github.com/NicolasRM370990/credit-scoring-observability.git
cd credit-scoring-observability
make setup        # uv sync
# baixe o cs-training.csv do Kaggle para data/raw/ (ver data/README.md)
make data         # confere o SHA-256 e gera data/processed/full.parquet
make prepare      # gera os quatro splits estratificados
make train        # treina o baseline e grava modelo/métricas
make bias         # gera análise descritiva por faixa etária
make evaluate-labeled # avalia lote rotulado de concept drift quando P3 o disponibilizar
make test
```

`make demo-smoke` valida os estágios e o formato dos logs sem depender das entregas dos demais módulos. `make demo` executa os cenários completos e requer os componentes descritos em `docs/contracts.md`.

O handoff das interfaces P1 para P2/P3/P4 está em [`docs/p1-handoff.md`](docs/p1-handoff.md).

## Estrutura

```text
config/            # YAMLs: dados, contrato, modelo, cenários de drift, thresholds, alertas
data/              # NÃO versionado (ver data/README.md)
docs/              # contratos, ADRs (decisions/), evidências, roteiro do vídeo
notebooks/         # só exploração (EDA)
src/creditwatch/   # data, model, contracts, simulation, drift, observability, pipeline
tests/             # unit, smoke, functional, regression, security, pentest, performance
```

## Conteúdo do README final (a preencher até 11/11)

| Seção | Responsável | Status |
|---|---|---|
| 1. Problema e objetivo | P4 | [abaixo](#1-problema-e-objetivo) |
| 2. Arquitetura | P1 | [documentada](docs/architecture.md) |
| 3. Dataset e limitações (ver [ADR-001](docs/decisions/ADR-001-dataset.md)) | P1 | documentado; sujeito aos termos Kaggle |
| 4. Instalação e 5. Execução | P2 / P1 | comandos de preparo, treino e smoke acima |
| 6. Contratos de dados | P2 | a fazer |
| 7. Modelo baseline | P1 | métricas em `models/baseline_v1.json` |
| Análise descritiva por faixa etária | P1 | `make bias` gera `docs/evidence/bias_report.md` |
| 8. Simulação de drift e 9. Resultados estatísticos | P3 | a fazer |
| 10. Observabilidade e 11. Evidências | P4 | [abaixo](#10-observabilidade) |
| 12. Governança e LGPD | P4 | [abaixo](#12-governança-e-lgpd) |
| 13. Análise causal | P4 + P3 | [abaixo](#13-análise-causal) |
| 14. Testes e 15. Limitações | P2 / P3 | testes [resumidos abaixo](#testes); limitações a fazer |
| 16. Integrantes e link do vídeo | P4 | roteiro e vídeo em `docs/video/`; nomes abaixo |

## Testes

Todas as suítes usam dados sintéticos e rodam sem o CSV do Kaggle. Cada tipo tem um marker do pytest e um alvo no Makefile.

| Suíte | Comando | Testes | O que verifica |
|---|---|---:|---|
| Unitários | `uv run pytest -m "not smoke and not functional and not regression and not security and not pentest and not performance"` | 137 | contrato, simulação, drift, preparo, treino, scoring, avaliação |
| Smoke | `make test-smoke` | 32 | imports (incluindo drift), YAMLs válidos, metadata do modelo, `--help` das CLIs, `--smoke` com log JSON dos 7 estágios |
| Funcional | `make test-functional` | 14 | validar → pontuar → drift → métricas com modelo real; bloqueio antes do scoring; cenários de drift; viés por idade; relatório perf × drift |
| Regressão | `make test-regression` | 14 | ROC-AUC ≥ `min_auc_test`, ganho sobre o dummy, validação ≈ teste, matriz de confusão, alinhamento contrato/features/alertas/cenários |
| Segurança | `make test-security` | 64 | SAST por AST (`eval`/`exec`, `yaml.load`, `shell=True`, `pickle`), segredos nos arquivos versionados, `.gitignore`, LGPD (resultado e log sem valores de linha), modelo adulterado rejeitado, target nunca usado como feature |
| Pentest | `make test-pentest` | 73 | YAML com tags Python, modelo com payload no pickle, path traversal no cenário, injeção em log, entradas hostis (XSS, SQL, Log4Shell, CSV), NaN/±inf, confusão de tipos, *downgrade* do contrato, 25 rodadas de fuzzing |
| Performance | `make test-performance` | 5 | validação de 3 mil linhas < 5 s e de 200 mil < 30 s; scoring de 100 mil < 20 s; cenário severo < 60 s; latência estável |
| Scanners | `make security-scan` | n/a | Bandit (SAST) e pip-audit (CVEs das dependências) |

### Resultado da execução (16/11/2026, Windows, Python 3.12)

- **pytest:** 338 aprovados e 1 pulado (`test_prepare` sem o CSV bruto), em 45 s. Ruff sem apontamentos. O `xfail` do hash do modelo saiu depois da correção.
- **Cobertura:** 75% de `src/creditwatch`, medida em 12/11 antes do módulo de drift.
- **Bandit:** nenhum achado médio ou alto. Restam 2 de severidade baixa em `data/download.py`: import de `subprocess` (B404) e a chamada que baixa o dataset (B603). O executável do Kaggle passou a ser resolvido com `shutil.which` (B607).
- **pip-audit:** as 4 CVEs do `pyasn1` 0.6.4 saíram com a atualização para 0.6.5. Resta `opentelemetry-api` 1.45.1 (CVE-2026-126360, correção na 1.45.2), dependência transitiva do MLflow.

### Correções aplicadas em 16/11

| Gravidade | Ajuste |
|---|---|
| Alta | `score_batch` exige `model_sha256` e confere o arquivo antes de carregar o modelo |
| Média | `pyasn1` atualizado para 0.6.5 no `uv.lock` |
| Baixa | CLI do Kaggle resolvida com `shutil.which`; se não existir, o download não é tentado |

## Integrantes

| Pessoa | Papel |
|---|---|
| P1 | Dados, Modelo e Integração |
| P2 | Contratos, Testes e Vídeo |
| P3 | Drift e Simulação |
| P4 | Observabilidade, Governança e Docs |

<!-- Trocar P1 a P4 pelos nomes -->

## 1. Problema e objetivo

O modelo de credit scoring precisa continuar confiável depois do treino. Esta camada observa cada lote de produção: recusa o que viola o contrato de dados, mede drift de features e de score, registra métricas e alertas, e deixa trilha auditável sem gravar valores de linha (LGPD).

O objetivo da entrega P4 é tornar essa execução observável: log JSON por estágio, catálogo de métricas, dashboard HTML e o inventário de dados. Contratos, drift, alertas, MLflow e o vídeo já estão na `main` e foram preservados.

## 10. Observabilidade

| Peça | Onde | Como ver |
|---|---|---|
| Log por estágio | `src/creditwatch/observability/logging.py` | `artifacts/logs/pipeline.jsonl` |
| Métricas por lote | `observability/metrics.py` (`record_run`) | `artifacts/metrics/run_history.jsonl` |
| Alertas A01–A06 | `observability/alerts.py` | `artifacts/alerts/alerts.jsonl` e `config/alerts.yaml` |
| MLflow | `observability/mlflow_tracker.py` | `make mlflow-ui` (SQLite `mlflow/creditwatch.db`) |
| Dashboard | `observability/dashboard.py` | `make dashboard` → `artifacts/dashboard/index.html` |
| Catálogo | `docs/observability_metrics.md` | nomes, unidades e regra de alerta de cada métrica |

`make alerts-summary` imprime só as contagens dos JSONL já gravados. O dashboard lê esses arquivos; não regrava métricas nem alertas.

## 11. Evidências

As evidências de drift, calibração e viés por faixa etária já estão em `docs/evidence/` (P1/P3). O roteiro e o vídeo da apresentação estão em `docs/video/`. O dashboard e os JSONL de execução ficam fora do Git (`artifacts/`).

## 12. Governança e LGPD

O inventário, a classificação das colunas, a retenção proposta e o controle de acesso estão em [`docs/governance/data_inventory.md`](docs/governance/data_inventory.md).

O que o código garante hoje:

- `ValidationResult` só devolve `row_index`, nunca o valor da célula.
- `sanitize_extra` redige nomes de coluna do dataset e omite DataFrame/Series no log.
- Métricas, alertas e o dashboard guardam agregados por lote (`rows_in`, PSI, `score_mean`), não a linha do cliente.
- Dados brutos, lotes e predições estão no `.gitignore`.

Isso não é parecer jurídico. Prazos de retenção e papéis de acesso são proposta para validar com o negócio.

## 13. Análise causal

O monitoramento responde se a distribuição mudou (`drift_share`, PSI, KS, `score_psi`). Não atribui causa sozinho. A leitura combinada, já produzida pelo grupo, está em `docs/evidence/` e `docs/thresholds.md`: comparar o relatório de drift com o desempenho no lote rotulado (`evaluate-labeled`) e com o recorte por faixa etária (`make bias`). Um PSI alto com AUC estável sugere mudança de mix; queda de desempenho com PSI baixo sugere concept drift. A decisão de retreinar continua humana.

## 16. Integrantes e vídeo

A tabela de papéis está em [Integrantes](#integrantes). Roteiro, apresentação e vídeo já versionados pelo grupo: `docs/video/roteiro_star.md`, `docs/video/creditwatch_apresentacao.pptx` e `docs/video/creditwatch_apresentacao.mp4`. Os nomes ainda precisam substituir os papéis P1–P4.
