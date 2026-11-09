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
tests/             # unit, integration, e2e
```

## Conteúdo do README final (a preencher até 11/11)

| Seção | Responsável | Status |
|---|---|---|
| 1. Problema e objetivo | P4 | a fazer |
| 2. Arquitetura | P1 | [documentada](docs/architecture.md) |
| 3. Dataset e limitações (ver [ADR-001](docs/decisions/ADR-001-dataset.md)) | P1 | documentado; sujeito aos termos Kaggle |
| 4. Instalação e 5. Execução | P2 / P1 | comandos de preparo, treino e smoke acima |
| 6. Contratos de dados | P2 | a fazer |
| 7. Modelo baseline | P1 | métricas em `models/baseline_v1.json` |
| Análise descritiva por faixa etária | P1 | `make bias` gera `docs/evidence/bias_report.md` |
| 8. Simulação de drift e 9. Resultados estatísticos | P3 | a fazer |
| 10. Observabilidade e 11. Evidências | P4 | a fazer |
| 12. Governança e LGPD | P4 | a fazer |
| 13. Análise causal | P4 + P3 | a fazer |
| 14. Testes e 15. Limitações | P2 / P3 | a fazer |
| 16. Integrantes e link do vídeo | P4 | a fazer |

## Integrantes

| Pessoa | Papel |
|---|---|
| P1 | Dados, Modelo e Integração |
| P2 | Contratos, Testes e Vídeo |
| P3 | Drift e Simulação |
| P4 | Observabilidade, Governança e Docs |

<!-- Trocar P1 a P4 pelos nomes -->
