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
| 14. Testes e 15. Limitações | P2 / P3 | testes [resumidos abaixo](#testes); limitações a fazer |
| 16. Integrantes e link do vídeo | P4 | a fazer |

## Testes

Todas as suítes usam dados sintéticos e rodam sem o CSV do Kaggle. Cada tipo tem um marker do pytest e um alvo no Makefile.

| Suíte | Comando | Testes | O que verifica |
|---|---|---:|---|
| Unitários (existentes) | `make test` | 82 | contrato, simulação, preparo, treino, scoring, avaliação |
| Smoke | `make test-smoke` | 27 | imports, YAMLs válidos, metadata do modelo, `--help` das CLIs, `--smoke` com log JSON dos 7 estágios |
| Funcional | `make test-functional` | 14 | validar → pontuar → drift → métricas com modelo real; bloqueio antes do scoring; cenários de drift; viés por idade; relatório perf × drift |
| Regressão | `make test-regression` | 14 | ROC-AUC ≥ `min_auc_test`, ganho sobre o dummy, validação ≈ teste, matriz de confusão, alinhamento contrato/features/alertas/cenários |
| Segurança | `make test-security` | 64 | SAST por AST (`eval`/`exec`, `yaml.load`, `shell=True`, `pickle`), segredos nos arquivos versionados, `.gitignore`, LGPD (resultado e log sem valores de linha), modelo adulterado rejeitado, target nunca usado como feature |
| Pentest | `make test-pentest` | 73 | YAML com tags Python, modelo com payload no pickle, path traversal no cenário, injeção em log, entradas hostis (XSS, SQL, Log4Shell, CSV), NaN/±inf, confusão de tipos, *downgrade* do contrato, 25 rodadas de fuzzing |
| Performance | `make test-performance` | 5 | validação de 3 mil linhas < 5 s e de 200 mil < 30 s; scoring de 100 mil < 20 s; cenário severo < 60 s; latência estável |
| Scanners | `make security-scan` | n/a | Bandit (SAST) e pip-audit (CVEs das dependências) |

### Resultado da execução (12/11/2026, Windows, Python 3.12)

- **pytest:** 277 aprovados, 1 pulado (`test_prepare` sem o CSV bruto) e 1 `xfail` (achado abaixo), em 61 s. Ruff sem apontamentos.
- **Cobertura:** 75% de `src/creditwatch` (`interfaces` 100%, `contracts/validate` 91%, `simulation/generate` 88%; o mais baixo é `data/download`, com 22%, porque depende do Kaggle).
- **Bandit:** nenhum achado médio ou alto. São 3 de severidade baixa, todos na chamada `subprocess` da CLI do Kaggle em `data/download.py`, que já usa lista de argumentos e não usa shell.
- **pip-audit:** 4 CVEs no `pyasn1` 0.6.4 (CVE-2026-119194 a 119197), corrigidas na 0.6.5. A dependência é transitiva: `mlflow` → `databricks-sdk` → `google-auth` → `pyasn1-modules`.

### Achados para corrigir

| Gravidade | Achado | Evidência | Correção sugerida |
|---|---|---|---|
| Alta | `score_batch` só confere o SHA-256 se a metadata tiver `model_sha256`. Sem esse campo, o `.joblib` é desserializado sem verificação, e um modelo malicioso executa código | `tests/pentest/test_pentest.py::test_metadata_without_hash_must_not_load_untrusted_model` (`xfail` estrito) | Tornar `model_sha256` obrigatório em `model/predict.py`. Quando corrigido, o `xfail` vira falha e o marcador deve sair |
| Média | `pyasn1` 0.6.4 com CVEs conhecidas | pip-audit | `uv lock --upgrade-package pyasn1` |
| Baixa | CLI do Kaggle chamada pelo nome, resolvida pelo `PATH` (Bandit B607) | Bandit | Resolver o executável com `shutil.which` e falhar se não existir |

## Integrantes

| Pessoa | Papel |
|---|---|
| P1 | Dados, Modelo e Integração |
| P2 | Contratos, Testes e Vídeo |
| P3 | Drift e Simulação |
| P4 | Observabilidade, Governança e Docs |

<!-- Trocar P1 a P4 pelos nomes -->
