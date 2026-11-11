# Roteiro do vídeo — método STAR (alvo 4:30, limite 5:00)

> Gravar **depois** do code freeze com `make demo-video` (determinístico) e artefatos pré-gerados
> como plano B. Terminal com fonte grande; subir como não listado; testar o link em janela anônima.
> Cada cena aponta o comando ou a tela. Narração: 1–2 pessoas. Nenhuma afirmação sem evidência.

| # | Parte | Tempo | Acum. | Fala (resumo) | Tela / comando |
|---|---|---:|---:|---|---|
| 1 | **S**ituation | 0:25 | 0:25 | Fintech com Credit Scoring em produção. Inflação e novos perfis. A diretoria suspeita de degradação silenciosa: o modelo responde, mas pode estar errando — e em crédito o rótulo chega meses depois. | Slide único: título + diagrama do fluxo (README §2) |
| 2 | **T**ask | 0:25 | 0:50 | Três requisitos: interceptar dados ruins antes do modelo; detectar drift sem esperar o rótulo; garantir compliance (LGPD) e explicar a causa. | Slide com os 3 itens |
| 3 | **A**ction · arquitetura | 0:20 | 1:10 | Pandera para o contrato, Evidently + PSI/KS para drift, MLflow + logs JSON + alertas + dashboard para observabilidade. Referência (treino) separada do pool de produção. | Mermaid do README §2 |
| 4 | **A**ction · contrato bloqueando | 0:40 | 1:50 | Lote com idade 17. O contrato falha na R01, exit code 2, arquivo vai para quarentena, **nenhuma predição é gerada**, alerta A01 CRITICAL. | `uv run python -m creditwatch.pipeline.run --scenario invalid` → `ls data/quarantine` → `tail artifacts/alerts/alerts.jsonl` |
| 5 | **A**ction · simulação de drift | 0:40 | 2:30 | Cenários no YAML: controle sem alteração; moderado infla renda ×1,25 e utilização +0,15; severo adiciona debt_ratio ×1,60 com intensidade progressiva. `age` e `dependents` intocadas como controle. | `config/drift_scenarios.yaml` + `make demo-video` rodando |
| 6 | **A**ction · governança | 0:20 | 2:50 | Dados pseudonimizados, não anonimizados. Base legal adotada: art. 7º V e X; legítimo interesse para monitoramento. Retenção proposta. Logs sem valores de linha — garantido por teste. Não é parecer jurídico. | README §12 rolando; `docs/governance/data_inventory.md` |
| 7 | **R**esult · relatório e alertas | 0:50 | 3:40 | HTML do Evidently apontando `monthly_income`, `revolving_utilization`, `debt_ratio` com drift e `age` sem. `alerts.jsonl` com A02/A03. MLflow: um run por lote, PSI por feature. Dashboard: heatmap de PSI, lotes bloqueados. | Browser: `artifacts/drift/<lote>.html` → MLflow UI → `artifacts/dashboard/index.html` |
| 8 | **R**esult · impacto e limites | 0:30 | 4:10 | Drift de dados ≠ queda de desempenho: no cenário `concept_drift` a performance cai e o relatório quase não reage. Hipótese causal (inflação → renda nominal → modelo subestima risco) rotulada como hipótese, não fato. | `docs/evidence/perf_vs_drift.md` + DAG do README §13 |
| 9 | **R**esult · lições | 0:20 | 4:30 | Contrato barato evita o pior; drift é gatilho de investigação; monitorar com e sem rótulo; governança desde o início. Próximos passos: calibrar thresholds, inferência de rejeitados, retreino gatilhado por alerta. | Slide final + link do repositório |

**Margem:** 0:30.

## Checklist antes de gravar

- [ ] `make demo-video` rodou duas vezes com o mesmo resultado (V-02)
- [ ] Artefatos pré-gerados salvos fora de `artifacts/` como plano B
- [ ] MLflow UI aberta e pré-populada (`make mlflow-ui`)
- [ ] Dashboard gerado (`make dashboard`) e aberto no navegador
- [ ] HTML do Evidently de um lote severo aberto em aba
- [ ] Terminal com fonte ≥ 18 pt, tema claro ou de alto contraste
- [ ] Cada afirmação técnica do roteiro conferida por P3 (drift/causalidade) e P4 (governança)
- [ ] Vídeo ≤ 5:00; link não listado funciona em janela anônima; link colado no README §16
