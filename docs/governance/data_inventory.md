# Inventário de dados e tratamento (E4-01)

> Parte do plano de proteção de dados do projeto (README §12). Este documento descreve o
> protótipo com o dataset público *Give Me Some Credit* e o que **mudaria com dados reais de
> clientes**. Não é parecer jurídico.

## 1. Classificação por coluna

Categorias usadas (LGPD, art. 5º, I e II; art. 12 para pseudonimização):
**ID** = identificador direto · **QI** = quase-identificador (identifica em combinação) ·
**S** = dado pessoal sensível (art. 5º, II) · **P** = dado pessoal não sensível ·
**NP** = não pessoal / derivado.

| Coluna (snake_case) | Origem | Classe | Justificativa | Onde aparece | Tratamento no projeto |
|---|---|---|---|---|---|
| `customer_id` | gerado em `prepare` | ID (pseudônimo) | Chave sintética sem relação com qualquer identificador real; o dataset original não traz nome, CPF ou conta | processed, lotes, scored | Pseudonimização. Nunca entra em logs, alertas ou relatórios; só contagens de duplicidade |
| `age` | `age` | QI / P | Idade combinada com renda e dependentes pode reidentificar em bases pequenas | processed, lotes, scored, **agregada** em drift/viés | Contrato R01 (18–110). Relatórios só por faixa etária ou estatística agregada |
| `monthly_income` | `MonthlyIncome` | QI / P | Dado financeiro pessoal; forte quase-identificador | processed, lotes, scored, agregada em drift | Contrato R02 (não nulo, 0–25.000); só PSI/KS e médias em relatórios |
| `dependents` | `NumberOfDependents` | QI / P | Informação familiar; identifica em combinação | processed, lotes, scored, agregada | Variável de controle da simulação (não alterada); só agregados |
| `revolving_utilization` | `RevolvingUtilization...` | P | Comportamento financeiro | processed, lotes, scored, agregada | R05 (≥ 0); agregados |
| `debt_ratio` | `DebtRatio` | P | Comportamento financeiro | idem | R05; agregados |
| `open_credit_lines` | `NumberOfOpenCreditLinesAndLoans` | P | Comportamento financeiro | idem | R05; agregados |
| `real_estate_loans` | `NumberRealEstateLoansOrLines` | P | Patrimonial/financeiro | idem | R05; agregados |
| `past_due_30_59`, `past_due_60_89`, `times_90_days_late` | `NumberOfTime...` | P | Histórico de atraso (dado negativo de crédito) | idem | Códigos 96/98 → NaN; só agregados |
| `target` | `SeriousDlqin2yrs` | P (derivado) | Inadimplência observada | processed (referência) e lotes `labeled` | Ausente nos lotes de produção sem rótulo |
| `score`, `prediction` | gerados em `score_batch` | P (inferido) | Avaliação automatizada do titular (art. 20) | `artifacts/scored/` | Não versionado; só `score_mean`/`score_psi` em métricas |
| `event_date`, `batch_id`, `run_id` | gerados | NP | Metadados operacionais | lotes, logs, métricas | Livres para logs |

**Dados sensíveis (art. 11):** o dataset **não contém** raça, religião, saúde, biometria, opinião
política etc., e o projeto **não deve incluí-los**. A análise de vieses (E4-04) por isso fica
limitada a faixa etária (`docs/evidence/bias_report.md`).

## 2. Pseudonimização ≠ anonimização

`customer_id` é um **pseudônimo** (art. 13, §4º): quebra o vínculo direto, mas a combinação
`age + monthly_income + dependents + histórico` pode permitir reidentificação em um cenário
real com bases auxiliares. Por isso **não afirmamos que os dados estão anonimizados**
(art. 12 exige que a reversão seja inviável com meios razoáveis). No protótipo o risco é
baixo porque a base é pública e já não traz identificadores; com dados reais seria necessário:
tabela de correspondência separada e cifrada, controle de acesso por papel e avaliação de
risco de reidentificação (k-anonimato ou equivalente) antes de qualquer compartilhamento.

## 3. Minimização por artefato

| Artefato | Contém valores de linha? | O que contém | Versionado no Git? |
|---|---|---|---|
| `data/raw/`, `data/processed/`, `data/production/`, `data/quarantine/` | **Sim** | dados completos | **Não** (`.gitignore`) |
| `artifacts/scored/` | Sim (+ score) | predições por cliente | Não |
| `artifacts/logs/pipeline.jsonl` | **Não** | contagens, durações, status, ids de execução | Não |
| `artifacts/validation/*.json` (`ValidationResult`) | **Não** | regras, contagens, `row_index` (posição, não valor) | Não |
| `artifacts/drift/*.html` (Evidently) | Não (distribuições agregadas) | histogramas, PSI/KS por feature | Não; 1–2 exemplos pequenos em `docs/evidence/` |
| `artifacts/metrics/run_history.jsonl`, `artifacts/alerts/alerts.jsonl` | **Não** | métricas agregadas por lote | Não; cópia leve via `make evidence` |
| `artifacts/dashboard/index.html` | **Não** | agregados | Não; cópia via `make evidence` |
| MLflow (`mlflow/`) | Não | métricas, tags, artefatos acima | Não |
| `models/baseline_v1.json` | Não | métricas de treino, threshold, hash | Sim |
| `models/baseline_v1.joblib` | Não diretamente (modelo treinado) | pesos | Não |

Garantia executável: `tests/unit/test_governance_logs.py::test_logs_do_not_contain_row_values`
e `sanitize_extra` em `observability/logging.py`, que descarta DataFrames/Series/objetos do
campo `extra`.

## 4. Retenção proposta (decisão de projeto, a validar com jurídico e negócio)

| Artefato | Prazo proposto | Justificativa | Descarte |
|---|---|---|---|
| Dados brutos (`data/raw/`) | enquanto durar a finalidade de treino/validação; no protótipo, apagar ao fim do desafio | art. 15, I (término do tratamento) e art. 16 | exclusão do arquivo local |
| Lotes de produção e quarentena | 90 dias | reprocessamento e investigação de incidentes de qualidade | exclusão automática por data |
| Predições (`artifacts/scored/`) | 5 anos em ambiente real (prazo prescricional típico de relações de consumo e contestação de decisão automatizada, art. 20) — **confirmar com jurídico**; no protótipo, apagar ao fim | direito de revisão e auditoria | exclusão ou agregação |
| Logs, métricas e alertas (sem valores de linha) | 12 meses | trilha de auditoria operacional (art. 37) | rotação de arquivos |
| Relatórios HTML de drift | 12 meses | evidência de monitoramento | rotação |
| Modelo e metadata | enquanto o modelo estiver em uso + 5 anos | explicar decisões passadas (art. 20) | arquivamento |

## 5. Controle de acesso proposto

| Papel | Acesso |
|---|---|
| Engenharia de dados (P1/P2) | `data/` completo, contrato, quarentena |
| Ciência de dados (P1/P3) | `data/processed/` e `artifacts/scored/` (pseudonimizados), modelo |
| Observabilidade/SRE (P4) | logs, métricas, alertas, MLflow, dashboard — **sem acesso a `data/`** |
| Negócio / auditoria / DPO | dashboard, relatórios agregados, documentação |

No repositório: `data/`, `artifacts/`, `mlflow/`, `mlruns/`, `*.joblib`, `.env` e `kaggle.json`
estão no `.gitignore`. Segredos nunca entram no Git. Com dados reais: armazenamento cifrado,
autenticação por papel e registro de acesso (art. 46 e 47).
