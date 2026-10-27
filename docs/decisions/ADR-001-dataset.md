# ADR-001: Escolha do dataset

- **Status:** proposto (aprovar no PR)
- **Data:** 2026-10-26
- **Responsável:** P1 (Dados e Modelo)

## Contexto
O Tech Challenge Fase 4 exige um dataset financeiro de classificação binária com **no mínimo 5.000 amostras**, adequado a risco de crédito. O PDF sugere German Credit, Home Credit Default Risk e Lending Club.

## Decisão
Usar o **Give Me Some Credit** (Kaggle), arquivo `cs-training.csv`.

## Dados (medidos no EDA, `notebooks/eda.ipynb`)

| Item | Valor |
|---|---|
| Linhas | 150.000 |
| Colunas (sem o índice `Unnamed: 0`) | 11 (10 features + target) |
| Target | `SeriousDlqin2yrs` (1 = atraso grave de 90 dias ou mais em 2 anos; 0 = não) |
| Taxa do target | ~6,7% de positivos (dataset desbalanceado) |
| `MonthlyIncome` nulo | 29.731 linhas (19,8%) |
| `NumberOfDependents` nulo | 3.924 linhas (2,6%) |
| Idade | mínimo 0 (inválido), máximo 109 |
| Códigos especiais | valor 98 (e possivelmente 96) nas colunas de atraso |
| Valores extremos | utilização máx. 50.708; `DebtRatio` máx. 329.664 |
| Idade < 18 | <!-- PREENCHER: célula 3 do EDA --> |
| Percentis p99 / p99,9 de renda | <!-- PREENCHER: célula 4 --> |
| Linhas após limpeza (idade >= 18 e renda não nula) | <!-- PREENCHER: célula 7 --> |
| Taxa do target após limpeza | <!-- PREENCHER: célula 7 --> |
| Taxa do target com renda nula × com renda | <!-- PREENCHER: célula 5 --> |

## Licença e origem
- Fonte: página da competição *Give Me Some Credit* no Kaggle
- O dataset **não é versionado** no repositório. O `data/README.md` explica como baixar e traz o SHA-256.

## Consequências
- Renda e idade existem nativamente, então as regras de exemplo do PDF (idade, renda) se aplicam direto.
- A **renda nula (19,8%)** será removida na preparação, para a referência passar no contrato. Isso reduz a amostra e pode mudar o perfil dos dados. Registrar no README como limitação.
- Não há atributos sensíveis explícitos (sexo, raça). A **análise de vieses** (E4-04) fica limitada a **faixa etária** e variáveis correlatas. Registrar essa limitação.
- Pool de produção estimado em ~20% das linhas limpas, o que define o `batch_size` dos cenários (3.000).