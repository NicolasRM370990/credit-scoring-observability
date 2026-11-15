# ADR-002: Simulação de concept drift com rótulos sintéticos

- **Status:** decisão técnica proposta para o protótipo; hipótese experimental, não representa uma relação causal comprovada.
- **Data:** 2026-11-15
- **Responsável:** P3 (Drift e Simulação)

## Contexto

O Tech Challenge Fase 4 exige demonstrar que a distribuição das covariáveis pode permanecer estável enquanto a relação entre as variáveis de entrada (`X`) e o resultado (`y`) se altera.

O cenário `concept_drift` (E2-07a) deve utilizar covariáveis provenientes do `production_pool`, sem aplicar transformações de distribuição, e gerar rótulos sintéticos com uma relação alternativa entre as features e o target.

Essa simulação permite investigar uma limitação importante do monitoramento: métricas de data drift podem não sinalizar uma alteração na relação entre as covariáveis e o resultado.

## Decisão

Implementar o cenário `concept_drift` utilizando rótulos sintéticos gerados por uma função logística.

As covariáveis são amostradas do `production_pool` e preservadas. Somente a coluna `target` é substituída pelos rótulos sintéticos.

A geração utiliza as seguintes features e coeficientes:

| Feature | Coeficiente |
|---|---:|
| `revolving_utilization` | -0,8 |
| `age` | +0,5 |
| `past_due_30_59` | -0,7 |
| `debt_ratio` | +0,8 |
| `monthly_income` | +0,5 |
| `times_90_days_late` | -1,0 |

As features são padronizadas utilizando médias e desvios-padrão calculados uma única vez sobre o pool amostrado. Os valores padronizados são limitados ao intervalo [-3, 3].

A probabilidade de cada registro receber o rótulo positivo é calculada por:

\[
P(y=1 \mid X) = \frac{1}{1+e^{-z}}
\]

em que:

\[
z = \sum_{j=1}^{n} \beta_j \tilde{x}_j + b
\]

Onde:

- \(\beta_j\) representa o coeficiente sintético da feature.
- \(\tilde{x}_j\) representa o valor da feature padronizado e limitado ao intervalo [-3, 3].
- \(b\) representa o intercepto da função logística.

Os coeficientes representam a relação sintética assumida entre as features e o target.

O intercepto é controlado pelo parâmetro `logit_offset`, configurado inicialmente como `0.8` e aplicado progressivamente conforme a intensidade do lote.

A intensidade é calculada por:

\[
intensidade_i = \frac{i}{N}
\]

O intercepto efetivo de cada lote é:

\[
b_i = logit\_offset \times intensidade_i
\]

Para quatro lotes, a progressão é:

| Lote | Intensidade | Intercepto efetivo |
|---|---:|---:|
| 1 | 0,25 | 0,20 |
| 2 | 0,50 | 0,40 |
| 3 | 0,75 | 0,60 |
| 4 | 1,00 | 0,80 |

Os coeficientes das features permanecem fixos durante os lotes. A progressão ocorre somente no intercepto.

Os rótulos são amostrados por uma distribuição Bernoulli, utilizando seeds determinísticas para garantir reprodutibilidade.

## Tratamento de valores ausentes

Foram identificados 23 registros com valores ausentes simultaneamente nas variáveis `past_due_30_59` e `times_90_days_late` no `production_pool` de 23.801 registros.

Para permitir o cálculo das probabilidades sintéticas:

- As medianas das features são calculadas uma única vez sobre o pool amostrado.
- Os valores ausentes são substituídos pelas medianas somente em uma cópia interna utilizada no cálculo dos rótulos.
- O DataFrame original e as covariáveis dos lotes permanecem inalterados.
- Os valores ausentes continuam presentes nos arquivos Parquet gerados.

Essa imputação é exclusivamente operacional para a geração dos rótulos sintéticos e não representa uma alteração do processo de preparação dos dados.

## Consequências

- O cenário mantém as covariáveis do pool, sem transformações de distribuição.
- A coluna `target` passa a representar uma relação sintética entre `X` e `y`.
- A geração é reprodutível por meio das seeds configuradas.
- O cenário permite avaliar se o detector de data drift sinaliza pouco ou não sinaliza a alteração simulada.
- A avaliação quantitativa da degradação do desempenho será realizada separadamente na atividade E2-07b.

## Limitações e premissas

- Os coeficientes foram definidos como hipótese experimental e não foram estimados a partir de uma relação causal real.
- Os rótulos sintéticos não devem ser interpretados como inadimplência observada ou real.
- A simulação altera a regra de geração dos rótulos em relação ao target original, mas não comprova que a mesma relação ocorreria em produção.
- O `logit_offset` altera o intercepto e, consequentemente, pode modificar a prevalência dos rótulos. Ele não altera, isoladamente, os coeficientes das features.
- A avaliação da queda de desempenho do modelo não faz parte desta decisão e permanece no escopo do E2-07b.