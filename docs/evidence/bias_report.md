# Relatório descritivo de desempenho e viés por faixa etária

Amostra avaliada: 11900 linhas do conjunto de teste de referência.

Atributos sensíveis como sexo e raça não estão disponíveis neste dataset. Faixa etária é uma análise limitada;
não demonstra ausência de viés nem atende, sozinha, a uma avaliação de equidade em produção.

Threshold registrado na metadata: 0.237631. Aprovação é definida como prediction = 0.

## Resultados por faixa

| Faixa | N | Inadimplência | Aprovação | ROC-AUC | PR-AUC | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 18-29 | 767 | 0.1017 | 0.8657 | 0.8314 | 0.4136 | 0.5256 | 0.0900 |
| 30-44 | 3337 | 0.0929 | 0.8825 | 0.8500 | 0.4276 | 0.5484 | 0.0733 |
| 45-59 | 4370 | 0.0721 | 0.9201 | 0.8473 | 0.3943 | 0.4984 | 0.0473 |
| 60+ | 3426 | 0.0368 | 0.9729 | 0.8767 | 0.2679 | 0.2778 | 0.0176 |

## Diferenças observadas

- Razão entre menor e maior taxa de aprovação: 0.8899
- Diferença absoluta nas taxas de aprovação: 0.1071
- Diferença de recall entre grupos: 0.2706

## Limitações e mitigação

Essas métricas são descritivas e dependem da amostra, do threshold e da definição operacional de aprovação.
Não são limiares legais nem provam discriminação ou ausência dela. Mudanças de política, seleção de aprovados e
rótulos atrasados podem distorcer a comparação. Em uso real, avaliar atributos permitidos com revisão jurídica,
análise por subgrupos, revisão humana de decisões contestadas e monitoramento periódico das diferenças.
