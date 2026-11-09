# Arquitetura CreditWatch

```mermaid
flowchart LR
  RAW[Dataset Kaggle] --> PREP[download + prepare]
  PREP --> REF[(reference train / val / test)]
  PREP --> POOL[(production pool)]
  REF --> TRAIN[baseline HistGradientBoosting]
  TRAIN --> MODEL[(modelo + metadata)]
  POOL --> SIM[simulação de cenários]
  SIM --> ING[ingestão + contrato]
  ING --> SCORE[score_batch]
  MODEL --> SCORE
  REF --> DRIFT[PSI / KS / Evidently]
  SCORE --> DRIFT
  DRIFT --> OBS[logs + métricas + MLflow]
  OBS --> ALERT[alertas]
  OBS --> REPORT[relatórios]
  ING -. lote inválido .-> QUAR[(quarentena)]
```

O comando `make demo-smoke` valida os seis estágios e o formato de log sem depender dos módulos dos demais responsáveis. `make demo` conecta os componentes pelas assinaturas de `src/creditwatch/interfaces.py`; a CLI informa as interfaces de equipe que ainda não estiverem disponíveis.

Os splits são estratificados, gerados com seed fixa e separados por `customer_id`. `production_pool` não participa do treino. Dados brutos e processados permanecem fora do Git.
