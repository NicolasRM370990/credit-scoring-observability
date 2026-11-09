# Dados

Os dados **não são versionados** (licença, privacidade e tamanho). Esta pasta só guarda este README.

## Origem
- Dataset: *Give Me Some Credit* (Kaggle). Link e termos: ver `docs/decisions/ADR-001-dataset.md`.
- Arquivo usado: `cs-training.csv`
- SHA-256 observado para `cs-training.csv`: `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac` (também em `config/data.yaml`)

**Termos de uso:** o arquivo é distribuído como dado de uma competição Kaggle. A equipe deve respeitar as regras exibidas na página da competição e os termos da plataforma; este repositório não atribui uma licença aberta ao dataset. Não redistribua o CSV.

## Como obter
**Opção 1: Kaggle CLI** (requer conta, `kaggle.json` configurado e aceite das regras da competição):
```bash
kaggle competitions download -c GiveMeSomeCredit -p data/raw
unzip data/raw/GiveMeSomeCredit.zip -d data/raw
```
**Opção 2: manual.** Baixe pelo site e salve em `data/raw/cs-training.csv`.

Depois rode:
```bash
make data
```
O comando confere o SHA-256 e gera `data/processed/full.parquet` com as colunas renomeadas para snake_case. A execução falha se o hash configurado não estiver preenchido ou não corresponder ao arquivo.

## Estrutura (criada pelos scripts)
```text
data/
├── raw/          # arquivo original do Kaggle
├── processed/    # parquets (full, reference_*, production_pool)
├── production/   # lotes simulados por cenário
└── quarantine/   # lotes rejeitados pelo contrato
```
