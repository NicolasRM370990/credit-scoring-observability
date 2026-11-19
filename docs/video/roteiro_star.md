# Roteiro STAR — CreditWatch (até 5:00, alvo 4:30)

Narração para gravar sobre a apresentação `docs/video/creditwatch_apresentacao.pptx`.
Uma pessoa narra. Se forem duas, a primeira fica com Situação, Tarefa, arquitetura, modelo e contrato. A segunda fica com simulação, drift, governança e fechamento.

Fale olhando para a câmera ou para a tela, em tom de quem explica para a diretoria. Não leia os RMs número a número. Alertas, MLflow e quarentena já estão no código. Não diga que o PSI e o Evidently apontaram as mesmas colunas.

## Como gravar

- Apresentação em tela cheia, fonte do terminal em 18 ou mais, se abrir o terminal.
- Plano A: só os slides, na ordem abaixo. Cabe em 4:30.
- Plano B, se sobrar 20 segundos no ensaio: no bloco do contrato, mostrar um lote com `age = 17` bloqueado por `R01_AGE_RANGE` e movido para `data/quarantine`. Não rodar `pytest` ao vivo. A suíte leva cerca de 3 minutos e 30 segundos.
- Se passar de 4:40 no ensaio, cortar as frases marcadas com *(corte)*.
- Conferir no take final: o vídeo tem no máximo 5:00, o áudio não estoura e o gráfico de PSI continua legível.

## Mapa

| Tempo | Bloco STAR | Slide | O que está na tela |
|---|---|---|---|
| 0:00–0:18 | Abertura | 1 | Capa e equipe |
| 0:18–0:40 | Situação | 2 | Risco silencioso |
| 0:40–1:02 | Tarefa | 3 | Três exigências |
| 1:02–1:24 | Ação | 4 | Arquitetura |
| 1:24–1:48 | Ação | 5 | Baseline |
| 1:48–2:16 | Ação | 6 | Contrato |
| 2:16–2:38 | Ação | 7 | Três cenários |
| 2:38–3:22 | Resultado | 8 | Gráfico de PSI |
| 3:22–3:48 | Resultado | 9 | Governança e viés |
| 3:48–4:12 | Resultado | 10 | Testes e limites |
| 4:12–4:30 | Resultado | 11 | Fechamento |

## Narração

### 0:00 — Slide 1 — Abertura

A seguir vamos apresentar o projeto do Tech Challenge fase 4 - CreditWatch - observabilidade continua de um modelo de credit scoring. 

Esta apresentação segue o modelo STAR - Situação, Tarefa, Ação e Resultado. A equipe é Alexis, George, Leandro e Nicolas.

### 0:18 — Slide 2 — Situação

Um modelo de crédito pode piorar sem quebrar o sistema. A renda muda com a inflação, o uso do limite muda, e o score continua saindo todo dia. A diretoria só percebe quando a inadimplência já subiu.

### 0:40 — Slide 3 — Tarefa

A tarefa foi sustentar esse modelo em três pontos. Barrar dado inválido antes do score. Detectar drift com variáveis de controle. E registrar a execução sem gravar valor de cliente no log.

### 1:02 — Slide 4 — Ação, arquitetura

Separamos quem treina o modelo de quem simula a produção. O lote entra pelo contrato. Se o contrato bloqueia, o scoring não roda. Se passa, comparamos o lote pontuado com a referência. Cada estágio grava um log sem valor de linha. Alertas, MLflow e um dashboard HTML fecham a execução.

### 1:24 — Slide 5 — Ação, modelo

O baseline é um HistGradientBoosting. No teste, o ROC-AUC foi 0,86, contra 0,50 de um classificador ingênuo. *(corte)* Só 7% das linhas são inadimplentes, então também olhamos PR-AUC e KS. O corte de 0,24 foi o que maximizou o F1 na validação.

### 1:48 — Slide 6 — Ação, contrato

Cinco regras bloqueiam o lote inteiro: idade, renda, duplicata, schema e número negativo. Volume baixo e nulos demais só avisam. A resposta guarda o índice da linha, não o valor. Um menor de idade não chega a ser pontuado.

### 2:16 — Slide 7 — Ação, simulação

São três cenários de dado, com a mesma semente. Um não muda nada. O moderado mexe em renda e utilização. O severo acrescenta debt ratio, com intensidade crescente. Idade e dependentes ficam parados, de propósito. O quarto cenário muda só o rótulo.

### 2:38 — Slide 8 — Resultado, drift

O gráfico usa os limiares calibrados: aviso 0,016 e crítico 0,017. No cenário severo, renda e debt ratio sobem desde o primeiro lote. A utilização dispara no último, acima de 1,2. Idade e dependentes ficam no zero. O Evidently, com outra distância, não aponta as mesmas colunas.

### 3:22 — Slide 9 — Resultado, governança

O CSV, o parquet e o modelo não entram no Git. O identificador é um pseudônimo, não uma anonimização. O log descarta valor de linha. Entre 18 e 29 anos, a aprovação foi 87%. Acima de 60, 97%. A diferença de recall entre as faixas é 0,27. Isso descreve o recorte. Não prova discriminação, e a base não tem sexo nem raça.

### 3:48 — Slide 10 — Resultado, limites

Quatrocentos e trinta e seis testes passaram nesta pasta. Cinco de ponta a ponta não iniciaram: faltam o parquet processado e o arquivo do modelo. No concept drift sintético, o detector de dado não acende e o ROC-AUC cai de 0,86 para cerca de 0,32.

### 4:12 — Slide 11 — Fechamento

O contrato manda o lote inválido para a quarentena. O alerta de PSI usa o limite calibrado. O log, o dashboard HTML e o MLflow mostram a execução sem valor de cliente. PSI e Evidently continuam lado a lado, porque não contam a mesma história.

## O que não dizer

- Não dizer que o Evidently e o PSI apontaram as mesmas variáveis. O gráfico de PSI destaca `revolving_utilization`. O HTML do Evidently marca `monthly_income` e `score`.
- Não dizer que a utilização é a única variável acima do limite. No gráfico atual, renda e debt ratio sobem antes; a utilização dispara no último lote.
- Não tratar a diferença por idade como prova de viés ilegal.
- Não dizer que o `customer_id` anonimiza o cliente. Ele é um pseudônimo.
- Não dizer que os cinco testes de ponta a ponta falharam por defeito de código. Nesta pasta eles não iniciaram porque o parquet e o `.joblib` não estão versionados.

## Evidências usadas

- `models/baseline_v1.json`: ROC-AUC de teste 0,860, PR-AUC 0,386, KS 0,565, threshold 0,238, dummy 0,50.
- `docs/evidence/bias_report.md`: aprovação 86,6% (18–29) e 97,3% (60+); diferença de recall 0,271.
- `docs/evidence/e2-06-psi-evolution.png` e `docs/evidence/e2-06-drift-report.png`.
- `config/drift_thresholds.yaml` (17/11/2026): PSI de aviso 0,0156 e crítico 0,0172; KS 0,0427.
- `docs/evidence/perf_vs_drift.md`: no concept drift sintético, drift de dado não detectado e ROC-AUC perto de 0,32.
- `docs/governance/data_inventory.md`: inventário LGPD; `customer_id` é pseudônimo.
- `docs/evidence/e2-14-dashboard.png`: dashboard HTML com execuções, bloqueios, PSI e alertas.
- Suíte local de 19/11/2026: 436 aprovados, 1 pulado, 5 erros de preparação por falta de `data/processed/*.parquet` e `models/baseline_v1.joblib`.
