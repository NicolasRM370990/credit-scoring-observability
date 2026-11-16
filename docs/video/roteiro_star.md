# Roteiro STAR — CreditWatch (até 5:00, alvo 4:30)

Narração para gravar sobre a apresentação `docs/video/creditwatch_apresentacao.pptx`.
Uma pessoa narra. Se forem duas, a primeira fica com Situação, Tarefa, arquitetura, modelo e contrato. A segunda fica com simulação, drift, governança e fechamento.

Fale olhando para a câmera ou para a tela, em tom de quem explica para a diretoria. Não leia os RMs número a número. Não afirme que o MLflow ou o motor de alertas já fecham o ciclo: estão especificados e ainda não implementados.

## Como gravar

- Apresentação em tela cheia, fonte do terminal em 18 ou mais, se abrir o terminal.
- Plano A: só os slides, na ordem abaixo. Cabe em 4:30.
- Plano B, se sobrar 20 segundos no ensaio: no bloco do contrato, mostrar um lote com `age = 17` bloqueado por `R01_AGE_RANGE`. Não rodar `pytest` ao vivo. A suíte leva cerca de 45 segundos.
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

Este é o CreditWatch, da fase 4. A equipe é Alexis Ferreira Duarte, George Ricardo dos Santos, Leandro de Souza Pignataro e Nicolas de Araújo Gomes.

### 0:18 — Slide 2 — Situação

Um modelo de crédito pode piorar sem quebrar o sistema. A renda muda com a inflação, o uso do limite muda, e o score continua saindo todo dia. A diretoria só percebe quando a inadimplência já subiu.

### 0:40 — Slide 3 — Tarefa

A tarefa foi sustentar esse modelo em três pontos. Barrar dado inválido antes do score. Detectar drift com variáveis de controle. E registrar a execução sem gravar valor de cliente no log.

### 1:02 — Slide 4 — Ação, arquitetura

Separamos quem treina o modelo de quem simula a produção. O lote entra pelo contrato. Se o contrato bloqueia, o scoring não roda. Se passa, comparamos o lote pontuado com a referência.

### 1:24 — Slide 5 — Ação, modelo

O baseline é um HistGradientBoosting. No teste, o ROC-AUC foi 0,86, contra 0,50 de um classificador ingênuo. *(corte)* Só 7% das linhas são inadimplentes, então também olhamos PR-AUC e KS. O corte de 0,24 foi o que maximizou o F1 na validação.

### 1:48 — Slide 6 — Ação, contrato

Cinco regras bloqueiam o lote inteiro: idade, renda, duplicata, schema e número negativo. Volume baixo e nulos demais só avisam. A resposta guarda o índice da linha, não o valor. Um menor de idade não chega a ser pontuado.

### 2:16 — Slide 7 — Ação, simulação

São três cenários, com a mesma semente. Um não muda nada. O moderado mexe em renda e utilização. O severo acrescenta debt ratio, com intensidade crescente. Idade e dependentes ficam parados, de propósito.

### 2:38 — Slide 8 — Resultado, drift

No PSI, a utilização de crédito passou muito do limite crítico de 0,25. Idade, dependentes e atrasos ficaram perto de zero. O Evidently usa outra distância e marcou renda e score: 2 de 11 colunas, 18%. Como o limiar dele é 50%, o dataset inteiro não foi declarado em drift. Os dois métodos não apontam a mesma coluna. Por isso os dois ficam no monitoramento.

### 3:22 — Slide 9 — Resultado, governança

O CSV não entra no Git. O identificador do cliente é um código gerado. Entre 18 e 29 anos, a aprovação foi 87%. Acima de 60, 97%. A diferença de recall entre as faixas é 0,27. Isso descreve o recorte. Não prova discriminação, e a base não tem sexo nem raça.

### 3:48 — Slide 10 — Resultado, limites

Trezentos e trinta e oito testes passaram. Drift de dado, porém, não é queda de desempenho: sem o rótulo atrasado, concept drift fica fora da conta. Alertas e MLflow estão especificados. Ainda não fecham o ciclo.

### 4:12 — Slide 11 — Fechamento

O que já protege a decisão é a porta do contrato e a comparação com a referência. O próximo passo é transformar a discordância entre PSI e Evidently em alerta do lote.

## O que não dizer

- Não dizer que o MLflow já lista um run por lote.
- Não dizer que o lote inválido já cai sozinho na quarentena. O contrato bloqueia; a ingestão com quarentena ainda não está no código.
- Não dizer que o Evidently e o PSI apontaram as mesmas variáveis. O gráfico de PSI destaca `revolving_utilization`. O HTML do Evidently marca `monthly_income` e `score`.
- Não tratar a diferença por idade como prova de viés ilegal.

## Evidências usadas

- `models/baseline_v1.json`: ROC-AUC de teste 0,860, PR-AUC 0,386, KS 0,565, threshold 0,238, dummy 0,50.
- `docs/evidence/bias_report.md`: aprovação 86,6% (18–29) e 97,3% (60+); diferença de recall 0,271.
- `docs/evidence/e2-06-psi-evolution.png` e `docs/evidence/e2-06-drift-report.png`.
- Suíte local de 16/11/2026: 338 aprovados, 1 pulado por falta do CSV bruto.
