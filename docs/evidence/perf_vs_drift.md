# Desempenho rotulado × drift

As métricas de desempenho exigem rótulos observados. `drift_detected` vem do resultado do detector; a comparação não interpreta drift como causa de degradação.

| Cenário | Lote | Drift detectado | Drift share | Score PSI | Score KS | Score médio ref. | Score médio atual | Taxa positiva | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | reference | não | n/a | n/a | n/a | n/a | n/a | 0.0697 | 0.8601 | 0.3863 | 0.0515 |
| batch_01 | batch_01 | não | 0.0000 | 0.0038 | 0.0233 | 0.0697 | 0.0716 | 0.5593 | 0.3280 | 0.4411 | 0.5305 |
| batch_02 | batch_02 | não | 0.0000 | 0.0022 | 0.0132 | 0.0697 | 0.0692 | 0.6130 | 0.3187 | 0.4893 | 0.5808 |
| batch_03 | batch_03 | não | 0.0000 | 0.0023 | 0.0190 | 0.0697 | 0.0708 | 0.6310 | 0.3255 | 0.5094 | 0.5966 |
| batch_04 | batch_04 | não | 0.0000 | 0.0023 | 0.0132 | 0.0697 | 0.0686 | 0.6877 | 0.3248 | 0.5691 | 0.6432 |
