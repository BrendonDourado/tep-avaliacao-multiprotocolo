# Material suplementar S1 – Os treze trabalhos da Tabela 1

Relação dos trabalhos analisados na introdução do artigo, com a etapa de busca em que cada um foi encontrado e a avaliação dos critérios da Tabela 1. A avaliação foi feita por busca literal no texto completo de cada trabalho; a auditoria original, com os trechos citados, está no repositório do protocolo P3: https://github.com/ricardoarfr/MEI0028-modelagem-e-simulacao (pasta `revisao_bibliografica`).

## Estratégia de busca

- **Etapa 1 – trabalhos que usam a mesma base.** Busca no OpenAlex, pelo texto completo, de marcas próprias da base de Rieth et al. (2017): o código de registro `6C3JR1` e os termos `simulationRun`, `faultNumber` e `TEP_Faulty_Training`. Foram encontrados 101 trabalhos que citam a base e 31 que citam o artigo que a descreve; destes, cinco aplicam métodos clássicos de aprendizado de máquina à base.
- **Etapa 2 – amostra de trabalhos recentes.** Busca no Crossref, no OpenAlex e no Semantic Scholar por trabalhos de 2021 a 2026 que combinassem o nome do processo com os métodos avaliados (árvore de decisão, Random Forest, gradient boosting, XGBoost, SVM, regressão logística e classificação multiclasse de falhas). Dos quinze localizados, oito foram lidos integralmente e incluídos.
- Foram considerados apenas trabalhos com texto completo acessível. Os DOI foram conferidos no Crossref.
- A etapa 2 é uma amostra, e não uma revisão sistemática.

## Trabalhos e critérios

Legenda: **S** = atende ao critério; **N** = não atende; **L** = limítrofe.

| # | Trabalho | Veículo | DOI | Etapa | Divisão por execução declarada | Rótulo pré-falha do teste declarado | MCC | Custo para todas as famílias (inferência e memória, mesmo equipamento) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Peter et al. (2024) | ML4CCE Workshop, ECML PKDD | sem DOI (anais de workshop) | 1 | N | N | N | N |
| 2 | Sokolov, Pyatnitsky e Alabugin (2019) | FME Transactions, 47(4) | 10.5937/fmet1904782s | 1 | N | N | N | N |
| 3 | Çancıoğlu, Şahin e İşler (2021) | European Journal of Science and Technology | 10.31590/ejosat.952761 | 1 | N (divisão aleatória) | N | N | N |
| 4 | Miraliakbar e Jiang (2024) | Systems and Control Transactions, 3 | 10.69997/sct.184473 | 1 | N (10-fold sobre amostras) | N | N | N |
| 5 | Khan et al. (2026) | Scientific Reports, 16, art. 17488 | 10.1038/s41598-026-48227-6 | 1 | S | L | N | N (só o modelo proposto) |
| 6 | Koçak et al. (2026) | Processes, 14(16), art. 2569 | 10.3390/pr14162569 | 2 | S | S | N | N (só o XGBoost) |
| 7 | Lyu et al. (2026) | ChemRxiv (preprint, não revisado por pares) | 10.26434/chemrxiv.10001628/v1 | 2 | S | S | N | N (tempo de treino de todos, sem equipamento declarado) |
| 8 | Avanashilingam e Pandiyath Velayudhan (2026) | PHM Society Asia-Pacific Conference, 5 | 10.36001/phmap.2025.v5i1.4633 | 2 | S | N | N | N |
| 9 | Li et al. (2026) | Scientific Reports, 16, art. 19405 | 10.1038/s41598-026-49874-5 | 2 | N | S | N | N |
| 10 | Zhang et al. (2021) | Sensors, 21(20), art. 6715 | 10.3390/s21206715 | 2 | N | N | N | N |
| 11 | Hu et al. (2022) | Energies, 15(9), art. 3198 | 10.3390/en15093198 | 2 | N | N | N | N |
| 12 | Xu et al. (2022) | Applied Sciences, 12(17), art. 8868 | 10.3390/app12178868 | 2 | N | N | N | N |
| 13 | Márquez-Vera et al. (2026) | Algorithms, 19(6), art. 485 | 10.3390/a19060485 | 2 | N | N | N | N |

**Totais (Tabela 1 do artigo):** divisão por execução 4/13; rótulo pré-falha declarado 3/13 (4/13 se o caso limítrofe de Khan et al. for aceito); MCC 0/13; custo de inferência e memória para todas as famílias no mesmo equipamento 0/13 (três trabalhos reportam custo parcial: Lyu, Koçak e Khan).

**Critério do rótulo pré-falha:** conta apenas quando o texto declara o rótulo atribuído ou a exclusão das amostras anteriores à falha no teste. Descrever que "as primeiras 160 amostras são normais" não conta.
