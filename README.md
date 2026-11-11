# Avaliação multiprotocolo de métodos clássicos de aprendizado de máquina no diagnóstico de falhas do Tennessee Eastman Process

Código, manifestos e resultados do artigo:

> ROCHA, A. R. F.; BERNARDO, A. F. S.; SANTOS, B. E. D.; COELHO, C. J. *Desempenho preditivo e custo computacional de métodos clássicos de aprendizado de máquina no diagnóstico de falhas do Tennessee Eastman Process: uma avaliação multiprotocolo*. PUC Goiás, 2026.

Repositório: https://github.com/BrendonDourado/tep-avaliacao-multiprotocolo

## Conteúdo

| Caminho | O que é |
| --- | --- |
| `experimento_unificado.py` | Script do protocolo unificado (PU): preparação, busca de hiperparâmetros, treino final, estatística, piloto, figuras e ablações |
| `requirements.txt` | Dependências Python |
| `resultados/manifesto_divisao.json` | Execuções de treino e validação por classe (semente 42) e hash SHA-256 |
| `resultados/config_congelada.json` | Configurações congeladas pela validação antes do teste |
| `resultados/busca_hiperparametros.csv` | Grade completa avaliada na validação |
| `resultados/metricas_*.csv` | F1 macro, MCC e acurácias por modelo, semente e convenção de rótulo; métricas por classe |
| `resultados/bootstrap_*.csv`, `friedman.json`, `wilcoxon_holm.csv` | Inferência estatística por execução |
| `resultados/custos.csv` | Tempo de treino e inferência, tamanho do modelo e pico de memória |
| `resultados/voto_majoritario.csv`, `piloto_*.csv` | Avaliação por execução e efeito da unidade de divisão |
| `resultados/ablacao.csv` | Ablações (um fator por vez) |
| `resultados/ambiente.json` | Equipamento e versões das bibliotecas |
| `resultados/log.txt` | Log de execução com carimbos de tempo: busca, congelamento das configurações (9/11/2026, 21h40) e, só depois, a predição do teste |
| `artigo/` | PDF do artigo e figuras |

## Dados

Os dados não estão no repositório. Baixe os quatro arquivos `.RData` de Rieth et al. (2017) no Harvard Dataverse: https://doi.org/10.7910/DVN/6C3JR1

## Como reproduzir

```
pip install -r requirements.txt
python experimento_unificado.py --etapa tudo --dados <pasta com os .RData>
python experimento_unificado.py --etapa ablacao --dados <pasta com os .RData>
```

Teste rápido (minutos): `python experimento_unificado.py --etapa tudo --rapido --saida teste_rapido`.

Ambiente usado: AMD Ryzen 7 5700X (8 núcleos), 32 GB, Windows 11, Python 3.14.5, scikit-learn 1.9.1, XGBoost 3.4.1, 8 threads para todos os modelos. O experimento completo levou cerca de 12 horas.

## Protocolos P1, P2 e P3

Os três protocolos independentes foram executados separadamente por cada autor; suas configurações estão descritas nas Tabelas 3 e 4 do artigo.
