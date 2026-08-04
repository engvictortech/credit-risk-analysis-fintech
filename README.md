# 📊 Análise e Previsão de Inadimplência
Projeto de Data Science aplicado a Risco de Crédito em Fintech
## 🎯 Visão Executiva

Este projeto simula um cenário real de uma fintech de crédito que precisa antecipar movimentos de deterioração da carteira.

O objetivo é transformar variáveis macroeconômicas em insights estratégicos acionáveis, permitindo decisões como:

Ajuste de políticas de concessão

Revisão de limites de crédito

Reprecificação de taxas

Reforço de provisionamento (PDD)

A proposta vai além da análise técnica — o foco está em impacto direto no negócio.

## 🧠 Problema de Negócio

Instituições financeiras operam sob risco constante de inadimplência.

Movimentos macroeconômicos, como aumento do desemprego ou da taxa de juros, impactam diretamente a qualidade da carteira.

Pergunta central:

É possível antecipar movimentos de alta na inadimplência utilizando variáveis macroeconômicas?

📁 data/
 
 ├── raw/
 
 └── processed/


📁 notebooks/
 
 ├── 01_coleta_dados.ipynb

 ├── 02_tratamento_dados.ipynb
 
 ├── 03_analise_exploratoria.ipynb

 ├── 04_modelagem_macro.ipynb


📁 models/
 
 └── modelo_macro_pipeline.pkl


📁 outputs/
 
 ├── graficos/
 
 └── metricas/

README.md

requirements.txt

# 🔎 Abordagem Analítica

## 1️⃣ Coleta de Dados

Taxa de inadimplência

Taxa de juros

Taxa de desemprego

Estruturação em série temporal

## 2️⃣ Tratamento e Engenharia de Dados

Padronização temporal

Tratamento de valores ausentes

Organização cronológica

Criação de variáveis defasadas (lags)

Aplicação de boas práticas para evitar vazamento de dados

## 3️⃣ Análise Exploratória (EDA)

Principais análises realizadas:

Evolução histórica da inadimplência

Correlação entre desemprego e inadimplência

Impacto da taxa de juros

Identificação de tendências e ciclos econômicos

📌 Insight Estratégico:
O desemprego apresenta forte relação positiva com a inadimplência, enquanto juros indicam impacto com efeito defasado — reforçando a importância do monitoramento macroeconômico na gestão de risco.

## 4️⃣ Modelagem Preditiva

Modelo econométrico baseado em Regressão Linear.

Boas práticas aplicadas:

Separação treino/teste cronológica (80/20)

Avaliação com MAE, RMSE e R²

Análise de multicolinearidade (VIF)

Inclusão de variáveis defasadas

Comparação visual entre valores reais e previstos

## 🎯 Objetivo: antecipar movimentos de deterioração da carteira.

## ⚙️ Pipeline de Modelagem

O modelo foi estruturado em formato de Pipeline do Scikit-Learn, permitindo:

Reprodutibilidade

Organização do fluxo de pré-processamento + modelagem

Facilidade de deploy

Exportação para ambiente produtivo

Arquivo salvo em:
models/modelo_macro_pipeline.pkl

## 📊 Resultados Relevantes

✔ Captura consistente da tendência de crescimento da inadimplência
✔ Evidência de poder explicativo das variáveis macroeconômicas
✔ Aplicabilidade direta em contexto de fintech

O projeto demonstra como transformar variáveis econômicas em sinal preditivo para gestão de risco.

## 🛠 Stack Tecnológica

Python

Pandas

NumPy

Matplotlib

Scikit-Learn

## 💼 Aplicação em Ambiente Real

Em uma fintech ou banco digital, este modelo pode apoiar:

Ajuste de score de crédito

Segmentação por risco

Revisão de política de concessão

Definição de limites

Estratégia de precificação

## 🚀 Roadmap Técnico

Implementação de modelo de classificação (inadimplente vs adimplente)

Regressão Logística

Random Forest

Avaliação com ROC-AUC e Recall

Backtesting temporal

Construção de dashboard executivo

## 👤 Autor

* Victor Hugo Miranda Crispim

* Bacharel em Análise de Dados

* Experiência em Crédito B2B e B2C

* Foco em análise de risco e tomada de decisão baseada em dados
