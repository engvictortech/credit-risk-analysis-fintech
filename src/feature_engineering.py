"""
feature_engineering.py
-----------------------
Cria as variáveis defasadas (lags) usadas como preditores e os dois
targets do projeto:

1. taxa_inadimplencia (contínuo)      -> usado na Regressão Linear
2. alta_inadimplencia (binário, 0/1)  -> usado na Regressão Logística e
                                          Random Forest

O target binário é definido como 1 quando a taxa de inadimplência do
mês está ACIMA da taxa observada no mesmo mês do ano anterior
(comparação year-over-year, YoY) — ou seja, "mês em que a carteira
piorou na comparação anual". Essa definição foi escolhida no lugar de
um corte por mediana móvel: em uma série com tendência de longo prazo
(como a gerada aqui, com um choque em 2020), um corte por mediana
móvel fica desbalanceado nos períodos mais recentes do teste, e o
classificador acaba "aprendendo" a tendência em vez de um padrão real
(um teste inicial com esse critério gerou recall perfeito e ROC-AUC
próximo de 0.55 — sinal de classificador degenerado). A comparação
YoY neutraliza a tendência e obriga o modelo a captar sinal de fato.

Isso mantém os dois modelos (regressão e classificação) operando sobre
a MESMA granularidade de dados (série macro mensal), evitando misturar
modelagem agregada com modelagem "por cliente" (que exigiria uma base
transacional diferente).
"""

from __future__ import annotations

import pandas as pd

LAGS = (1, 3, 6)


def criar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona lags das variáveis macro e o target binário de alta inadimplência.

    Args:
        df: DataFrame com colunas data, taxa_desemprego, taxa_juros,
            taxa_inadimplencia, ordenado cronologicamente.

    Returns:
        DataFrame com as colunas originais + lags + target binário,
        sem linhas com NaN gerado pelos lags/rolling.
    """
    df = df.copy()

    for lag in LAGS:
        df[f"taxa_desemprego_lag{lag}"] = df["taxa_desemprego"].shift(lag)
        df[f"taxa_juros_lag{lag}"] = df["taxa_juros"].shift(lag)

    # Target binário: comparação year-over-year (YoY), robusta a tendência
    df["taxa_inadimplencia_ano_anterior"] = df["taxa_inadimplencia"].shift(12)
    df["alta_inadimplencia"] = (
        df["taxa_inadimplencia"] > df["taxa_inadimplencia_ano_anterior"]
    ).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def colunas_preditoras() -> list[str]:
    """Retorna a lista de colunas preditoras (features) usada pelos modelos."""
    cols = ["taxa_desemprego", "taxa_juros"]
    for lag in LAGS:
        cols += [f"taxa_desemprego_lag{lag}", f"taxa_juros_lag{lag}"]
    return cols


if __name__ == "__main__":
    from data_loader import carregar_serie

    df = carregar_serie("data/raw/serie_macro_inadimplencia.csv")
    df_feat = criar_atributos(df)
    df_feat.to_csv("data/processed/serie_macro_features.csv", index=False)
    print(f"Dataset com features salvo: {df_feat.shape[0]} linhas, {df_feat.shape[1]} colunas")
    print(f"Distribuição do target binário:\n{df_feat['alta_inadimplencia'].value_counts()}")
