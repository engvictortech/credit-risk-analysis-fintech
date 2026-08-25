"""
data_loader.py
--------------
Carregamento, validação e tratamento inicial da série temporal macro.
"""

from __future__ import annotations

import pandas as pd


def carregar_serie(caminho_csv: str) -> pd.DataFrame:
    """Carrega a série temporal a partir de um CSV e valida a estrutura.

    Args:
        caminho_csv: caminho do arquivo CSV de origem.

    Returns:
        DataFrame ordenado cronologicamente, com a coluna 'data' como
        datetime e índice temporal.

    Raises:
        ValueError: se colunas obrigatórias estiverem ausentes.
    """
    df = pd.read_csv(caminho_csv, parse_dates=["data"])

    colunas_esperadas = {"data", "taxa_desemprego", "taxa_juros", "taxa_inadimplencia"}
    faltantes = colunas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {faltantes}")

    df = df.sort_values("data").reset_index(drop=True)

    # Tratamento de valores ausentes: interpolação temporal (não usar média
    # global, que quebraria a estrutura de série temporal)
    colunas_numericas = ["taxa_desemprego", "taxa_juros", "taxa_inadimplencia"]
    n_nulos_antes = df[colunas_numericas].isna().sum().sum()
    df[colunas_numericas] = df[colunas_numericas].interpolate(method="linear")
    if n_nulos_antes > 0:
        print(f"[data_loader] {n_nulos_antes} valores ausentes tratados por interpolação linear.")

    return df


if __name__ == "__main__":
    df = carregar_serie("data/raw/serie_macro_inadimplencia.csv")
    print(df.info())
    print(df.describe())
