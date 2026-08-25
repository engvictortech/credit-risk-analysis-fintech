"""
data_generator.py
-----------------
Gera uma série temporal mensal sintética (2015-2024) de indicadores
macroeconômicos e taxa de inadimplência, simulando o cenário de uma
fintech de crédito. Os dados são sintéticos, mas construídos com
relações econômicas plausíveis (desemprego e juros elevam a
inadimplência com efeito defasado), incluindo um choque em 2020
(equivalente à pandemia) para tornar a série realista.

Este script existe para permitir que o projeto seja 100% reprodutível
sem depender de fontes de dados externas/privadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gerar_serie_macro(
    data_inicio: str = "2015-01-01",
    n_meses: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera série temporal mensal sintética de indicadores macro e inadimplência.

    Args:
        data_inicio: primeira data da série (formato ISO).
        n_meses: quantidade de meses a gerar.
        seed: semente do gerador aleatório, para reprodutibilidade.

    Returns:
        DataFrame com colunas: data, taxa_desemprego, taxa_juros,
        taxa_inadimplencia.
    """
    rng = np.random.default_rng(seed)
    datas = pd.date_range(start=data_inicio, periods=n_meses, freq="MS")

    t = np.arange(n_meses)

    # Desemprego: tendência + ciclo + choque pandemia (mês ~60 = 2020)
    desemprego = 11 + 1.5 * np.sin(2 * np.pi * t / 48) + 0.01 * t
    choque_pandemia = 6 * np.exp(-((t - 62) ** 2) / (2 * 6**2))
    desemprego = desemprego + choque_pandemia + rng.normal(0, 0.25, n_meses)
    desemprego = np.clip(desemprego, 6, 20)

    # Juros (Selic-like): ciclo de política monetária + choque 2021-2022
    juros = 8 + 3 * np.sin(2 * np.pi * t / 60 + 1) 
    choque_juros = 5 / (1 + np.exp(-(t - 78) / 4)) - 5 / (1 + np.exp(-(t - 100) / 4))
    juros = juros + choque_juros + rng.normal(0, 0.2, n_meses)
    juros = np.clip(juros, 2, 15)

    # Inadimplência: depende de desemprego e juros defasados (3 meses) + ruído
    lag = 3
    desemprego_lag = pd.Series(desemprego).shift(lag).bfill().to_numpy()
    juros_lag = pd.Series(juros).shift(lag).bfill().to_numpy()

    inadimplencia = (
        1.2
        + 0.35 * (desemprego_lag - desemprego_lag.min())
        + 0.15 * (juros_lag - juros_lag.min())
        + rng.normal(0, 0.3, n_meses)
    )
    inadimplencia = np.clip(inadimplencia, 0.5, 12)

    df = pd.DataFrame(
        {
            "data": datas,
            "taxa_desemprego": desemprego.round(2),
            "taxa_juros": juros.round(2),
            "taxa_inadimplencia": inadimplencia.round(2),
        }
    )
    return df


if __name__ == "__main__":
    df = gerar_serie_macro()
    df.to_csv("data/raw/serie_macro_inadimplencia.csv", index=False)
    print(f"Gerados {len(df)} registros mensais em data/raw/serie_macro_inadimplencia.csv")
    print(df.head())
