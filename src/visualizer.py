"""
visualizer.py
-------------
Geração dos gráficos do projeto, salvos em outputs/graficos/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend sem tela, para rodar em qualquer ambiente
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, RocCurveDisplay

sns.set_style("whitegrid")
PALETA = "#1f4e5f"


def plot_serie_historica(df: pd.DataFrame, caminho: str) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(df["data"], df["taxa_inadimplencia"], color="#c0392b", linewidth=2, label="Taxa de Inadimplência (%)")
    ax1.set_ylabel("Inadimplência (%)", color="#c0392b")
    ax1.set_xlabel("Período")

    ax2 = ax1.twinx()
    ax2.plot(df["data"], df["taxa_desemprego"], color="#2980b9", linestyle="--", label="Desemprego (%)")
    ax2.plot(df["data"], df["taxa_juros"], color="#27ae60", linestyle=":", label="Juros (%)")
    ax2.set_ylabel("Desemprego / Juros (%)")

    fig.suptitle("Evolução Histórica: Inadimplência vs. Indicadores Macro (2015-2024)", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_correlacao(df: pd.DataFrame, colunas: list[str], caminho: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[colunas].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Matriz de Correlação — Variáveis Macro e Inadimplência")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_real_vs_previsto(datas_teste, y_real, y_pred, caminho: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(datas_teste, y_real, marker="o", label="Real", color="#1f4e5f")
    ax.plot(datas_teste, y_pred, marker="x", linestyle="--", label="Previsto (Regressão Linear)", color="#c0392b")
    ax.set_title("Regressão Linear — Valores Reais vs. Previstos (conjunto de teste)")
    ax.set_ylabel("Taxa de Inadimplência (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_roc_curves(modelos: dict, X_teste, y_teste, caminho: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for nome, pipe in modelos.items():
        RocCurveDisplay.from_estimator(pipe, X_teste, y_teste, ax=ax, name=nome)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    ax.set_title("Curva ROC — Classificação de Alta Inadimplência")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_importancia_features(importancias: dict, caminho: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    itens = sorted(importancias.items(), key=lambda x: x[1])
    nomes = [i[0] for i in itens]
    valores = [i[1] for i in itens]
    ax.barh(nomes, valores, color=PALETA)
    ax.set_title("Random Forest — Importância das Variáveis")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
