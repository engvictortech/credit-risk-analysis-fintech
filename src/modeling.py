"""
modeling.py
-----------
Treina e avalia dois modelos complementares sobre a mesma base macro
mensal, com split CRONOLÓGICO (80/20) — nunca aleatório, para não
vazar informação futura para o treino:

1. Regressão Linear      -> prevê a taxa de inadimplência (contínua)
2. Regressão Logística e
   Random Forest         -> classificam se o mês será de
                             "alta inadimplência" (binário)

Métricas de regressão: MAE, RMSE, R²
Métricas de classificação: Acurácia, Precisão, Recall, ROC-AUC
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import colunas_preditoras


@dataclass
class ResultadoRegressao:
    mae: float
    rmse: float
    r2: float
    coeficientes: dict


@dataclass
class ResultadoClassificacao:
    modelo: str
    acuracia: float
    precisao: float
    recall: float
    roc_auc: float
    matriz_confusao: list


def split_cronologico(df: pd.DataFrame, frac_treino: float = 0.8):
    """Divide o DataFrame em treino/teste respeitando a ordem cronológica."""
    corte = int(len(df) * frac_treino)
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


def treinar_regressao(df_treino: pd.DataFrame, df_teste: pd.DataFrame) -> tuple[Pipeline, ResultadoRegressao, np.ndarray]:
    """Treina Regressão Linear para prever taxa_inadimplencia (contínua)."""
    features = colunas_preditoras()
    X_treino, y_treino = df_treino[features], df_treino["taxa_inadimplencia"]
    X_teste, y_teste = df_teste[features], df_teste["taxa_inadimplencia"]

    pipe = Pipeline([("scaler", StandardScaler()), ("modelo", LinearRegression())])
    pipe.fit(X_treino, y_treino)
    y_pred = pipe.predict(X_teste)

    mae = mean_absolute_error(y_teste, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_teste, y_pred)))
    r2 = r2_score(y_teste, y_pred)

    coefs = dict(zip(features, pipe.named_steps["modelo"].coef_.round(4)))
    resultado = ResultadoRegressao(mae=round(mae, 4), rmse=round(rmse, 4), r2=round(r2, 4), coeficientes={k: float(v) for k, v in coefs.items()})
    return pipe, resultado, y_pred


def treinar_classificacao(df_treino: pd.DataFrame, df_teste: pd.DataFrame):
    """Treina Regressão Logística e Random Forest para o target binário."""
    features = colunas_preditoras()
    X_treino, y_treino = df_treino[features], df_treino["alta_inadimplencia"]
    X_teste, y_teste = df_teste[features], df_teste["alta_inadimplencia"]

    resultados = []
    modelos_treinados = {}

    # Regressão Logística
    pipe_log = Pipeline([("scaler", StandardScaler()), ("modelo", LogisticRegression(random_state=42))])
    pipe_log.fit(X_treino, y_treino)
    y_pred_log = pipe_log.predict(X_teste)
    y_proba_log = pipe_log.predict_proba(X_teste)[:, 1]
    resultados.append(_avaliar_classificacao("Regressão Logística", y_teste, y_pred_log, y_proba_log))
    modelos_treinados["logistic_regression"] = pipe_log

    # Random Forest
    pipe_rf = Pipeline(
        [
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=150, max_depth=3, min_samples_leaf=6, random_state=42
                ),
            )
        ]
    )
    pipe_rf.fit(X_treino, y_treino)
    y_pred_rf = pipe_rf.predict(X_teste)
    y_proba_rf = pipe_rf.predict_proba(X_teste)[:, 1]
    resultados.append(_avaliar_classificacao("Random Forest", y_teste, y_pred_rf, y_proba_rf))
    modelos_treinados["random_forest"] = pipe_rf

    importancias = dict(
        zip(features, pipe_rf.named_steps["modelo"].feature_importances_.round(4))
    )

    return modelos_treinados, resultados, importancias


def _avaliar_classificacao(nome: str, y_teste, y_pred, y_proba) -> ResultadoClassificacao:
    return ResultadoClassificacao(
        modelo=nome,
        acuracia=round(accuracy_score(y_teste, y_pred), 4),
        precisao=round(precision_score(y_teste, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_teste, y_pred, zero_division=0), 4),
        roc_auc=round(roc_auc_score(y_teste, y_proba), 4),
        matriz_confusao=confusion_matrix(y_teste, y_pred).tolist(),
    )


def salvar_metricas(resultado_regressao: ResultadoRegressao, resultados_classificacao: list[ResultadoClassificacao], caminho: str) -> None:
    payload = {
        "regressao_linear": asdict(resultado_regressao),
        "classificacao": [asdict(r) for r in resultados_classificacao],
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
