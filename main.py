"""
main.py
-------
Pipeline completo do projeto:
1. Gera/carrega dados macro
2. Cria features (lags + target binário)
3. Split cronológico treino/teste
4. Treina Regressão Linear (previsão contínua) e Regressão Logística +
   Random Forest (classificação de "alta inadimplência")
5. Salva métricas em outputs/metricas/ e gráficos em outputs/graficos/
6. Exporta os modelos treinados em models/
"""

import pickle
import sys

sys.path.insert(0, "src")

from data_loader import carregar_serie
from feature_engineering import criar_atributos, colunas_preditoras
from modeling import split_cronologico, treinar_regressao, treinar_classificacao, salvar_metricas
from visualizer import (
    plot_serie_historica,
    plot_correlacao,
    plot_real_vs_previsto,
    plot_roc_curves,
    plot_importancia_features,
)


def main():
    print("=" * 60)
    print("PIPELINE: Análise e Previsão de Inadimplência (Macro)")
    print("=" * 60)

    # 1. Carregar dados
    df = carregar_serie("data/raw/serie_macro_inadimplencia.csv")
    print(f"\n[1/6] Dados carregados: {len(df)} meses ({df['data'].min().date()} a {df['data'].max().date()})")

    # 2. Features
    df_feat = criar_atributos(df)
    df_feat.to_csv("data/processed/serie_macro_features.csv", index=False)
    print(f"[2/6] Features criadas: {df_feat.shape[1]} colunas, {len(df_feat)} linhas úteis após lags")

    # 3. Split cronológico
    df_treino, df_teste = split_cronologico(df_feat, frac_treino=0.8)
    print(f"[3/6] Split cronológico: treino={len(df_treino)} meses | teste={len(df_teste)} meses")
    print(f"       Treino: {df_treino['data'].min().date()} a {df_treino['data'].max().date()}")
    print(f"       Teste:  {df_teste['data'].min().date()} a {df_teste['data'].max().date()}")

    # 4a. Regressão Linear
    pipe_reg, resultado_reg, y_pred_reg = treinar_regressao(df_treino, df_teste)
    print(f"\n[4/6] Regressão Linear — MAE: {resultado_reg.mae} | RMSE: {resultado_reg.rmse} | R²: {resultado_reg.r2}")

    # 4b. Classificação
    modelos_clf, resultados_clf, importancias = treinar_classificacao(df_treino, df_teste)
    for r in resultados_clf:
        print(f"       {r.modelo:22s} — Acurácia: {r.acuracia} | Recall: {r.recall} | ROC-AUC: {r.roc_auc}")

    # 5. Salvar métricas
    salvar_metricas(resultado_reg, resultados_clf, "outputs/metricas/metricas.json")
    print("\n[5/6] Métricas salvas em outputs/metricas/metricas.json")

    # 6. Gráficos
    plot_serie_historica(df, "outputs/graficos/01_serie_historica.png")
    plot_correlacao(
        df_feat,
        ["taxa_desemprego", "taxa_juros", "taxa_inadimplencia"],
        "outputs/graficos/02_matriz_correlacao.png",
    )
    plot_real_vs_previsto(df_teste["data"], df_teste["taxa_inadimplencia"], y_pred_reg, "outputs/graficos/03_real_vs_previsto.png")
    plot_roc_curves(modelos_clf, df_teste[colunas_preditoras()], df_teste["alta_inadimplencia"], "outputs/graficos/04_curva_roc.png")
    plot_importancia_features(importancias, "outputs/graficos/05_importancia_features.png")
    print("[6/6] Gráficos salvos em outputs/graficos/")

    # Exportar modelos
    with open("models/modelo_regressao_linear.pkl", "wb") as f:
        pickle.dump(pipe_reg, f)
    with open("models/modelo_random_forest.pkl", "wb") as f:
        pickle.dump(modelos_clf["random_forest"], f)
    with open("models/modelo_logistic_regression.pkl", "wb") as f:
        pickle.dump(modelos_clf["logistic_regression"], f)

    print("\nPipeline concluído com sucesso. Modelos salvos em models/.")
    print("=" * 60)

    return resultado_reg, resultados_clf, importancias


if __name__ == "__main__":
    main()
