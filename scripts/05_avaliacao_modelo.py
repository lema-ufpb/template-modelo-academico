#!/usr/bin/env python3
"""
05_avaliacao_modelo.py — Avaliação Final dos Modelos
=====================================================

Este script avalia os modelos treinados no conjunto de teste,
gera métricas finais, tabelas LaTeX e gráficos de avaliação.

Uso:
    python scripts/05_avaliacao_modelo.py

Entradas:
    dados/processados/dataset_teste.csv
    modelos/*.joblib

Saídas:
    resultados/metricas/metricas_teste.csv
    resultados/metricas/coeficientes_regressao.csv
    resultados/metricas/matriz_confusao.csv
    resultados/figuras/curva_roc.png
    resultados/figuras/importancia_features.png
    latex/<doc>/tabelas/tab_comparacao_modelos.tex    (cada documento)
    latex/<doc>/tabelas/tab_coeficientes_modelo.tex   (cada documento)
    latex/<doc>/tabelas/tab_matriz_confusao.tex       (cada documento)
    latex/<doc>/tabelas/tab_estatisticas_descritivas.tex (cada documento)

Como adaptar para seu projeto:
    1. Ajuste as métricas em `avaliar_modelos()` conforme seu problema:
       - Classificação binária: AUC-ROC, F1, Precisão, Recall
       - Classificação multiclasse: macro/micro F1, confusion matrix NxN
       - Regressão: R², MAE, RMSE, MAPE

    2. Para problemas de regressão, substitua `plotar_curva_roc()`
       por gráficos de resíduos ou previsão vs. real.

    3. As funções `gerar_tabela_latex_*()` produzem tabelas LaTeX
       que são salvas em TODOS os diretórios latex/<doc>/tabelas/.
       Se você adicionar um novo tipo de documento, basta adicioná-lo
       à lista DOC_TYPES no topo do script.

    4. Para adicionar novas figuras ao trabalho, crie funções
       que salvem PNGs em `resultados/figuras/`. O script 06
       converterá automaticamente para EPS e distribuirá para
       cada diretório de documento.

    5. Se precisar de tabelas adicionais no LaTeX, siga o padrão
       das funções existentes: retorne uma string com o código
       LaTeX e use `_salvar_tabela()` para distribuir.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ── Configurações ──────────────────────────────────────────────
CAMINHO_TESTE = os.path.join("dados", "processados", "dataset_teste.csv")
CAMINHO_TREINO = os.path.join("dados", "processados", "dataset_treino.csv")
CAMINHO_BRUTO = os.path.join("dados", "brutos", "dataset_clientes.csv")
DIR_MODELOS = "modelos"
DIR_FIGURAS = os.path.join("resultados", "figuras")
DIR_METRICAS = os.path.join("resultados", "metricas")
DOC_TYPES = ["tcc_artigo", "tcc", "artigo", "dissertacao", "tese", "livro", "relatorio", "projeto"]

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
sns.set_theme(style="whitegrid")


def carregar_teste():
    """Carrega o dataset de teste."""
    df = pd.read_csv(CAMINHO_TESTE)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    print(f"Dataset de teste: {X.shape[0]} registros, {X.shape[1]} features")
    return X, y


def carregar_modelos() -> dict:
    """Carrega todos os modelos salvos."""
    modelos = {}
    for arquivo in sorted(os.listdir(DIR_MODELOS)):
        if arquivo.endswith(".joblib"):
            nome = arquivo.replace(".joblib", "").replace("_", " ").title()
            caminho = os.path.join(DIR_MODELOS, arquivo)
            modelos[nome] = joblib.load(caminho)
            print(f"  ✓ Carregado: {nome}")
    return modelos


def avaliar_modelos(modelos: dict, X, y) -> pd.DataFrame:
    """Calcula métricas no conjunto de teste para todos os modelos."""
    resultados = []

    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X)
        y_proba = modelo.predict_proba(X)[:, 1]

        resultado = {
            "Modelo": nome,
            "AUC-ROC": roc_auc_score(y, y_proba),
            "F1-Score": f1_score(y, y_pred),
            "Acurácia": accuracy_score(y, y_pred),
        }
        resultados.append(resultado)

        print(f"\n{nome}:")
        print(f"  AUC-ROC:  {resultado['AUC-ROC']:.4f}")
        print(f"  F1-Score: {resultado['F1-Score']:.4f}")
        print(f"  Acurácia: {resultado['Acurácia']:.4f}")

    return pd.DataFrame(resultados)


def plotar_curva_roc(modelos: dict, X, y):
    """Plota curvas ROC de todos os modelos em um único gráfico."""
    fig, ax = plt.subplots(figsize=(8, 7))
    cores = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    for (nome, modelo), cor in zip(modelos.items(), cores):
        y_proba = modelo.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        ax.plot(fpr, tpr, color=cor, lw=2, label=f"{nome} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.set_title("Curva ROC — Comparação de Modelos", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.02))
    ax.grid(True, alpha=0.3)

    caminho = os.path.join(DIR_FIGURAS, "curva_roc.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Curva ROC salva em: {caminho}")


def plotar_importancia_features(modelos: dict, X):
    """Plota importância de features do melhor modelo baseado em árvore."""
    modelo_arvore = None
    nome_arvore = None
    for nome, modelo in modelos.items():
        if hasattr(modelo, "feature_importances_"):
            modelo_arvore = modelo
            nome_arvore = nome

    if modelo_arvore is None:
        print("⚠ Nenhum modelo com feature_importances_ encontrado.")
        return

    importancias = pd.Series(
        modelo_arvore.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=True)

    top = importancias.tail(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top.index.tolist(), top.tolist(), color="#3498db", edgecolor="white")
    ax.set_xlabel("Importância")
    ax.set_title(f"Top 15 Features — {nome_arvore}", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, top.tolist()):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9
        )

    plt.tight_layout()
    caminho = os.path.join(DIR_FIGURAS, "importancia_features.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Importância de features salva em: {caminho}")


def _salvar_tabela(conteudo: str, nome: str):
    """Salva conteúdo LaTeX em todos os diretórios de documento."""
    for doc in DOC_TYPES:
        dirpath = os.path.join("tex", doc, "tabelas")
        os.makedirs(dirpath, exist_ok=True)
        caminho = os.path.join(dirpath, nome)
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)


def gerar_tabela_latex_modelos(df_metricas: pd.DataFrame) -> str:
    """Gera tabela LaTeX de comparação de modelos."""
    linhas = []
    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(r"\caption{Comparação de Performance dos Modelos no Conjunto de Teste}")
    linhas.append(r"\label{tab:comparacao}")
    linhas.append(r"\small")
    linhas.append(r"\begin{tabular}{lccc}")
    linhas.append(r"\toprule")
    linhas.append(r"\textbf{Modelo} & \textbf{AUC-ROC} & \textbf{F1-Score} & \textbf{Acurácia} \\")
    linhas.append(r"\midrule")

    for _, row in df_metricas.iterrows():
        linha = (
            f"{row['Modelo']} & {row['AUC-ROC']:.4f}"
            f" & {row['F1-Score']:.4f} & {row['Acurácia']:.4f} \\\\"
        )
        linhas.append(linha)

    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    linhas.append(r"\vspace{2mm}")
    linhas.append(r"\fonte{Elaborado pelo autor (\the\year)}")
    linhas.append(r"\end{table}")

    return "\n".join(linhas)


def gerar_tabela_latex_coeficientes(modelos: dict, X) -> str | None:
    """Gera tabela LaTeX com coeficientes da Regressão Logística."""
    modelo_lr = None
    for nome, modelo in modelos.items():
        if hasattr(modelo, "coef_"):
            modelo_lr = modelo
            break

    if modelo_lr is None:
        print("⚠ Nenhum modelo com coeficientes encontrado.")
        return None

    coefs = pd.DataFrame(
        {
            "Variável": X.columns,
            "Coeficiente (β)": modelo_lr.coef_[0],
        }
    ).sort_values("Coeficiente (β)", key=abs, ascending=False)

    caminho_csv = os.path.join(DIR_METRICAS, "coeficientes_regressao.csv")
    coefs.to_csv(caminho_csv, index=False, encoding="utf-8")

    top_coefs = coefs.head(10)
    linhas = []
    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(r"\caption{Coeficientes da Regressão Logística (Top 10 por magnitude)}")
    linhas.append(r"\label{tab:coeficientes}")
    linhas.append(r"\label{tab:coef}")
    linhas.append(r"\small")
    linhas.append(r"\begin{tabular}{lrr}")
    linhas.append(r"\toprule")
    linhas.append(r"\textbf{Variável} & \textbf{Coeficiente ($\beta$)} & \textbf{Odds Ratio} \\")
    linhas.append(r"\midrule")

    for _, row in top_coefs.iterrows():
        var_name = row["Variável"].replace("_", r"\_")
        coef = row["Coeficiente (β)"]
        odds = np.exp(coef)
        linhas.append(f"{var_name} & {coef:.4f} & {odds:.4f} \\\\")

    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    linhas.append(r"\vspace{2mm}")
    linhas.append(r"\fonte{Elaborado pelo autor (\the\year)}")
    linhas.append(r"\end{table}")

    return "\n".join(linhas)


def gerar_tabela_latex_confusao(modelos: dict, X, y) -> str:
    """Gera tabela LaTeX com matriz de confusão do melhor modelo."""
    melhor_nome, melhor_modelo, melhor_auc = None, None, 0
    for nome, modelo in modelos.items():
        y_proba = modelo.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        if auc > melhor_auc:
            melhor_nome, melhor_modelo, melhor_auc = nome, modelo, auc

    if melhor_modelo is None:
        return ""

    y_pred = melhor_modelo.predict(X)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    df_cm = pd.DataFrame(
        cm, index=["Real: Não-Churn", "Real: Churn"], columns=["Pred: Não-Churn", "Pred: Churn"]
    )
    caminho_csv = os.path.join(DIR_METRICAS, "matriz_confusao.csv")
    df_cm.to_csv(caminho_csv, encoding="utf-8")

    linhas = []
    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(f"\\caption{{Matriz de Confusão — {melhor_nome}}}")
    linhas.append(r"\label{tab:matriz}")
    linhas.append(r"\small")
    linhas.append(r"\begin{tabular}{lcc|c}")
    linhas.append(r"\toprule")
    linhas.append(r"& \textbf{Pred: Não-Churn} & \textbf{Pred: Churn} & \textbf{Total} \\")
    linhas.append(r"\midrule")
    linhas.append(f"\\textbf{{Real: Não-Churn}} & {tn} & {fp} & {tn + fp} \\\\")
    linhas.append(f"\\textbf{{Real: Churn}} & {fn} & {tp} & {fn + tp} \\\\")
    linhas.append(r"\midrule")
    linhas.append(f"\\textbf{{Total}} & {tn + fn} & {fp + tp} & {tn + fp + fn + tp} \\\\")
    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    linhas.append(r"\vspace{2mm}")
    linhas.append(r"\fonte{Elaborado pelo autor (\the\year)}")
    linhas.append(r"\end{table}")

    return "\n".join(linhas)


def gerar_tabela_dicionario_dados() -> str:
    """Gera tabela LaTeX com o dicionário de dados do dataset."""
    linhas = []
    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(r"\caption{Dicionário de Dados do Dataset de Clientes}")
    linhas.append(r"\label{tab:dicionario}")
    linhas.append(r"\small")
    linhas.append(r"\begin{tabular}{l l p{5.5cm} c}")
    linhas.append(r"\toprule")
    linhas.append(r"\textbf{Variável} & \textbf{Tipo} & \textbf{Descrição} & \textbf{Exemplo} \\")
    linhas.append(r"\midrule")
    linhas.append(r"\texttt{id\_cliente} & Inteiro & Identificador único do cliente & 1, 2, 3 \\")
    linhas.append(r"\texttt{idade} & Inteiro & Idade do cliente em anos & 18--70 \\")
    linhas.append(r"\texttt{sexo} & Categórico & Sexo biológico (M/F) & M, F \\")
    linhas.append(r"\texttt{renda\_mensal} & Contínuo & Renda mensal em R\$ & 2.500--15.000 \\")
    linhas.append(
        r"\texttt{tempo\_conta\_meses} & Inteiro & Tempo como cliente (meses) & 1--120 \\"
    )
    linhas.append(
        r"\texttt{num\_produtos} & Inteiro & Quantidade de produtos contratados & 1--4 \\"
    )
    linhas.append(r"\texttt{tem\_credito} & Binário & Possui crédito especial (0/1) & 0, 1 \\")
    linhas.append(r"\texttt{saldo} & Contínuo & Saldo médio em conta (R\$) & 0--250.000 \\")
    linhas.append(r"\texttt{ativo} & Binário & Cliente está ativo (0/1) & 0, 1 \\")
    linhas.append(
        r"\texttt{canal\_aquisicao} & Categórico & Canal de aquisição do cliente "
        r"& Loja Física, Online \\"
    )
    linhas.append(r"\texttt{num\_reclamacoes} & Inteiro & Número de reclamações abertas & 0--5 \\")
    linhas.append(r"\texttt{satisfacao} & Ordinal & Nível de satisfação (1--5) & 1, 2, 3, 4, 5 \\")
    linhas.append(r"\texttt{churn} & Binário & Evadiu-se? (target: 0/1) & 0, 1 \\")
    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    linhas.append(r"\vspace{2mm}")
    linhas.append(r"\fonte{Elaborado pelo autor (\the\year)}")
    linhas.append(r"\end{table}")
    return "\n".join(linhas)


def gerar_tabela_latex_descritivas() -> str:
    """Gera tabela LaTeX com estatísticas descritivas do dataset bruto."""
    df = pd.read_csv(CAMINHO_BRUTO)
    colunas_num = [
        "idade",
        "renda_mensal",
        "tempo_conta_meses",
        "num_produtos",
        "saldo",
        "num_reclamacoes",
        "satisfacao",
    ]

    stats = df[colunas_num].agg(["count", "mean", "median", "std", "min", "max"]).T
    stats.columns = ["n", "Média", "Mediana", "Desvio Padrão", "Mín", "Máx"]
    stats["n"] = stats["n"].astype(int)

    linhas = []
    linhas.append(r"\begin{table}[htbp]")
    linhas.append(r"\centering")
    linhas.append(r"\caption{Estatísticas Descritivas das Variáveis Numéricas}")
    linhas.append(r"\label{tab:estatisticas}")
    linhas.append(r"\small")
    linhas.append(r"\setlength{\tabcolsep}{4pt}")
    linhas.append(r"\begin{tabular}{lrrrrrr}")
    linhas.append(r"\toprule")
    header = (
        r"\textbf{Variável} & \textbf{n} & \textbf{Média}"
        r" & \textbf{Mediana} & \textbf{DP} & \textbf{Mín} & \textbf{Máx} \\"
    )
    linhas.append(header)
    linhas.append(r"\midrule")

    for var, row in stats.iterrows():
        var_fmt = str(var).replace("_", r"\_")
        linhas.append(
            f"{var_fmt} & {int(row['n'])} & {row['Média']:.2f} & {row['Mediana']:.2f} "
            f"& {row['Desvio Padrão']:.2f} & {row['Mín']:.2f} & {row['Máx']:.2f} \\\\"
        )

    linhas.append(r"\bottomrule")
    linhas.append(r"\end{tabular}")
    linhas.append(r"\vspace{2mm}")
    linhas.append(r"\fonte{Elaborado pelo autor (\the\year)}")
    linhas.append(r"\end{table}")

    return "\n".join(linhas)


def main():
    """Ponto de entrada principal."""
    print("=" * 60)
    print("05 — Avaliação Final dos Modelos")
    print("=" * 60)

    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_METRICAS, exist_ok=True)

    X, y = carregar_teste()
    modelos = carregar_modelos()

    print("\nMétricas no Conjunto de Teste:")
    print("-" * 50)
    df_metricas = avaliar_modelos(modelos, X, y)

    caminho_metricas = os.path.join(DIR_METRICAS, "metricas_teste.csv")
    df_metricas.to_csv(caminho_metricas, index=False, encoding="utf-8")
    print(f"\n✓ Métricas salvas em: {caminho_metricas}")

    plotar_curva_roc(modelos, X, y)
    plotar_importancia_features(modelos, X)

    print("\nGerando tabelas LaTeX:")
    print("-" * 50)

    tabelas = [
        ("tab_comparacao_modelos.tex", gerar_tabela_latex_modelos(df_metricas)),
        ("tab_coeficientes_modelo.tex", gerar_tabela_latex_coeficientes(modelos, X)),
        ("tab_matriz_confusao.tex", gerar_tabela_latex_confusao(modelos, X, y)),
        ("tab_estatisticas_descritivas.tex", gerar_tabela_latex_descritivas()),
        ("tab_dicionario_dados.tex", gerar_tabela_dicionario_dados()),
    ]

    for nome, conteudo in tabelas:
        if conteudo is None:
            continue
        _salvar_tabela(conteudo, nome)
        print(f"  ✓ {nome}")

    print("\n✓ Avaliação concluída com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
