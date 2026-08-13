#!/usr/bin/env python3
"""
02_analise_exploratoria.py — Análise Exploratória de Dados (EDA)
================================================================

Este script realiza a análise exploratória do dataset de churn,
gerando estatísticas descritivas e visualizações.

Uso:
    python scripts/02_analise_exploratoria.py

Entradas:
    dados/brutos/dataset_clientes.csv

Saídas:
    resultados/figuras/distribuicao_features.png
    resultados/figuras/correlacao_heatmap.png
    resultados/metricas/estatisticas_descritivas.csv

Como adaptar para seu projeto:
    1. Atualize `CAMINHO_DADOS` para apontar para seu dataset.
    2. Na função `estatisticas_descritivas()`, ajuste a lista
       `colunas_num` com os nomes das suas variáveis numéricas.
    3. Na função `plotar_distribuicoes()`, ajuste `colunas` e
       substitua "churn" pelo nome da sua variável-alvo.
    4. Na função `plotar_churn_por_categoria()`, ajuste `colunas_cat`
       com suas variáveis categóricas.
    5. Adicione novas visualizações conforme necessário:
       - Boxplots para identificar outliers
       - Gráficos de dispersão para relações bivariadas
       - Análise de séries temporais (se aplicável)
    6. A EDA é exploratória por natureza — não tenha medo de
       criar gráficos "feios" nesta fase. O objetivo é entender
       os dados, não publicar figuras bonitas (isso vem depois).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configurações ──────────────────────────────────────────────
CAMINHO_DADOS = os.path.join("dados", "brutos", "dataset_clientes.csv")
DIR_FIGURAS = os.path.join("resultados", "figuras")
DIR_METRICAS = os.path.join("resultados", "metricas")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.figsize": (12, 8),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)
sns.set_theme(style="whitegrid", palette="muted")


def carregar_dados() -> pd.DataFrame:
    """Carrega o dataset bruto."""
    print(f"Carregando dados de: {CAMINHO_DADOS}")
    df = pd.read_csv(CAMINHO_DADOS)
    print(f"  → {df.shape[0]} registros, {df.shape[1]} colunas\n")
    return df


def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula e salva estatísticas descritivas das variáveis numéricas."""
    colunas_num = [
        "idade",
        "renda_mensal",
        "tempo_conta_meses",
        "num_produtos",
        "saldo",
        "num_reclamacoes",
        "satisfacao",
    ]

    stats = df[colunas_num].describe().T
    stats["mediana"] = df[colunas_num].median()
    stats["cv"] = (stats["std"] / stats["mean"] * 100).round(2)  # coef. variação
    stats = stats[["count", "mean", "mediana", "std", "min", "25%", "50%", "75%", "max", "cv"]]
    stats.columns = [
        "n",
        "média",
        "mediana",
        "desvio_padrão",
        "mín",
        "Q1",
        "Q2",
        "Q3",
        "máx",
        "CV(%)",
    ]

    caminho = os.path.join(DIR_METRICAS, "estatisticas_descritivas.csv")
    stats.to_csv(caminho, encoding="utf-8")
    print(f"✓ Estatísticas descritivas salvas em: {caminho}")
    print(stats.to_string())
    print()
    return stats


def plotar_distribuicoes(df: pd.DataFrame):
    """Gera histogramas das variáveis numéricas segmentados por churn."""
    colunas = [
        "idade",
        "renda_mensal",
        "tempo_conta_meses",
        "saldo",
        "num_reclamacoes",
        "satisfacao",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Distribuição das Variáveis por Status de Churn", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.ravel(), colunas):
        for churn_val, cor, label in [(0, "#3498db", "Não-Churn"), (1, "#e74c3c", "Churn")]:
            subset = df[df["churn"] == churn_val][col].dropna()
            ax.hist(subset, bins=25, alpha=0.6, color=cor, label=label, edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=9)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    caminho = os.path.join(DIR_FIGURAS, "distribuicao_features.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Distribuições salvas em: {caminho}")


def plotar_correlacao(df: pd.DataFrame):
    """Gera heatmap de correlação das variáveis numéricas."""
    colunas_num = df.select_dtypes(include="number").columns.tolist()
    colunas_num = [c for c in colunas_num if c != "id_cliente"]

    corr = df[colunas_num].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,  # type: ignore
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Matriz de Correlação de Pearson", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    caminho = os.path.join(DIR_FIGURAS, "correlacao_heatmap.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Heatmap de correlação salvo em: {caminho}")


def plotar_churn_por_categoria(df: pd.DataFrame):
    """Gera gráficos de barras da taxa de churn por variável categórica."""
    colunas_cat = ["sexo", "canal_aquisicao", "tem_credito", "ativo"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Taxa de Churn por Variável Categórica", fontsize=14, fontweight="bold")

    for ax, col in zip(axes.ravel(), colunas_cat):
        taxa = df.groupby(col)["churn"].mean().sort_values(ascending=False)
        bars = ax.bar(taxa.index.astype(str), taxa.values, color="#2ecc71", edgecolor="white")
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Taxa de Churn")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, taxa.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.1%}",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    caminho = os.path.join(DIR_FIGURAS, "churn_por_categoria.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Churn por categoria salvo em: {caminho}")


def main():
    """Ponto de entrada principal."""
    print("=" * 60)
    print("02 — Análise Exploratória de Dados (EDA)")
    print("=" * 60)

    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_METRICAS, exist_ok=True)

    df = carregar_dados()
    estatisticas_descritivas(df)
    plotar_distribuicoes(df)
    plotar_correlacao(df)
    plotar_churn_por_categoria(df)

    print("\n✓ Análise exploratória concluída com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
