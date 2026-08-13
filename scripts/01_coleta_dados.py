#!/usr/bin/env python3
"""
01_coleta_dados.py — Geração de Dados Fictícios para Demonstração
=================================================================

Este script gera um dataset fictício de **churn de clientes** para
servir como demonstração do pipeline completo de Ciência de Dados.

O dataset simula dados de uma empresa de telecomunicações com ~1000
clientes, contendo variáveis numéricas e categóricas.

Uso:
    python scripts/01_coleta_dados.py

Saída:
    dados/brutos/dataset_clientes.csv

Como adaptar para seu projeto:
    Este é provavelmente o primeiro script que você vai modificar.
    Substitua o corpo da função `gerar_dataset()` pela lógica de
    obtenção dos seus dados reais. Exemplos de cenários comuns:

    - Leitura de CSV local:
        df = pd.read_csv("dados/brutos/meu_dataset.csv")

    - Download de API:
        import requests
        response = requests.get("https://api.exemplo.com/dados")
        df = pd.DataFrame(response.json())

    - Download de base pública (Kaggle, UCI, etc.):
        from urllib.request import urlretrieve
        urlretrieve("https://url-do-dataset.csv", CAMINHO_SAIDA)

    - Web scraping:
        from bs4 import BeautifulSoup
        # ... lógica de scraping

    O importante é que, ao final, o script salve o arquivo em
    `dados/brutos/` sem nenhuma transformação — os dados brutos
    devem ser preservados exatamente como foram coletados.
"""

import os

import numpy as np
import pandas as pd

# ── Configurações ──────────────────────────────────────────────
SEED = 42
N_CLIENTES = 1000
CAMINHO_SAIDA = os.path.join("dados", "brutos", "dataset_clientes.csv")

# ── Reprodutibilidade ─────────────────────────────────────────
np.random.seed(SEED)


def gerar_dataset(n: int = N_CLIENTES) -> pd.DataFrame:
    """Gera um DataFrame fictício de clientes com risco de churn.

    Parameters
    ----------
    n : int
        Número de registros a gerar.

    Returns
    -------
    pd.DataFrame
        Dataset com variáveis numéricas e categóricas.
    """
    dados = {
        "id_cliente": range(1, n + 1),
        "idade": np.random.randint(18, 70, size=n),
        "sexo": np.random.choice(["M", "F"], size=n, p=[0.48, 0.52]),
        "renda_mensal": np.round(np.random.lognormal(mean=8.5, sigma=0.6, size=n), 2),
        "tempo_conta_meses": np.random.randint(1, 120, size=n),
        "num_produtos": np.random.choice([1, 2, 3, 4], size=n, p=[0.40, 0.35, 0.15, 0.10]),
        "tem_credito": np.random.choice([0, 1], size=n, p=[0.25, 0.75]),
        "saldo": np.round(np.random.exponential(scale=50000, size=n), 2),
        "ativo": np.random.choice([0, 1], size=n, p=[0.12, 0.88]),
        "canal_aquisicao": np.random.choice(
            ["Loja Física", "Online", "Telefone", "Indicação"],
            size=n,
            p=[0.30, 0.40, 0.15, 0.15],
        ),
    }

    df = pd.DataFrame(dados)

    # ── Variáveis com relacionamento realista ──────────────────
    # Clientes inativos e com pouco tempo de conta tendem a
    # registrar mais reclamações.
    reclamacoes_base = np.random.poisson(lam=0.6, size=n)
    df["num_reclamacoes"] = np.where(
        (df["tempo_conta_meses"] < 12) | (df["ativo"] == 0),
        reclamacoes_base + np.random.poisson(lam=1.5, size=n),
        reclamacoes_base,
    )

    # Satisfação depende inversamente das reclamações e
    # diretamente do tempo de conta.
    satisfacao_continua = (
        4.5
        - 0.6 * df["num_reclamacoes"]
        + 0.008 * df["tempo_conta_meses"]
        + np.random.normal(0, 0.6, size=n)
    )
    df["satisfacao"] = np.clip(np.round(satisfacao_continua), 1, 5).astype(int)

    # ── Variável-alvo (churn) via modelo logístico ─────────────
    # A probabilidade de churn é modelada como uma função
    # logística (sigmóide) das variáveis preditoras, garantindo
    # que os modelos de ML consigam aprender o padrão subjacente.
    reclamacoes_norm = df["num_reclamacoes"] / df["num_reclamacoes"].max()
    log_odds = (
        -2.5  # bias
        + 1.5 * reclamacoes_norm  # +reclamações → +churn
        + 2.0 * (1 - df["satisfacao"] / 5)  # -satisfação → +churn
        + 1.2 * (1 - df["tempo_conta_meses"] / 120)  # -tempo → +churn
        + 1.0 * (1 - df["ativo"])  # inativo → +churn
        + 0.5 * (1 - df["num_produtos"] / 4)  # -produtos → +churn
    )
    prob_churn = 1 / (1 + np.exp(-log_odds))
    df["churn"] = np.random.binomial(1, prob_churn)

    # ── Introduzir valores ausentes realistas (~5%) ────────────
    idx_renda = np.random.choice(n, size=int(n * 0.05), replace=False)
    df.loc[idx_renda, "renda_mensal"] = np.nan

    idx_satisfacao = np.random.choice(n, size=int(n * 0.03), replace=False)
    df.loc[idx_satisfacao, "satisfacao"] = np.nan

    return df


def main():
    """Ponto de entrada principal."""
    print("=" * 60)
    print("01 — Geração de Dados Fictícios")
    print("=" * 60)

    # Criar diretório se não existir
    os.makedirs(os.path.dirname(CAMINHO_SAIDA), exist_ok=True)

    # Gerar e salvar
    df = gerar_dataset()
    df.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")

    print(f"\n✓ Dataset gerado com {len(df)} registros e {len(df.columns)} colunas.")
    print(f"✓ Salvo em: {CAMINHO_SAIDA}")
    print("\nDistribuição da variável-alvo (churn):")
    print(df["churn"].value_counts().to_string())
    print("\nValores ausentes:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    print("\nPrimeiras 5 linhas:")
    print(df.head().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
