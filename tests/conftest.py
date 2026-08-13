"""
conftest.py — Fixtures compartilhadas para toda a suite de testes.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Adiciona a raiz do projeto ao path para que os scripts sejam importáveis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fixtures de dados ──────────────────────────────────────────────────────────


@pytest.fixture
def df_bruto_pequeno() -> pd.DataFrame:
    """Dataset bruto mínimo — idêntico ao schema de 01_coleta_dados.py."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "id_cliente": range(1, n + 1),
            "idade": np.random.randint(18, 70, size=n),
            "sexo": np.random.choice(["M", "F"], size=n),
            "renda_mensal": np.where(
                np.random.rand(n) < 0.05,
                np.nan,
                np.random.lognormal(8.5, 0.6, size=n),
            ),
            "tempo_conta_meses": np.random.randint(1, 120, size=n),
            "num_produtos": np.random.choice([1, 2, 3, 4], size=n),
            "tem_credito": np.random.choice([0, 1], size=n),
            "saldo": np.random.exponential(50000, size=n),
            "ativo": np.random.choice([0, 1], size=n),
            "canal_aquisicao": np.random.choice(
                ["Loja Física", "Online", "Telefone", "Indicação"], size=n
            ),
            "num_reclamacoes": np.random.poisson(0.6, size=n),
            "satisfacao": np.where(
                np.random.rand(n) < 0.03,
                np.nan,
                np.random.choice([1, 2, 3, 4, 5], size=n).astype(float),
            ),
            "churn": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        }
    )


@pytest.fixture
def df_sem_nans(df_bruto_pequeno) -> pd.DataFrame:
    """Dataset sem NaNs (pós tratar_valores_ausentes)."""
    df = df_bruto_pequeno.copy()
    df["renda_mensal"] = df["renda_mensal"].fillna(df["renda_mensal"].median())
    df["satisfacao"] = df["satisfacao"].fillna(df["satisfacao"].mode()[0])
    return df


@pytest.fixture
def df_com_features(df_sem_nans) -> pd.DataFrame:
    """Dataset com engineered features (pós engenharia_features)."""
    df = df_sem_nans.copy()
    df["renda_por_produto"] = df["renda_mensal"] / df["num_produtos"]
    df["reclamacao_por_tempo"] = df["num_reclamacoes"] / (df["tempo_conta_meses"] + 1)
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    mediana_saldo = df["saldo"].median()
    mediana_renda = df["renda_mensal"].median()
    df["cliente_premium"] = (
        (df["saldo"] > mediana_saldo) & (df["renda_mensal"] > mediana_renda)
    ).astype(int)
    return df
