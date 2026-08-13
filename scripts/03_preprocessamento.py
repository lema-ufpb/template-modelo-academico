#!/usr/bin/env python3
"""
03_preprocessamento.py — Pré-processamento e Engenharia de Features
====================================================================

Este script realiza limpeza de dados, tratamento de valores ausentes,
codificação de variáveis categóricas e engenharia de features.

Uso:
    python scripts/03_preprocessamento.py

Entradas:
    dados/brutos/dataset_clientes.csv

Saídas:
    dados/processados/dataset_processado.csv
    dados/processados/dataset_treino.csv
    dados/processados/dataset_teste.csv

Como adaptar para seu projeto:
    Este script é o "coração" da preparação de dados. Para adaptá-lo:

    1. `tratar_valores_ausentes()`: Ajuste as estratégias de imputação
       para cada variável do seu dataset. Regra geral:
       - Variáveis numéricas assimétricas → mediana
       - Variáveis numéricas simétricas → média
       - Variáveis categóricas/ordinais → moda
       - Muitos ausentes (>50%) → considere remover a variável

    2. `tratar_outliers()`: Defina quais variáveis recebem winsorização
       e em quais percentis. Para dados com outliers legítimos (ex:
       renda de CEOs), considere não tratar.

    3. `engenharia_features()`: Crie variáveis derivadas que façam
       sentido para o seu domínio. Pergunte-se: "Que combinações
       de variáveis poderiam ser informativas?"

    4. `codificar_categoricas()`: Ajuste a lista de variáveis a serem
       codificadas. Para categóricas com muitas categorias (>10),
       considere Target Encoding em vez de One-Hot.

    5. `dividir_treino_teste()`: Ajuste PROPORCAO_TESTE e as colunas
       a normalizar conforme suas variáveis.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Configurações ──────────────────────────────────────────────
SEED = 42
CAMINHO_DADOS = os.path.join("dados", "brutos", "dataset_clientes.csv")
DIR_SAIDA = os.path.join("dados", "processados")
PROPORCAO_TESTE = 0.20


def carregar_dados() -> pd.DataFrame:
    """Carrega o dataset bruto."""
    df = pd.read_csv(CAMINHO_DADOS)
    print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df


def tratar_valores_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Trata valores ausentes com estratégias apropriadas.

    - renda_mensal: imputação pela mediana (distribuição assimétrica)
    - satisfacao: imputação pela moda (variável ordinal)
    """
    print("\nTratamento de valores ausentes:")
    ausentes = df.isnull().sum()
    print(ausentes[ausentes > 0].to_string())

    # Renda mensal → mediana (robusta a outliers)
    mediana_renda = df["renda_mensal"].median()
    df["renda_mensal"] = df["renda_mensal"].fillna(mediana_renda)
    print(f"  → renda_mensal: imputada com mediana = {mediana_renda:.2f}")

    # Satisfação → moda (variável ordinal)
    moda_satisfacao = df["satisfacao"].mode()[0]
    df["satisfacao"] = df["satisfacao"].fillna(moda_satisfacao)
    print(f"  → satisfacao: imputada com moda = {moda_satisfacao}")

    return df


def tratar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Trata outliers extremos via winsorização (clipar nos percentis 1%-99%).

    Aplica-se apenas às variáveis numéricas contínuas.
    """
    colunas = ["renda_mensal", "saldo"]
    print("\nTratamento de outliers (winsorização P1–P99):")

    for col in colunas:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        n_antes = ((df[col] < p1) | (df[col] > p99)).sum()
        df[col] = df[col].clip(lower=p1, upper=p99)
        print(f"  → {col}: {n_antes} valores ajustados [{p1:.2f}, {p99:.2f}]")

    return df


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas (features engenheiradas).

    Hipóteses de negócio:
    - renda_por_produto: clientes com menor renda por produto contratado
      podem ter maior insatisfação.
    - reclamacao_por_tempo: taxa de reclamação normalizada pelo tempo de
      conta indica intensidade de insatisfação.
    - faixa_etaria: segmentação demográfica para capturar padrões geracionais.
    """
    print("\nEngenharia de features:")

    # Renda por produto contratado
    df["renda_por_produto"] = df["renda_mensal"] / df["num_produtos"]
    print("  + renda_por_produto = renda_mensal / num_produtos")

    # Taxa de reclamação por tempo de conta
    df["reclamacao_por_tempo"] = df["num_reclamacoes"] / (df["tempo_conta_meses"] + 1)
    print("  + reclamacao_por_tempo = num_reclamacoes / (tempo_conta_meses + 1)")

    # Faixa etária
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    print("  + faixa_etaria: [18-25, 26-35, 36-45, 46-55, 56+]")

    # Indicador de cliente premium (saldo acima da mediana E renda acima da mediana)
    mediana_saldo = df["saldo"].median()
    mediana_renda = df["renda_mensal"].median()
    df["cliente_premium"] = (
        (df["saldo"] > mediana_saldo) & (df["renda_mensal"] > mediana_renda)
    ).astype(int)
    print("  + cliente_premium: saldo > mediana AND renda > mediana")

    return df


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variáveis categóricas via One-Hot Encoding."""
    print("\nCodificação de variáveis categóricas (One-Hot Encoding):")

    colunas_cat = ["sexo", "canal_aquisicao", "faixa_etaria"]
    df = pd.get_dummies(df, columns=colunas_cat, drop_first=True, dtype=int)

    novas_colunas = [c for c in df.columns if any(c.startswith(p) for p in colunas_cat)]
    print(f"  → Colunas criadas: {novas_colunas}")

    return df


def dividir_treino_teste(df: pd.DataFrame):
    """Divide o dataset em treino e teste de forma estratificada."""
    print(f"\nDivisão treino/teste ({1 - PROPORCAO_TESTE:.0%}/{PROPORCAO_TESTE:.0%}):")

    # Remover coluna de identificação
    df = df.drop(columns=["id_cliente"], errors="ignore")

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=PROPORCAO_TESTE,
        random_state=SEED,
        stratify=y,
    )

    # Normalização das variáveis numéricas
    colunas_norm = [
        "idade",
        "renda_mensal",
        "tempo_conta_meses",
        "saldo",
        "num_reclamacoes",
        "satisfacao",
        "renda_por_produto",
        "reclamacao_por_tempo",
    ]
    colunas_norm = [c for c in colunas_norm if c in X_treino.columns]

    scaler = StandardScaler()
    X_treino[colunas_norm] = scaler.fit_transform(X_treino[colunas_norm])
    X_teste[colunas_norm] = scaler.transform(X_teste[colunas_norm])

    # Reunir X e y para salvar
    treino = pd.concat([X_treino, y_treino], axis=1)
    teste = pd.concat([X_teste, y_teste], axis=1)

    print(f"  → Treino: {treino.shape[0]} registros")
    print(f"  → Teste:  {teste.shape[0]} registros")
    print(f"  → Churn no treino: {y_treino.mean():.2%}")
    print(f"  → Churn no teste:  {y_teste.mean():.2%}")

    return treino, teste


def main():
    """Ponto de entrada principal."""
    print("=" * 60)
    print("03 — Pré-processamento e Engenharia de Features")
    print("=" * 60)

    os.makedirs(DIR_SAIDA, exist_ok=True)

    df = carregar_dados()
    df = tratar_valores_ausentes(df)
    df = tratar_outliers(df)
    df = engenharia_features(df)
    df = codificar_categoricas(df)

    # Salvar dataset processado completo
    caminho_completo = os.path.join(DIR_SAIDA, "dataset_processado.csv")
    df.to_csv(caminho_completo, index=False, encoding="utf-8")
    print(f"\n✓ Dataset processado salvo em: {caminho_completo}")
    print(f"  → {df.shape[0]} linhas × {df.shape[1]} colunas")

    # Dividir e salvar treino/teste
    treino, teste = dividir_treino_teste(df)

    caminho_treino = os.path.join(DIR_SAIDA, "dataset_treino.csv")
    caminho_teste = os.path.join(DIR_SAIDA, "dataset_teste.csv")
    treino.to_csv(caminho_treino, index=False, encoding="utf-8")
    teste.to_csv(caminho_teste, index=False, encoding="utf-8")
    print(f"✓ Treino salvo em: {caminho_treino}")
    print(f"✓ Teste salvo em:  {caminho_teste}")

    print("\n✓ Pré-processamento concluído com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
