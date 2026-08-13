"""
test_preprocessamento.py — Camadas 1 e 2: testes de 03_preprocessamento.py

Testa cada função de transformação de forma isolada usando
DataFrames em memória (sem I/O em disco).
"""

import importlib.util
import os

import numpy as np
import pandas as pd

# ── Importação por caminho ─────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(filename: str):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load("03_preprocessamento.py")

tratar_valores_ausentes = prep.tratar_valores_ausentes
tratar_outliers = prep.tratar_outliers
engenharia_features = prep.engenharia_features
codificar_categoricas = prep.codificar_categoricas


# ── Helpers ────────────────────────────────────────────────────────────────────


def _df_base(n: int = 60) -> pd.DataFrame:
    """DataFrame mínimo com schema completo, sem NaNs."""
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "id_cliente": range(1, n + 1),
            "idade": np.random.randint(18, 70, size=n),
            "sexo": np.random.choice(["M", "F"], size=n),
            "renda_mensal": np.random.lognormal(8.5, 0.6, size=n),
            "tempo_conta_meses": np.random.randint(1, 120, size=n),
            "num_produtos": np.random.choice([1, 2, 3, 4], size=n),
            "tem_credito": np.random.choice([0, 1], size=n),
            "saldo": np.random.exponential(50000, size=n),
            "ativo": np.random.choice([0, 1], size=n),
            "canal_aquisicao": np.random.choice(
                ["Loja Física", "Online", "Telefone", "Indicação"], size=n
            ),
            "num_reclamacoes": np.random.poisson(0.6, size=n),
            "satisfacao": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n),
            "churn": np.random.choice([0, 1], size=n),
        }
    )
    return df


def _df_com_nans(n: int = 60) -> pd.DataFrame:
    df = _df_base(n)
    idx_renda = np.random.choice(n, size=max(1, int(n * 0.05)), replace=False)
    idx_sat = np.random.choice(n, size=max(1, int(n * 0.03)), replace=False)
    df.loc[idx_renda, "renda_mensal"] = np.nan
    df.loc[idx_sat, "satisfacao"] = np.nan
    return df


# ── tratar_valores_ausentes ────────────────────────────────────────────────────


class TestTratarValoresAusentes:
    def test_remove_todos_nans(self):
        df = _df_com_nans()
        result = tratar_valores_ausentes(df.copy())
        assert result.isnull().sum().sum() == 0, "Ainda existem NaNs após tratamento"

    def test_nao_altera_linhas(self):
        df = _df_com_nans()
        result = tratar_valores_ausentes(df.copy())
        assert len(result) == len(df)

    def test_nao_altera_colunas(self):
        df = _df_com_nans()
        result = tratar_valores_ausentes(df.copy())
        assert set(result.columns) == set(df.columns)

    def test_renda_imputada_com_mediana(self):
        """Valores imputados em renda_mensal devem ser iguais à mediana original."""
        df = _df_com_nans()
        mediana_esperada = df["renda_mensal"].median()
        result = tratar_valores_ausentes(df.copy())
        # Linhas que eram NaN agora devem ter o valor da mediana
        assert (result["renda_mensal"] == mediana_esperada).any()

    def test_sem_nans_retorna_inalterado(self):
        df = _df_base()
        result = tratar_valores_ausentes(df.copy())
        pd.testing.assert_frame_equal(result, df)


# ── tratar_outliers ────────────────────────────────────────────────────────────


class TestTratarOutliers:
    def test_valores_dentro_do_range_p1_p99(self):
        df = _df_base()
        for col in ["renda_mensal", "saldo"]:
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
        result = tratar_outliers(df.copy())
        for col in ["renda_mensal", "saldo"]:
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            assert result[col].between(p1, p99).all(), f"{col}: valor fora de [P1, P99]"

    def test_nao_altera_outras_colunas(self):
        df = _df_base()
        result = tratar_outliers(df.copy())
        colunas_intocadas = [c for c in df.columns if c not in ("renda_mensal", "saldo")]
        for col in colunas_intocadas:
            pd.testing.assert_series_equal(result[col], df[col], check_names=True)

    def test_sem_nans_apos_winsorization(self):
        df = _df_base()
        result = tratar_outliers(df.copy())
        assert result[["renda_mensal", "saldo"]].isnull().sum().sum() == 0


# ── engenharia_features ────────────────────────────────────────────────────────


class TestEngenhariaFeatures:
    def test_novas_colunas_criadas(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        novas = {"renda_por_produto", "reclamacao_por_tempo", "faixa_etaria", "cliente_premium"}
        assert novas.issubset(set(result.columns))

    def test_renda_por_produto_positiva(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        assert (result["renda_por_produto"] > 0).all()

    def test_reclamacao_por_tempo_nao_negativa(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        assert (result["reclamacao_por_tempo"] >= 0).all()

    def test_sem_inf_nas_features(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        for col in ["renda_por_produto", "reclamacao_por_tempo"]:
            assert not np.isinf(result[col]).any(), f"Infinito encontrado em {col}"

    def test_sem_nans_nas_features(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        for col in ["renda_por_produto", "reclamacao_por_tempo", "cliente_premium"]:
            assert result[col].isnull().sum() == 0, f"NaN encontrado em {col}"

    def test_cliente_premium_binario(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        assert result["cliente_premium"].isin([0, 1]).all()

    def test_faixa_etaria_categorias(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        esperadas = {"18-25", "26-35", "36-45", "46-55", "56+"}
        assert set(result["faixa_etaria"].dropna().astype(str)).issubset(esperadas)

    def test_nao_remove_colunas_originais(self):
        df = _df_base()
        result = engenharia_features(df.copy())
        assert "renda_mensal" in result.columns
        assert "num_produtos" in result.columns


# ── codificar_categoricas ──────────────────────────────────────────────────────


class TestCodificarCategoricas:
    def _df_para_codificacao(self):
        df = _df_base()
        return engenharia_features(df.copy())

    def test_colunas_originais_removidas(self):
        df = self._df_para_codificacao()
        result = codificar_categoricas(df.copy())
        for col in ["sexo", "canal_aquisicao", "faixa_etaria"]:
            assert col not in result.columns, f"Coluna original '{col}' não foi removida"

    def test_dummies_criadas(self):
        df = self._df_para_codificacao()
        result = codificar_categoricas(df.copy())
        # Com drop_first=True, cada categórica gera (k-1) dummies
        assert any(c.startswith("sexo_") for c in result.columns)
        assert any(c.startswith("canal_aquisicao_") for c in result.columns)
        assert any(c.startswith("faixa_etaria_") for c in result.columns)

    def test_valores_dummies_sao_binarios(self):
        df = self._df_para_codificacao()
        result = codificar_categoricas(df.copy())
        dummy_cols = [
            c
            for c in result.columns
            if any(c.startswith(p) for p in ("sexo_", "canal_aquisicao_", "faixa_etaria_"))
        ]
        for col in dummy_cols:
            assert result[col].isin([0, 1]).all(), f"Dummy {col} com valor não-binário"

    def test_sem_nans_apos_codificacao(self):
        df = self._df_para_codificacao()
        result = codificar_categoricas(df.copy())
        assert result.isnull().sum().sum() == 0
