"""
test_coleta_dados.py — Camada 1: testes unitários de 01_coleta_dados.py
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

# ── Importação por caminho (prefixo numérico impede import normal) ─────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(filename: str):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


coleta = _load("01_coleta_dados.py")
gerar_dataset = coleta.gerar_dataset

# ── Schema esperado ────────────────────────────────────────────────────────────

COLUNAS_ESPERADAS = {
    "id_cliente",
    "idade",
    "sexo",
    "renda_mensal",
    "tempo_conta_meses",
    "num_produtos",
    "tem_credito",
    "saldo",
    "ativo",
    "canal_aquisicao",
    "num_reclamacoes",
    "satisfacao",
    "churn",
}


class TestGerardataset:
    def test_shape_padrao(self):
        df = gerar_dataset(n=100)
        assert df.shape[0] == 100

    @pytest.mark.parametrize("n", [10, 200, 500])
    def test_shape_customizado(self, n):
        assert gerar_dataset(n=n).shape[0] == n

    def test_colunas(self):
        df = gerar_dataset(n=50)
        assert COLUNAS_ESPERADAS == set(df.columns)

    def test_reproducibilidade(self):
        """Mesmo seed deve gerar o mesmo dataset."""
        np.random.seed(42)
        df1 = gerar_dataset(n=100)
        np.random.seed(42)
        df2 = gerar_dataset(n=100)
        pd.testing.assert_frame_equal(df1, df2)

    def test_churn_binario(self):
        df = gerar_dataset(n=500)
        assert set(df["churn"].unique()).issubset({0, 1})

    def test_churn_variado(self):
        df = gerar_dataset(n=500)
        assert df["churn"].sum() > 0
        assert (df["churn"] == 0).sum() > 0

    def test_nans_apenas_nas_colunas_esperadas(self):
        df = gerar_dataset(n=500)
        colunas_com_nan = set(df.columns[df.isnull().any()])
        assert colunas_com_nan.issubset({"renda_mensal", "satisfacao"})

    def test_satisfacao_range(self):
        df = gerar_dataset(n=500)
        sat = df["satisfacao"].dropna()
        assert sat.between(1, 5).all()

    def test_id_cliente_unico(self):
        df = gerar_dataset(n=100)
        assert df["id_cliente"].nunique() == 100

    def test_idade_range(self):
        df = gerar_dataset(n=500)
        assert df["idade"].between(18, 69).all()

    def test_num_produtos_range(self):
        df = gerar_dataset(n=500)
        assert df["num_produtos"].isin([1, 2, 3, 4]).all()

    def test_tem_credito_binario(self):
        df = gerar_dataset(n=500)
        assert df["tem_credito"].isin([0, 1]).all()

    def test_sexo_categorias(self):
        df = gerar_dataset(n=500)
        assert set(df["sexo"].unique()).issubset({"M", "F"})

    def test_canal_aquisicao_categorias(self):
        df = gerar_dataset(n=500)
        esperados = {"Loja Física", "Online", "Telefone", "Indicação"}
        assert set(df["canal_aquisicao"].unique()).issubset(esperados)

    def test_saldo_nao_negativo(self):
        df = gerar_dataset(n=500)
        assert (df["saldo"] >= 0).all()

    def test_num_reclamacoes_nao_negativo(self):
        df = gerar_dataset(n=500)
        assert (df["num_reclamacoes"] >= 0).all()
