"""
test_avaliacao.py — Camada 1: testes das funções de geração de LaTeX em 05_avaliacao_modelo.py

Todas as funções testadas são puras (sem I/O em disco).
"""

import importlib.util
import os

import pandas as pd
import pytest

# ── Importação por caminho ─────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(filename: str):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aval = _load("05_avaliacao_modelo.py")

gerar_tabela_latex_modelos = aval.gerar_tabela_latex_modelos
gerar_tabela_latex_coeficientes = aval.gerar_tabela_latex_coeficientes
gerar_tabela_latex_confusao = aval.gerar_tabela_latex_confusao
gerar_tabela_dicionario_dados = aval.gerar_tabela_dicionario_dados


# ── Helpers ────────────────────────────────────────────────────────────────────


def _df_metricas() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Modelo": "Regressão Logística", "AUC-ROC": 0.85, "F1-Score": 0.72, "Acurácia": 0.80},
            {"Modelo": "Random Forest", "AUC-ROC": 0.90, "F1-Score": 0.76, "Acurácia": 0.83},
        ]
    )


def _latex_valido(latex: str) -> bool:
    """Verifica estrutura mínima de um ambiente table LaTeX."""
    return (
        r"\begin{table}" in latex
        and r"\end{table}" in latex
        and r"\begin{tabular}" in latex
        and r"\end{tabular}" in latex
        and r"\toprule" in latex
        and r"\bottomrule" in latex
    )


# ── gerar_tabela_latex_modelos ─────────────────────────────────────────────────


class TestGerartabelaLatexModelos:
    def test_retorna_string(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert isinstance(result, str)

    def test_estrutura_latex_valida(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert _latex_valido(result)

    def test_contem_label(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert r"\label{" in result

    def test_contem_caption(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert r"\caption{" in result

    def test_contem_nomes_de_modelos(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert "Regressão Logística" in result
        assert "Random Forest" in result

    def test_contem_fonte(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert r"\fonte{" in result

    def test_contem_metricas_formatadas(self):
        result = gerar_tabela_latex_modelos(_df_metricas())
        assert "0.8500" in result or "0.85" in result


# ── gerar_tabela_dicionario_dados ──────────────────────────────────────────────


class TestGerartabelaDicionarioDados:
    def test_retorna_string_nao_vazia(self):
        result = gerar_tabela_dicionario_dados()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_estrutura_latex_valida(self):
        result = gerar_tabela_dicionario_dados()
        assert _latex_valido(result)

    @pytest.mark.parametrize(
        "variavel",
        [
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
        ],
    )
    def test_contem_todas_as_variaveis(self, variavel):
        result = gerar_tabela_dicionario_dados()
        # LaTeX escapa underscores com \_
        assert variavel.replace("_", r"\_") in result, (
            f"Variável '{variavel}' não encontrada no dicionário de dados"
        )

    def test_label_presente(self):
        result = gerar_tabela_dicionario_dados()
        assert r"\label{tab:dicionario}" in result


# ── Integração leve: pipeline de geração de tabelas ───────────────────────────


class TestPipelineTabelas:
    """Verifica que todas as funções de tabela retornam LaTeX válido
    sem precisar de modelos treinados ou arquivos em disco."""

    def test_tabela_modelos_completa(self):
        df = _df_metricas()
        latex = gerar_tabela_latex_modelos(df)
        assert _latex_valido(latex)

    def test_tabela_dicionario_completa(self):
        latex = gerar_tabela_dicionario_dados()
        assert _latex_valido(latex)
