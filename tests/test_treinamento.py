"""
test_treinamento.py — Camada 1: testes unitários de 04_treinamento_modelo.py

Foco em funções puras (sem I/O): sanitizar_nome e definir_modelos.
"""

import importlib.util
import os

import pytest

# ── Importação por caminho ─────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(filename: str):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


treino = _load("04_treinamento_modelo.py")
sanitizar_nome = treino.sanitizar_nome
definir_modelos = treino.definir_modelos


# ── sanitizar_nome ─────────────────────────────────────────────────────────────


class TestSanitizarNome:
    @pytest.mark.parametrize(
        "entrada, esperado",
        [
            ("Regressão Logística", "regressao_logistica"),
            ("Random Forest", "random_forest"),
            ("Gradient Boosting", "gradient_boosting"),
            ("LightGBM", "lightgbm"),
            ("Ação & Reação", "acao__reacao"),
            ("nome com  espaços", "nome_com__espacos"),
            ("MAIÚSCULAS", "maiusculas"),
            ("já-com-hifen", "jacomhifen"),  # hifens são removidos
        ],
    )
    def test_sanitizacao(self, entrada, esperado):
        assert sanitizar_nome(entrada) == esperado

    def test_resultado_e_minusculo(self):
        assert sanitizar_nome("QUALQUER COISA").islower()

    def test_sem_caracteres_especiais(self):
        result = sanitizar_nome("Olá mundo! @#$%")
        assert all(c.isalnum() or c == "_" for c in result)

    def test_string_vazia(self):
        assert sanitizar_nome("") == ""

    def test_apenas_letras(self):
        assert sanitizar_nome("simples") == "simples"

    def test_preserva_underscores(self):
        result = sanitizar_nome("já_tem_underscore")
        assert "_" in result


# ── definir_modelos ────────────────────────────────────────────────────────────


class TestDefinirModelos:
    def test_retorna_dict_nao_vazio(self):
        modelos = definir_modelos()
        assert isinstance(modelos, dict)
        assert len(modelos) > 0

    def test_nomes_sao_strings(self):
        modelos = definir_modelos()
        for nome in modelos:
            assert isinstance(nome, str)

    def test_modelos_tem_fit_e_predict(self):
        """Todos os modelos devem ser estimadores scikit-learn compatíveis."""
        modelos = definir_modelos()
        for nome, modelo in modelos.items():
            assert hasattr(modelo, "fit"), f"Modelo '{nome}' sem método fit()"
            assert hasattr(modelo, "predict"), f"Modelo '{nome}' sem método predict()"
            assert hasattr(modelo, "predict_proba"), f"Modelo '{nome}' sem predict_proba()"

    def test_modelos_conhecidos_presentes(self):
        """Regressão Logística e Random Forest são sempre esperados."""
        modelos = definir_modelos()
        nomes = list(modelos.keys())
        assert any("Logística" in n or "Logistica" in n for n in nomes), (
            "Regressão Logística não encontrada"
        )
        assert any("Random" in n for n in nomes), "Random Forest não encontrado"

    def test_modelos_sao_distintos(self):
        modelos = definir_modelos()
        objetos = list(modelos.values())
        # Não deve haver referências duplicadas
        assert len(objetos) == len(set(id(m) for m in objetos))
