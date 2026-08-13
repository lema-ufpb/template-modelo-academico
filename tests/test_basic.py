"""
test_basic.py — Smoke tests de ambiente e dependências.
"""

import importlib
import sys

import pytest


def test_environment():
    """Pytest está configurado corretamente."""
    assert True


@pytest.mark.parametrize(
    "pacote",
    [
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "joblib",
        "openpyxl",
    ],
)
def test_dependencias_instaladas(pacote):
    """Todas as dependências do pyproject.toml devem ser importáveis."""
    try:
        importlib.import_module(pacote)
    except ImportError:
        pytest.fail(f"Dependência não instalada: {pacote}")


def test_python_version():
    """Python >= 3.9 (conforme pyproject.toml)."""
    assert sys.version_info >= (3, 9), f"Python {sys.version_info} < 3.9"
