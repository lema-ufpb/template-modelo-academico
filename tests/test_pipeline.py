"""
test_pipeline.py — Camada 3: Teste de integração end-to-end.

Executa todo o pipeline principal com um dataset miniatura para
garantir que nenhum script quebra durante a execução ponta-a-ponta.
"""

import importlib.util
import os
from unittest import mock

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load(filename: str):
    path = os.path.join(_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


coleta = _load("01_coleta_dados.py")
prep = _load("03_preprocessamento.py")
treino = _load("04_treinamento_modelo.py")
aval = _load("05_avaliacao_modelo.py")


class TestPipelineCompleto:
    def test_pipeline_end_to_end(self, tmp_path, monkeypatch):
        """Roda todo o fluxo de dados em um diretório temporário."""

        # 1. Configurar diretórios temporários para o teste
        temp_dir = str(tmp_path)
        dir_brutos = os.path.join(temp_dir, "dados", "brutos")
        dir_proc = os.path.join(temp_dir, "dados", "processados")
        dir_mod = os.path.join(temp_dir, "modelos")
        dir_fig = os.path.join(temp_dir, "resultados", "figuras")
        dir_met = os.path.join(temp_dir, "resultados", "metricas")

        caminho_csv = os.path.join(dir_brutos, "dataset_clientes.csv")

        # Fazer mock dos caminhos em todos os scripts
        monkeypatch.setattr(coleta, "CAMINHO_SAIDA", caminho_csv)
        monkeypatch.setattr(prep, "CAMINHO_DADOS", caminho_csv)
        monkeypatch.setattr(prep, "DIR_SAIDA", dir_proc)
        monkeypatch.setattr(treino, "CAMINHO_TREINO", os.path.join(dir_proc, "dataset_treino.csv"))
        monkeypatch.setattr(treino, "DIR_MODELOS", dir_mod)
        monkeypatch.setattr(treino, "DIR_METRICAS", dir_met)
        monkeypatch.setattr(aval, "CAMINHO_TESTE", os.path.join(dir_proc, "dataset_teste.csv"))
        monkeypatch.setattr(aval, "CAMINHO_TREINO", os.path.join(dir_proc, "dataset_treino.csv"))
        monkeypatch.setattr(aval, "CAMINHO_BRUTO", caminho_csv)
        monkeypatch.setattr(aval, "DIR_MODELOS", dir_mod)
        monkeypatch.setattr(aval, "DIR_FIGURAS", dir_fig)
        monkeypatch.setattr(aval, "DIR_METRICAS", dir_met)
        monkeypatch.setattr(aval, "DOC_TYPES", [])  # Pula exportação LaTeX

        # Reduzir tamanho para rodar rápido
        monkeypatch.setattr(coleta, "N_CLIENTES", 100)
        monkeypatch.setattr(treino, "K_FOLDS", 2)

        # Fazer mock do LightGBM se estiver ausente, ou usar fallback

        # 2. Executar as funções main de cada script em sequência
        coleta.main()
        assert os.path.exists(caminho_csv)

        prep.main()
        assert os.path.exists(os.path.join(dir_proc, "dataset_treino.csv"))
        assert os.path.exists(os.path.join(dir_proc, "dataset_teste.csv"))

        treino.main()
        modelos_salvos = [f for f in os.listdir(dir_mod) if f.endswith(".joblib")]
        assert len(modelos_salvos) >= 2

        # Fazemos um mock no plt.savefig para evitar problemas de GUI no CI
        with mock.patch("matplotlib.pyplot.Figure.savefig") as mock_savefig:
            aval.main()
            assert mock_savefig.called

        assert os.path.exists(os.path.join(dir_met, "metricas_teste.csv"))
