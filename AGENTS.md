# Regras e Diretrizes do Projeto (AGENTS.md)

Este documento define as regras, a arquitetura e as convenções que devem ser rigorosamente seguidas por qualquer agente de IA (e desenvolvedores) que interaja com este repositório.

## 1. Ferramentas e Dependências
- **Gerenciador de Pacotes e Ambiente:** O projeto utiliza exclusivamente o `uv`.
- **Fonte da Verdade:** Todas as dependências e configurações residem no `pyproject.toml`.
  - **NÃO** crie nem atualize o arquivo `requirements.txt`.
  - Para instalar/sincronizar dependências, utilize `uv sync` ou os comandos do `Makefile` (`make setup`, `make setup-dev`).
- **Qualidade de Código:** O projeto usa o `ruff` para formatação e linting.
  - Para checar e formatar, use `make lint` e `make format`.
  - Estilo: aspas duplas, line-length 100, indentação com 4 espaços (definido no `pyproject.toml`).
- **Type Checking:** Utilize type hints sistematicamente para permitir a validação estática.

## 2. Estrutura do Pipeline de Dados
O pipeline metodológico em Python é estritamente sequencial, dividido em scripts numerados dentro do diretório `scripts/`. Ele deve sempre poder ser executado de ponta a ponta sem intervenção humana através de `make pipeline`.

Fluxos de I/O padrão:
1. `01_coleta_dados.py` salva em `dados/brutos/`.
2. `02_analise_exploratoria.py` consome brutos e salva em `resultados/figuras/` e `resultados/metricas/`.
3. `03_preprocessamento.py` consome brutos e salva dados limpos (treino, teste) em `dados/processados/`.
4. `04_treinamento_modelo.py` consome treino e salva `.joblib` em `modelos/`.
5. `05_avaliacao_modelo.py` carrega modelos e teste, salva métricas em `resultados/`, tabelas `.tex` em `tex/<doc>/tabelas/` e figuras (`.png`).
6. `06_exportar_figuras_eps.py` converte automaticamente `.png` para `.eps` e as distribui para as pastas `tex/<doc>/figuras/`.

**Regra Crítica:** Nenhum script pode criar subdiretórios ou modificar a estrutura base, a não ser as saídas esperadas nas pastas de `dados/`, `modelos/`, `resultados/` e as exportações no diretório `tex/`.

## 3. Testes (`tests/`)
Os testes são implementados utilizando `pytest` e estruturados estritamente em **3 camadas** de integração:

1. **Camada 1 (Unitários Puros):** Testam a lógica do negócio, comportamentos limítrofes e funções puras (ex: sanitização, schemas padrão). Nenhum I/O de disco é permitido nesta camada (ver `test_coleta_dados.py`, `test_treinamento.py`, `test_avaliacao.py`).
2. **Camada 2 (Transformação em Memória):** Testam as rotinas matemáticas e o pré-processamento (ex: remoção de NaNs, codificação categórica). Usam *fixtures* mockadas contidas no `conftest.py`. Sem I/O de disco (ver `test_preprocessamento.py`).
3. **Camada 3 (Integração Leve):** Um único arquivo (`test_pipeline.py`) garante a integração End-to-End criando um diretório temporário (`tmp_path`) e executando o pipeline por completo para garantir a passagem de bastão correta dos artefatos.

Para rodar os testes, use `make test` ou `uv run pytest tests/ -v`.
- **Aviso de Matplotlib:** Ao testar funções que utilizam Matplotlib/Seaborn de forma não interativa, faça mocks apropriados da GUI (ex: `matplotlib.pyplot.Figure.savefig`) ou utilize o backend `Agg` para evitar travamentos durante as execuções de testes.

## 4. Integração com LaTeX
Este projeto une Ciência de Dados (Python) e documentação acadêmica automatizada (LaTeX).
- **Tabelas:** O script de avaliação (`05_avaliacao_modelo.py`) possui rotinas que convertem as saídas do Pandas para `strings` em LaTeX (ex: rotinas que utilizam `replace("_", r"\_")`). Nunca remova esses castings (para lidar com type hints como `Hashable`).
- **Compilação:** A compilação é gerenciada pelo utilitário `latexmk`. Os agentes devem instruir o usuário a compilar via `make latex DOC=<tipo>`.

## 5. Resumo de Comandos Relevantes (`Makefile`)
Antes de tentar rodar comandos complexos à mão, verifique se o Makefile não já providencia uma abstração.
- `make setup` / `make setup-dev`
- `make pipeline` (Roda do 01 ao 06)
- `make test`
- `make format` / `make lint`
- `make latex DOC=nome_do_tipo`
- `make new-doc TYPE=novo_tipo`
