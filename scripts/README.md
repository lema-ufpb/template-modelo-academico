# 🐍 Scripts do Pipeline de Ciência de Dados

> **Scripts Python numerados que formam o pipeline reprodutível do projeto. Cada script é independente, recebe entradas do anterior e gera saídas para o próximo.**

---

## 📊 Visão Geral

```
01_coleta_dados.py ──► 02_analise_exploratoria.py ──► 03_preprocessamento.py
                                                              │
                         06_exportar_figuras_eps.py ◄── 05_avaliacao_modelo.py ◄── 04_treinamento_modelo.py
```

---

## 📋 Tabela de Scripts

| # | Script | Entrada | Saída | Tempo* |
|---|--------|---------|-------|:------:|
| 01 | `01_coleta_dados.py` | — (gera dados sintéticos) | `dados/brutos/dataset_clientes.csv` | ~2s |
| 02 | `02_analise_exploratoria.py` | `dados/brutos/dataset_clientes.csv` | `resultados/figuras/*.png`, `resultados/metricas/estatisticas_descritivas.csv` | ~5s |
| 03 | `03_preprocessamento.py` | `dados/brutos/dataset_clientes.csv` | `dados/processados/dataset_{processado,treino,teste}.csv` | ~3s |
| 04 | `04_treinamento_modelo.py` | `dados/processados/dataset_treino.csv` | `modelos/*.joblib`, `resultados/metricas/validacao_cruzada.csv` | ~15s |
| 05 | `05_avaliacao_modelo.py` | `dados/processados/dataset_teste.csv`, `modelos/*.joblib` | `resultados/figuras/*.png`, `resultados/metricas/*.csv`, `latex/*/tabelas/*.tex` | ~10s |
| 06 | `06_exportar_figuras_eps.py` | `resultados/figuras/*.png` | `latex/*/figuras/*.eps` | ~20s |

\* Tempos aproximados em um computador moderno com o dataset de demonstração (1.000 registros).

---

## 🚀 Como Executar

### Pipeline completo (recomendado)
```bash
make pipeline
```

### Scripts individuais
```bash
uv run python scripts/01_coleta_dados.py
uv run python scripts/02_analise_exploratoria.py
# ... e assim por diante, na ordem
```

### Reexecutar apenas a partir de um ponto
Se você alterou apenas o treinamento, não precisa reexecutar a coleta e EDA:
```bash
uv run python scripts/04_treinamento_modelo.py
uv run python scripts/05_avaliacao_modelo.py
uv run python scripts/06_exportar_figuras_eps.py
```

---

## 🔧 O que Cada Script Faz (em detalhe)

### 01 — Coleta de Dados
Gera um dataset **fictício** de 1.000 clientes de telecomunicações usando NumPy. Os dados simulam relações realistas entre variáveis (ex: mais reclamações → menor satisfação → maior churn). Inclui valores ausentes intencionais (~5%) para demonstrar técnicas de imputação.

**Para adaptar:** Substitua o corpo de `gerar_dataset()` por código que carregue seu dataset real (leitura de CSV, chamada de API, web scraping, etc.).

### 02 — Análise Exploratória (EDA)
Gera estatísticas descritivas e 3 tipos de visualização: histogramas segmentados por churn, heatmap de correlação de Pearson e gráficos de barras de churn por categoria.

**Para adaptar:** Ajuste as listas de colunas (`colunas_num`, `colunas_cat`) para suas variáveis. Adicione novas visualizações conforme necessário.

### 03 — Pré-processamento
Aplica a pipeline de limpeza: imputação de missing values (mediana/moda), winsorização de outliers, feature engineering (renda por produto, taxa de reclamação, faixa etária, cliente premium), one-hot encoding e normalização com StandardScaler. Divide treino/teste com stratified split.

**Para adaptar:** Modifique as funções de tratamento para suas variáveis. A estratégia de imputação deve ser escolhida conforme a natureza de cada variável (mediana para assimétricas, média para normais, moda para categóricas).

### 04 — Treinamento de Modelos
Treina 3 classificadores (Regressão Logística, Random Forest, LightGBM) com validação cruzada estratificada (5-fold). Se o LightGBM não estiver instalado, usa GradientBoostingClassifier do sklearn como fallback. Salva os modelos treinados em formato `.joblib`.

**Para adaptar:** Adicione ou remova modelos no dicionário `modelos` da função `definir_modelos()`. Para ajustar hiperparâmetros, altere os parâmetros do construtor de cada classificador.

### 05 — Avaliação de Modelos
Calcula métricas no conjunto de teste (AUC-ROC, F1, Acurácia), gera curvas ROC, gráfico de importância de features, e tabelas LaTeX (comparação de modelos, coeficientes, matriz de confusão, estatísticas descritivas, dicionário de dados). As tabelas são salvas em **todos** os diretórios `latex/<doc>/tabelas/`.

**Para adaptar:** Ajuste as métricas conforme seu problema (ex: adicione Recall, Precisão). Para problemas multiclasse, a geração de tabelas LaTeX precisará ser adaptada.

### 06 — Exportação de Figuras EPS
Converte todas as figuras PNG em `resultados/figuras/` para o formato vetorial EPS (alta qualidade tipográfica) e as distribui para cada diretório `latex/<doc>/figuras/`. Também gera programaticamente o diagrama do pipeline metodológico.

**Para adaptar:** Geralmente não precisa de modificação. Se você adicionar novas figuras nos scripts anteriores, elas serão automaticamente convertidas e distribuídas.

---

## 🌱 Configurações Globais

Todos os scripts compartilham estas configurações para garantir reprodutibilidade:

| Configuração | Valor | Onde |
|-------------|-------|------|
| `SEED` | 42 | Topo de cada script |
| `PROPORCAO_TESTE` | 0.20 | Script 03 |
| `K_FOLDS` | 5 | Script 04 |

---

## 🐛 Dicas de Debug

- **Erro "FileNotFoundError":** Execute os scripts na ordem (01 → 06). Cada script depende da saída do anterior.
- **Erro "ModuleNotFoundError":** Execute `uv sync` para instalar dependências.
- **Aviso "LightGBM not found":** O script 04 usa GradientBoostingClassifier como fallback. Para instalar LightGBM: `uv sync --group ml`.
- **Figuras não aparecem no PDF:** Verifique se executou o script 06 e depois recompilou o LaTeX com `make latex`.
