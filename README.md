# Repositório-Modelo para Trabalhos Acadêmicos

> **Template multi-propósito para trabalhos acadêmicos da UFPB/CCSA: aplicável a áreas como Ciência de Dados, Economia e afins.**

Este repositório serve como modelo de referência para estudantes do **Centro de Ciências Sociais Aplicadas (CCSA)** da **Universidade Federal da Paraíba (UFPB)** que precisam organizar seus projetos de pesquisa (TCC, dissertação, tese, etc.). O foco é garantir **reprodutibilidade**, **boas práticas** e **documentação**, permitindo que a mesma pipeline de dados alimente diferentes formatos de documento acadêmico.

---

## O que é este projeto?

Fazer ciência não é apenas apresentar números ou publicar um documento. Um trabalho científico se baseia em princípios fundamentais de **transparência, ética, honestidade intelectual e curiosidade**.

Para que uma pesquisa tenha validade e seja útil para a comunidade (colegas, professores e outros cientistas), ela precisa permitir que os resultados sejam verificados de forma independente.

Este repositório fornece uma **arquitetura de projeto pronta para uso** que:
1. **Padroniza** a organização de dados, scripts e documentos.
2. **Automatiza** a geração de figuras e tabelas diretamente para o texto.
3. **Oferece** 8 templates LaTeX diferentes (TCC, Artigo, Dissertação, Tese, Livro, Relatório e Projeto), já configurados nas normas ABNT.
4. **Ensina** por meio de dicas pedagógicas inseridas dentro dos próprios arquivos LaTeX (`\begin{dica}`).

---

## Tipos de Documento Suportados

| Tipo | Descrição | Classe LaTeX | Estrutura |
|------|-----------|-------------|-----------|
| `tcc_artigo` | TCC em formato de artigo técnico-científico | `abntex2[article]` | Seções contínuas |
| `tcc` | TCC padrão / monografia com capítulos | `abntex2` | Capítulos |
| `artigo` | Artigo científico avulso | `abntex2[article]` | Seções contínuas |
| `dissertacao` | Dissertação de mestrado | `abntex2` | Capítulos |
| `tese` | Tese de doutorado | `abntex2` | Capítulos |
| `livro` | Livro didático / texto de apoio | `abntex2` | Capítulos |
| `relatorio` | Relatório técnico-científico | `abntex2[article]` | Seções contínuas |
| `projeto` | Projeto de pesquisa | `abntex2[article]` | Seções contínuas |

---

## Artefatos e Planejamento (`docs/`)

Este repositório também inclui modelos markdown em `docs/` para auxiliar na gestão do projeto de pesquisa no dia a dia. Alguns dos mais importantes são:

- [**Guia de Escrita Científica**](docs/guia_escrita_cientifica.md): Um guia abrangente sobre escrita acadêmica, citação e boas práticas.
- [**Diário de Pesquisa**](docs/diario_pesquisa.md): Template para registro de atividades, decisões tomadas e obstáculos diários.
- [**Anotação de Experimento**](docs/anotacao_experimento.md): Padrão para registrar hipóteses, parâmetros e resultados de cada experimento de Machine Learning executado.
- [**Reunião com Orientador(a)**](docs/reuniao_orientacao.md): Pauta estruturada para aproveitar ao máximo o tempo das reuniões de orientação.
- [**Planejamento de Sprint**](docs/planejamento_sprint.md): Gestão ágil (Scrum/Kanban) adaptada para pesquisa acadêmica.
- [**Checklist de Defesa**](docs/checklist_defesa.md): O que preparar e revisar antes da apresentação final para a banca.

---

## Estrutura do Repositório

```text
projeto-tcc/
├── README.md                          ← Você está aqui
├── docs/                              ← Templates para anotações e planejamento
├── pyproject.toml                     ← Configuração de dependências Python (uv)
├── Makefile                           ← Automação de tarefas
│
├── latex/                               ← Arquivos LaTeX (8 formatos)
│   ├── tcc_artigo/                    ← Exemplo: TCC formato artigo
│   │   ├── main.tex                   ← Texto com dicas pedagógicas
│   │   ├── referencias.bib
│   │   ├── figuras/                   ← Figuras geradas pela pipeline
│   │   └── tabelas/                   ← Tabelas geradas pela pipeline
│   └── ... (outros 7 formatos)
│
├── dados/                             ← Bases de dados (leia o README lá)
│   ├── brutos/                        ← Dados originais (imutáveis)
│   └── processados/                   ← Dados limpos
│
└── scripts/                           ← Pipeline em Python (01 a 06)
    ├── 01_coleta_dados.py
    └── ... (até 06)
```

Cada formato na pasta `latex/` é **completamente independente e autocontido**. Se você quer fazer uma dissertação, ignore/apague os outros 7 diretórios e trabalhe apenas dentro de `latex/dissertacao/`.

---

## Como Usar (Guia Passo a Passo)

### 1. Requisitos do Sistema
- **Make**: (Já vem no Mac/Linux. No Windows, use WSL ou Git Bash).
- **Python 3.9+**: Usamos o `uv` para instalar dependências.
- **LaTeX**: Necessário para gerar os PDFs.
  - Mac: `brew install --cask mactex`
  - Linux: `sudo apt install texlive-full`
  - Windows: Baixe o [MiKTeX](https://miktex.org/download)

### 2. Configurar o Ambiente Python
Nós recomendamos fortemente o gerenciador de pacotes **uv** por ser ultrarrápido.
Se não tiver, instale-o (`curl -LsSf https://astral.sh/uv/install.sh | sh` no Mac/Linux).

Após clonar este repositório, rode:
```bash
uv sync
```
*Isto instalará Pandas, Scikit-Learn, LightGBM, Matplotlib, etc.*

### 3. Executar a Pipeline de Dados
A pipeline inteira está automatizada. Ela vai do script `01` ao `06`, treinando modelos e exportando resultados.

```bash
make pipeline
```
*Isso gerará os dados na pasta `dados/`, modelos em `modelos/`, e enviará automaticamente as figuras (.eps) e tabelas (.tex) para **todos** os 8 diretórios dentro de `latex/`.*

### 4. Compilar o PDF (LaTeX)
Para gerar o PDF do seu texto, use o comando:

```bash
make latex DOC=tcc_artigo  # Troque tcc_artigo pelo formato que desejar
```
O PDF gerado estará dentro da pasta `latex/tcc_artigo/`.

---

## Como adaptar este repositório para o SEU projeto?

1. **Escolha seu modelo**: Defina qual será o formato (ex: `dissertacao`).
2. **Delete o que não for usar**: Apague as outras pastas de dentro do `latex/`.
3. **Coloque seus dados**: Substitua os dados em `dados/brutos/` pelo seu dataset.
4. **Altere os scripts**: Edite do `01` ao `06` em `scripts/` para aplicar *sua* análise e *seus* modelos.
5. **Escreva seu texto**: Edite o `latex/sua_pasta/main.tex`, preencha os metadados (nome, título) e escreva. *Aproveite as dicas no ambiente `\begin{dica}`.*
6. **Recompile**: Rode `make pipeline` e `make latex DOC=sua_pasta`.

---

## Comandos Úteis do Makefile

| Comando | Descrição |
|---------|-----------|
| `make pipeline` | Executa todos os scripts Python (01 ao 06) sequencialmente. |
| `make latex DOC=x`| Compila o documento `x` (ex: `DOC=artigo`). |
| `make clean` | Apaga caches temporários, \_\_pycache\_\_, modelos. |
| `make latex-clean`| Apaga os arquivos auxiliares pesados do LaTeX. |

---

Este projeto nasceu para promover a Ciência Aberta e a Reprodutibilidade na pesquisa acadêmica brasileira.
