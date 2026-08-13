# Changelog

Todas as mudanças relevantes deste repositório-modelo são documentadas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).

---

## [2.1.0] — 2026-06-07

### Adicionado
- **Diretório `docs/`** com 10 templates Markdown para organização da pesquisa:
  - Diário de pesquisa, anotação de experimento, brainstorm, planejamento semanal
  - Fichamento de artigo/livro, ata de reunião de orientação
  - Checklist de defesa/entrega, guia de escrita científica, glossário de termos
- **READMEs explicativos** em `dados/`, `dados/brutos/` e `dados/processados/`
- **README detalhado** em `scripts/` com tabela de E/S, diagramas e dicas de adaptação
- **CHANGELOG.md** (este arquivo)
- Seções "Como adaptar" nas docstrings de todos os 6 scripts Python

### Melhorado
- **Dicas pedagógicas expandidas** em todos os 8 templates LaTeX (`latex/*/main.tex`):
  - Orientações detalhadas (8-15 linhas) por seção/capítulo
  - Perguntas-guia, exemplos de boas e más práticas, erros comuns
  - Cobertura: Resumo, Introdução, Fundamentação, Metodologia, Resultados, Conclusão
- **README.md principal** reescrito como guia autocontido:
  - Seção "Início Rápido" com 5 comandos
  - Instruções de instalação de LaTeX por SO
  - FAQ / Troubleshooting
  - Diagrama visual do fluxo de trabalho

---

## [2.0.0] — 2026-05-XX

### Adicionado
- Arquitetura multi-documento: 8 tipos de documento LaTeX (tcc_artigo, tcc, artigo, dissertação, tese, livro, relatório, projeto)
- Cada documento é autocontido com figuras, tabelas e referências próprias
- Comandos `make latex-all`, `make list-docs`, `make new-doc TYPE=x`
- Ambiente `{dica}` para orientações pedagógicas nos templates LaTeX
- Fallback automático sklearn quando LightGBM não está disponível

### Melhorado
- Pipeline gera figuras e tabelas diretamente em cada `latex/<doc>/`
- Makefile com cores ANSI e menu de ajuda estilizado

---

## [1.0.0] — 2026-04-XX

### Adicionado
- Estrutura inicial do repositório
- Pipeline de 6 scripts Python (coleta → exportação EPS)
- Template LaTeX com classe abnTeX2
- Makefile com automação de pipeline e compilação
- pyproject.toml com gerenciamento via uv
- CONTRIBUTING.md com diretrizes de contribuição
- .gitignore abrangente para Python, LaTeX e dados
