#!/usr/bin/env python3
"""
06_exportar_figuras_eps.py — Exportação de Figuras em Formato EPS
=================================================================

Este script converte todas as figuras PNG geradas na pasta
resultados/figuras/ para o formato vetorial EPS, utilizado em
documentos LaTeX de alta qualidade tipográfica.

As figuras são salvas em cada diretório latex/<doc>/figuras/ para
que cada documento tenha suas próprias cópias.

Uso:
    python scripts/06_exportar_figuras_eps.py

Entradas:
    resultados/figuras/*.png

Saídas:
    latex/<doc>/figuras/*.eps                      (cada documento)

Como adaptar para seu projeto:
    Este script geralmente NÃO precisa de modificação. Ele detecta
    automaticamente todos os PNGs em `resultados/figuras/` e os
    converte para EPS.

    Se precisar ajustar:
    1. Para adicionar novos tipos de documento, adicione o nome
       à lista DOC_TYPES no topo do script.
    2. O diagrama do pipeline (`gerar_diagrama_pipeline()`) pode
       ser editado para refletir as etapas do seu projeto. Basta
       alterar a lista `etapas` com os nomes e cores desejados.
    3. Se suas figuras já forem geradas em formato vetorial (SVG,
       PDF), você pode pular a conversão e copiar diretamente.
"""

import glob
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.image import imread

# ── Configurações ──────────────────────────────────────────────
DIR_FIGURAS_PNG = os.path.join("resultados", "figuras")
DOC_TYPES = ["tcc_artigo", "tcc", "artigo", "dissertacao", "tese", "livro", "relatorio", "projeto"]


def _distribuir_para_docs(nome_arquivo: str, dir_origem: str):
    """Copia um arquivo EPS para cada diretório latex/<doc>/figuras/."""
    for doc in DOC_TYPES:
        dir_destino = os.path.join("tex", doc, "figuras")
        os.makedirs(dir_destino, exist_ok=True)
        shutil.copy2(
            os.path.join(dir_origem, nome_arquivo),
            os.path.join(dir_destino, nome_arquivo),
        )


def converter_png_para_eps(caminho_png: str, caminho_eps: str):
    """Converte uma imagem PNG para EPS usando Matplotlib.

    Parameters
    ----------
    caminho_png : str
        Caminho do arquivo PNG de entrada.
    caminho_eps : str
        Caminho do arquivo EPS de saída.
    """
    img = imread(caminho_png)
    fig, ax = plt.subplots(figsize=(img.shape[1] / 150, img.shape[0] / 150), dpi=150)
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(caminho_eps, format="eps", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def gerar_diagrama_pipeline():
    """Gera um diagrama visual do pipeline metodológico.

    Este diagrama é criado programaticamente para evitar
    dependências externas (como draw.io ou Inkscape).
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    etapas = [
        ("1. Coleta\nde Dados", "#3498db"),
        ("2. Análise\nExploratória", "#2ecc71"),
        ("3. Pré-\nProcessamento", "#f39c12"),
        ("4. Treinamento\nde Modelos", "#e74c3c"),
        ("5. Avaliação\ne Resultados", "#9b59b6"),
    ]

    for i, (texto, cor) in enumerate(etapas):
        x = i * 2.5
        rect = plt.Rectangle(
            (x, 0.5), 2, 2, linewidth=2, edgecolor=cor, facecolor=cor, alpha=0.2, joinstyle="round"
        )
        ax.add_patch(rect)
        ax.text(
            x + 1, 1.5, texto, ha="center", va="center", fontsize=11, fontweight="bold", color=cor
        )

        if i < len(etapas) - 1:
            ax.annotate(
                "",
                xy=(x + 2.5, 1.5),
                xytext=(x + 2, 1.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555"),
            )

    ax.set_xlim(-0.3, 12.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    ax.set_title("Pipeline Metodológico — Ciência de Dados", fontsize=14, fontweight="bold", pad=20)

    # Salvar PNG
    caminho_png = os.path.join(DIR_FIGURAS_PNG, "pipeline_metodologico.png")
    fig.savefig(caminho_png, format="png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Pipeline salvo em: {caminho_png}")

    # Salvar EPS e distribuir para cada diretório de documento
    caminho_eps = os.path.join(DIR_FIGURAS_PNG, "pipeline_metodologico.eps")
    fig.savefig(caminho_eps, format="eps", dpi=300, bbox_inches="tight")
    print(f"  ✓ Pipeline salvo em: {caminho_eps}")
    _distribuir_para_docs("pipeline_metodologico.eps", DIR_FIGURAS_PNG)

    plt.close(fig)


def main():
    """Ponto de entrada principal."""
    print("=" * 60)
    print("06 — Exportação de Figuras em Formato EPS")
    print("=" * 60)

    for doc in DOC_TYPES:
        os.makedirs(os.path.join("tex", doc, "figuras"), exist_ok=True)

    # Gerar diagrama de pipeline
    print("\nGerando diagrama do pipeline metodológico:")
    gerar_diagrama_pipeline()

    # Converter PNGs existentes para EPS
    pngs = glob.glob(os.path.join(DIR_FIGURAS_PNG, "*.png"))

    if not pngs:
        print("\n⚠ Nenhum PNG encontrado em resultados/figuras/")
        print("  Execute primeiro os scripts 02 a 05 para gerar as figuras.")
        return

    print(f"\nConvertendo {len(pngs)} figuras PNG → EPS:")
    for caminho_png in sorted(pngs):
        nome_base = os.path.splitext(os.path.basename(caminho_png))[0]
        caminho_eps = os.path.join(DIR_FIGURAS_PNG, f"{nome_base}.eps")
        converter_png_para_eps(caminho_png, caminho_eps)
        print(f"  ✓ {nome_base}.png → {nome_base}.eps")
        # Distribuir para cada diretório de documento
        _distribuir_para_docs(f"{nome_base}.eps", DIR_FIGURAS_PNG)

    print(f"\n✓ {len(pngs)} figuras exportadas para EPS com sucesso!")
    print("✓ Arquivos EPS disponíveis em cada latex/<doc>/figuras/")
    print("=" * 60)


if __name__ == "__main__":
    main()
