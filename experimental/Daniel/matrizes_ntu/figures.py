"""Export structural support, class mean matrices and their difference as PNGs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np

GRAY, TEAL = "#dce1e6", "#087f8c"


def canvas(columns=1):
    return plt.subplots(1, columns, figsize=(6.8 * columns, 6.3),
                        layout="constrained", squeeze=False)


def matrix_image(ax, values, **kwargs):
    image = ax.imshow(values, origin="upper", interpolation="nearest", **kwargs)
    ax.set(xlabel="NTU de destino", ylabel="NTU de origem",
           xticks=[0, 64, 128, 192, 255], yticks=[0, 64, 128, 192, 255])
    return image


def save(fig, output, name, title, note):
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.supxlabel(note, fontsize=10)
    path = Path(output) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return path


def plot_support(support, output):
    fig, axes = canvas()
    ax = axes[0, 0]
    matrix_image(ax, support, cmap=ListedColormap([GRAY, TEAL]), vmin=0, vmax=1)
    ax.set_title(f"{support.sum():,} transições possíveis de {support.size:,}")
    ax.legend(handles=[Patch(color=TEAL, label="Possível"),
                       Patch(color=GRAY, label="Impossível")], loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    return save(fig, output, "suporte_estrutural", "Suporte estrutural das transições NTU",
                "Suporte obtido enumerando todas as sequências de dez nucleotídeos.")


def plot_means(means, support, pair_count, output):
    fig, axes = canvas(2)
    values = means * 100
    positive = values[values > 0]
    low, high = positive.min(), positive.max()
    norm = LogNorm(low, max(high, low * 1.01))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    cmap.set_under("white")
    for ax, label, matrix in zip(axes.flat, ("mRNA", "ncRNA"), values):
        matrix = np.ma.array(np.where(matrix > 0, matrix, low / 10), mask=~support)
        image = matrix_image(ax, matrix, norm=norm, cmap=cmap)
        ax.set_title(label)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8,
                 label="Frequência média (%) · escala logarítmica")
    return save(fig, output, "matrizes_medias", "Matrizes médias de transições — Mus musculus",
                f"{pair_count:,} sequências por classe, pareadas por comprimento. Peso igual por sequência.\n"
                "Mesma escala nos dois painéis. Fundo branco: transição impossível ou não observada.")


def plot_difference(means, support, output):
    fig, axes = canvas(2)
    difference = (means[0] - means[1]) * 100
    limit = max(float(np.abs(difference).max()), np.finfo(float).eps)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(GRAY)
    image = matrix_image(axes[0, 0], np.ma.array(difference, mask=~support),
                         cmap=cmap, vmin=-limit, vmax=limit)
    axes[0, 0].set_title("Todas as transições")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.8, label="mRNA − ncRNA (pontos percentuais)")
    nodes = np.flatnonzero(support)
    top = nodes[np.argsort(np.abs(difference.ravel()[nodes]))[-10:][::-1]]
    values = difference.ravel()[top]
    labels = [f"{index // 256} → {index % 256}" for index in top]
    ax = axes[0, 1]
    bars = ax.barh(labels, values, color=np.where(values >= 0, "#b2182b", "#2166ac"))
    ax.bar_label(bars, fmt="%+.3f", padding=4)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.invert_yaxis()
    ax.set(title="10 maiores diferenças em módulo", ylabel="Origem → destino",
           xlabel="mRNA − ncRNA (pontos percentuais)", xlim=(-limit * 1.3, limit * 1.3))
    return save(fig, output, "diferenca_classes", "Diferença entre as matrizes médias — Mus musculus",
                "Vermelho: maior frequência em mRNA. Azul: maior frequência em ncRNA.\n"
                "Branco: diferença próxima de zero. Cinza: transição impossível.")
