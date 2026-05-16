from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


COLORBLIND_PALETTE = sns.color_palette("colorblind")
FIGURE_FACE_COLOR = "white"
AXES_FACE_COLOR = "#e6e6e6"
GRID_COLOR = "white"


def set_plot_style() -> None:
    """Apply the project-wide EDA chart style."""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=COLORBLIND_PALETTE,
        rc={
            "figure.facecolor": FIGURE_FACE_COLOR,
            "axes.facecolor": AXES_FACE_COLOR,
            "axes.edgecolor": "#bdbdbd",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.linewidth": 0.9,
            "axes.titlelocation": "left",
            "axes.titlesize": 0,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d9d9d9",
            "savefig.facecolor": FIGURE_FACE_COLOR,
            "savefig.edgecolor": FIGURE_FACE_COLOR,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
        },
    )


def clean_axis(ax: plt.Axes) -> plt.Axes:
    """Keep the grey plot area, white dashed grid, and no chart title."""
    ax.set_title("")
    ax.set_facecolor(AXES_FACE_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.9)

    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.8)

    return ax


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
