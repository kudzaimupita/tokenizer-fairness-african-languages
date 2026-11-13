"""Static PNG figures (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Diverging: blue = fewer tokens than English, grey = parity, red = more tokens.
DIVERGING = LinearSegmentedColormap.from_list(
    "parity", ["#184f95", "#86b6ef", "#f0efec", "#ec835a", "#b02a2a"]
)


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)


def parity_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    """Rows: languages, columns: tokenizers, values: parity vs English."""
    values = matrix.to_numpy(dtype=float)
    logv = np.log2(values)
    bound = float(np.abs(logv).max())
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * matrix.shape[1], 0.9 + 0.42 * matrix.shape[0]))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(logv, cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -bound, bound), aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            colour = "white" if abs(logv[i, j]) > 0.62 * bound else INK
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color=colour)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.tick_params(length=0, colors=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def parity_bars(
    frame: pd.DataFrame,
    series: list[tuple[str, str]],
    languages: list[str],
    path: Path,
    title: str,
) -> None:
    """Grouped horizontal bars with 95% CI whiskers.

    `frame` has columns language, tokenizer, parity, parity_lo, parity_hi.
    `series` is a list of (tokenizer key, label) in legend order.
    """
    n = len(series)
    height = 0.8 / n
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(languages) + 1.3))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    y = np.arange(len(languages))
    for k, (key, label) in enumerate(series):
        sub = frame[frame["tokenizer"] == key].set_index("language").loc[languages]
        pos = y - 0.4 + height * (k + 0.5)
        ax.barh(
            pos,
            sub["parity"],
            height=height * 0.9,
            color=SERIES[k],
            label=label,
            edgecolor=SURFACE,
            linewidth=1,
        )
        err = np.vstack([sub["parity"] - sub["parity_lo"], sub["parity_hi"] - sub["parity"]])
        ax.errorbar(
            sub["parity"], pos, xerr=err, fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=1.5
        )
    ax.axvline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_yticks(y, languages)
    ax.set_ylim(len(languages) - 0.5, -0.9)
    ax.text(1.0, -0.62, " English = 1.0", color=INK_2, fontsize=8, va="center")
    ax.set_xlim(0, None)
    ax.set_xlabel(
        "Tokens relative to English on the same sentences (parity)", color=INK_2, fontsize=9
    )
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=n,
        labelcolor=INK_2,
        borderaxespad=0.2,
    )
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def coverage_scatter(
    frame: pd.DataFrame, groups: list[tuple[str, list[str]]], path: Path, title: str
) -> None:
    """Single-token word rate (x) against parity (y), one point per language x tokenizer."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for k, (label, langs) in enumerate(groups):
        sub = frame[frame["lang"].isin(langs)]
        ax.scatter(
            sub["single_token_rate"] * 100,
            sub["parity"],
            s=26,
            color=SERIES[k],
            edgecolor=SURFACE,
            linewidth=1,
            label=label,
        )
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK_2)
    ax.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xlabel("Word occurrences kept as one token (%)", color=INK_2, fontsize=9)
    ax.set_ylabel("Parity vs English", color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
