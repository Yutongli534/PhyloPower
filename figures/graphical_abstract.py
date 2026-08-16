#!/usr/bin/env python3
"""Graphical abstract draft for NAR submission (5:2 landscape, Arial).

Message: PhyloPower turns a small empirical pilot into a prospective power
curve and a minimum per-group sample-size recommendation while preserving
phylogenetic and taxon-function structure.
Reading path: left to right. Code-drawn draft (not AI-generated imagery).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "output"

BLUE = "#35679a"
DARK = "#20344d"
AMBER = "#d99a00"
LIGHT = "#eaf1f8"
GREY = "#5a5a5a"


def style():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box(ax, x, y, w, h, ec=BLUE, fc=LIGHT, lw=2.0):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.02",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    )


def arrow(ax, x0, x1, y=0.5):
    ax.add_patch(
        FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=30,
                        linewidth=2.6, color=BLUE)
    )


def mini_table(ax, cx, cy, s=0.019, n=4, zero_frac=0.45, seed=3):
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j in range(n):
            filled = rng.random() > zero_frac
            ax.add_patch(plt.Rectangle((cx + j * s, cy - i * s), s * 0.88, s * 0.88,
                                       facecolor=BLUE if filled else "white",
                                       edgecolor="#9db6cf", linewidth=0.5))


def mini_tree(ax, cx, cy, w=0.055, h=0.13, color=DARK):
    # small cladogram rooted at left, drawn in axes fraction coords
    pts = [
        ((0.0, 0.5), (0.30, 0.5)),
        ((0.30, 0.20), (0.30, 0.80)),
        ((0.30, 0.80), (0.62, 0.97)),
        ((0.30, 0.80), (0.62, 0.63)),
        ((0.30, 0.20), (0.62, 0.37)),
        ((0.30, 0.20), (0.62, 0.03)),
        ((0.62, 0.97), (0.95, 0.97)),
        ((0.62, 0.63), (0.95, 0.63)),
        ((0.62, 0.37), (0.95, 0.37)),
        ((0.62, 0.03), (0.95, 0.03)),
    ]
    for (x0, y0), (x1, y1) in pts:
        ax.plot([cx + x0 * w, cx + x1 * w], [cy + (y0 - 0.5) * h, cy + (y1 - 0.5) * h],
                color=color, lw=1.5, solid_capstyle="round", clip_on=False)


def main() -> None:
    style()
    fig = plt.figure(figsize=(10.0, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    T_FS, S_FS = 13.5, 10.5

    # ---- stage 1: empirical pilot ----
    box(ax, 0.015, 0.28, 0.17, 0.46)
    ax.text(0.10, 0.655, "Empirical pilot", ha="center", fontsize=T_FS, fontweight="bold", color=DARK)
    ax.text(0.10, 0.575, "feature table +\nreference tree", ha="center", va="top", fontsize=S_FS, color=GREY)
    mini_table(ax, 0.045, 0.45, seed=3)
    mini_tree(ax, 0.115, 0.385)

    arrow(ax, 0.19, 0.235)

    # ---- stage 2: synthetic pools ----
    box(ax, 0.24, 0.28, 0.20, 0.46)
    ax.text(0.34, 0.655, "Synthetic pools", ha="center", fontsize=T_FS, fontweight="bold", color=DARK)
    ax.text(0.34, 0.575, "PCAM | MDC-TF-MC\nstructure preserved", ha="center", va="top", fontsize=S_FS, color=GREY)
    mini_table(ax, 0.256, 0.45, s=0.017, seed=5)
    mini_table(ax, 0.308, 0.45, s=0.017, seed=9)
    mini_table(ax, 0.360, 0.45, s=0.017, seed=12)

    arrow(ax, 0.445, 0.49)

    # ---- stage 3: effect modulation ----
    box(ax, 0.495, 0.28, 0.175, 0.46)
    ax.text(0.5825, 0.655, "Effect modulation", ha="center", fontsize=T_FS, fontweight="bold", color=DARK)
    ax.text(0.5825, 0.575, "realized ω² gradient", ha="center", va="top", fontsize=S_FS, color=GREY)
    ax.annotate("", xy=(0.515, 0.42), xytext=(0.560, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.6))
    ax.annotate("", xy=(0.615, 0.42), xytext=(0.560, 0.42),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.6))
    ax.text(0.535, 0.375, "dilution", ha="center", fontsize=9.5, color=GREY)
    ax.text(0.628, 0.375, "enhancement", ha="center", fontsize=9.5, color=GREY)

    arrow(ax, 0.675, 0.72)

    # ---- stage 4 (hero): power analysis -> minimum n ----
    box(ax, 0.725, 0.20, 0.265, 0.62, ec=AMBER, fc="white", lw=2.4)
    ax.text(0.8575, 0.735, "Power analysis", ha="center", fontsize=14.5, fontweight="bold", color=DARK)
    ax.text(0.8575, 0.660, "bootstrap PERMANOVA on\nrecomputed distances", ha="center", va="top",
            fontsize=S_FS, color=GREY)

    axin = ax.inset_axes([0.745, 0.26, 0.115, 0.24])
    x = np.linspace(0, 1, 200)
    for k, c, lw in ((7, BLUE, 2.2), (3.0, "#8fb3d4", 1.5)):
        axin.plot(x, 1 / (1 + np.exp(-k * (x - 0.45))), color=c, lw=lw)
    axin.axhline(0.8, color=GREY, lw=1.1, ls=":")
    axin.set_xticks([]); axin.set_yticks([])
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    axin.spines["left"].set_linewidth(0.9)
    axin.spines["bottom"].set_linewidth(0.9)
    axin.set_xlabel("realized ω²", fontsize=8.5)
    axin.set_ylabel("power", fontsize=8.5)

    ax.add_patch(FancyBboxPatch((0.878, 0.27), 0.10, 0.20, boxstyle="round,pad=0.006",
                                facecolor=AMBER, edgecolor="none", alpha=0.25))
    ax.text(0.928, 0.425, "minimum n", ha="center", fontsize=10.5, fontweight="bold", color=DARK)
    ax.text(0.928, 0.350, "per group", ha="center", fontsize=10.5, color=DARK)
    ax.text(0.928, 0.293, "n*", ha="center", fontsize=15, fontweight="bold", color=DARK)

    fig.savefig(OUT / "graphical_abstract.png", dpi=320)
    fig.savefig(OUT / "graphical_abstract.pdf")
    fig.savefig(OUT / "graphical_abstract.tiff", dpi=600)
    plt.close(fig)
    print(OUT / "graphical_abstract.png")


if __name__ == "__main__":
    main()
