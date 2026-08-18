#!/usr/bin/env python3
"""Figure 1. Analytical workflow of the PhyloPower framework (high-resolution
redraw replacing the 220-dpi hand-drawn original).

Five stages per the manuscript caption:
  1. empirical data + reference tree (inputs)
  2. optional tree-error simulation (branch-length jitter + NNI)
  3. synthetic sample pools (PCAM / MDC-TF-MC) + effect modulation
  4. distance recomputation (Gemelli / PhyloFunc) + bootstrap PERMANOVA
  5. realized omega2 and power curves -> minimum per-group sample size
Output: figure1_workflow.{png(600dpi),pdf,tiff(600dpi)}
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
    for fp in ("/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def box(ax, x, y, w, h, ec=BLUE, fc=LIGHT, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def arrow(ax, x0, x1, y=0.5):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=1.8, color=BLUE))


def mini_table(ax, cx, cy, s=0.017, n=4, zero_frac=0.45, seed=3):
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j in range(n):
            filled = rng.random() > zero_frac
            ax.add_patch(plt.Rectangle((cx + j * s, cy - i * s), s * 0.88, s * 0.88,
                                       facecolor=BLUE if filled else "white",
                                       edgecolor="#9db6cf", linewidth=0.5))


def mini_tree(ax, cx, cy, w=0.05, h=0.12, color=DARK):
    pts = [((0.0, 0.5), (0.3, 0.5)), ((0.3, 0.2), (0.3, 0.8)),
           ((0.3, 0.8), (0.62, 0.97)), ((0.3, 0.8), (0.62, 0.63)),
           ((0.3, 0.2), (0.62, 0.37)), ((0.3, 0.2), (0.62, 0.03)),
           ((0.62, 0.97), (0.95, 0.97)), ((0.62, 0.63), (0.95, 0.63)),
           ((0.62, 0.37), (0.95, 0.37)), ((0.62, 0.03), (0.95, 0.03))]
    for (x0, y0), (x1, y1) in pts:
        ax.plot([cx + x0 * w, cx + x1 * w], [cy + (y0 - 0.5) * h, cy + (y1 - 0.5) * h],
                color=color, lw=1.3, solid_capstyle="round", clip_on=False)


def main() -> None:
    style()
    fig = plt.figure(figsize=(13.0, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    T, S = 10.5, 8.0

    # stage 1: inputs
    box(ax, 0.008, 0.22, 0.155, 0.60)
    ax.text(0.0855, 0.72, "1. Empirical inputs", ha="center", fontsize=T, fontweight="bold", color=DARK)
    ax.text(0.0855, 0.635, "abundance / taxon-function\ntable + group labels +\nreference tree", ha="center", va="top", fontsize=S, color=GREY)
    mini_table(ax, 0.028, 0.42, seed=3)
    mini_tree(ax, 0.105, 0.33)

    arrow(ax, 0.166, 0.196)

    # stage 2: tree perturbation (optional)
    box(ax, 0.199, 0.22, 0.165, 0.60)
    ax.text(0.2815, 0.72, "2. Tree perturbation\n(optional)", ha="center", va="top", fontsize=T, fontweight="bold", color=DARK)
    ax.text(0.2815, 0.575, "branch-length jitter +\nnearest-neighbor\ninterchange", ha="center", va="top", fontsize=S, color=GREY)
    mini_tree(ax, 0.237, 0.31, w=0.045)
    mini_tree(ax, 0.298, 0.31, w=0.045, color=AMBER)

    arrow(ax, 0.367, 0.397)

    # stage 3: pools + modulation
    box(ax, 0.400, 0.22, 0.20, 0.60)
    ax.text(0.50, 0.72, "3. Synthetic pools +\neffect modulation", ha="center", va="top", fontsize=T, fontweight="bold", color=DARK)
    ax.text(0.50, 0.575, "PCAM | MDC-TF-MC\nbidirectional realized ω²\ngradient", ha="center", va="top", fontsize=S, color=GREY)
    mini_table(ax, 0.422, 0.42, s=0.014, seed=5)
    mini_table(ax, 0.470, 0.42, s=0.014, seed=9)
    mini_table(ax, 0.518, 0.42, s=0.014, seed=12)
    ax.annotate("", xy=(0.478, 0.30), xytext=(0.446, 0.30),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))
    ax.annotate("", xy=(0.524, 0.30), xytext=(0.478, 0.30),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.8))

    arrow(ax, 0.603, 0.633)

    # stage 4: distance + PERMANOVA
    box(ax, 0.636, 0.22, 0.175, 0.60)
    ax.text(0.7235, 0.72, "4. Distance recompute +\nbootstrap PERMANOVA", ha="center", va="top", fontsize=T, fontweight="bold", color=DARK)
    ax.text(0.7235, 0.575, "Gemelli | PhyloFunc\nB resampling iterations,\nα = 0.05", ha="center", va="top", fontsize=S, color=GREY)
    axin = ax.inset_axes([0.660, 0.27, 0.115, 0.20])
    rng = np.random.default_rng(7)
    axin.scatter(rng.random(30), rng.random(30) * 0.9, s=4, color=BLUE, alpha=0.6)
    axin.set_xticks([]); axin.set_yticks([])
    for s_ in ("top", "right"):
        axin.spines[s_].set_visible(False)
    axin.spines["left"].set_linewidth(0.8); axin.spines["bottom"].set_linewidth(0.8)
    axin.set_xlabel("PC1", fontsize=7); axin.set_ylabel("PC2", fontsize=7)

    arrow(ax, 0.814, 0.844)

    # stage 5: power curves + recommendation
    box(ax, 0.847, 0.22, 0.147, 0.60, ec=AMBER, fc="white", lw=2.0)
    ax.text(0.9205, 0.72, "5. Power curves &\nminimum n", ha="center", va="top", fontsize=T, fontweight="bold", color=DARK)
    axin2 = ax.inset_axes([0.858, 0.30, 0.085, 0.26])
    x = np.linspace(0, 1, 200)
    for k, c, lw in ((7, BLUE, 1.8), (3.0, "#8fb3d4", 1.2)):
        axin2.plot(x, 1 / (1 + np.exp(-k * (x - 0.45))), color=c, lw=lw)
    axin2.axhline(0.8, color=GREY, lw=0.9, ls=":")
    axin2.set_xticks([]); axin2.set_yticks([])
    for s_ in ("top", "right"):
        axin2.spines[s_].set_visible(False)
    axin2.spines["left"].set_linewidth(0.8); axin2.spines["bottom"].set_linewidth(0.8)
    axin2.set_xlabel("realized ω²", fontsize=7)
    axin2.set_ylabel("power", fontsize=7)
    ax.text(0.963, 0.45, "n*", ha="center", fontsize=13, fontweight="bold", color=DARK)
    ax.text(0.963, 0.36, "per group", ha="center", fontsize=7.5, color=GREY)

    fig.savefig(OUT / "figure1_workflow.png", dpi=600)
    fig.savefig(OUT / "figure1_workflow.pdf")
    fig.savefig(OUT / "figure1_workflow.tiff", dpi=600)
    plt.close(fig)
    print(OUT / "figure1_workflow.png")


if __name__ == "__main__":
    main()
