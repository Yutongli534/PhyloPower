#!/usr/bin/env python3
"""Redraw parameter-to-realized-omega validation with unified figure style."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "parameter_omega_dense_current_method_v4.csv"
OUT_PNG = ROOT / "fig_parameter_realized_omega_validation.png"
OUT_PDF = ROOT / "fig_parameter_realized_omega_validation.pdf"

COLORS = {
    "gene": {
        "point": "#7570b3",
        "line": "#5e4fa2",
        "band": "#dad7f0",
    },
    "protein": {
        "point": "#1b9e77",
        "line": "#0f766e",
        "band": "#d9f0e7",
    },
}


def apply_style() -> None:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.linewidth": 0.95,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.62,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def smooth_with_band(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    *,
    frac: float,
    seed: int,
    n_boot: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]

    sm = lowess(y, x, frac=frac, it=1, return_sorted=True)
    center = np.interp(grid, sm[:, 0], sm[:, 1])

    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        b_order = np.argsort(xb)
        xb = xb[b_order]
        yb = yb[b_order]
        grouped = pd.DataFrame({"x": xb, "y": yb}).groupby("x", as_index=False)["y"].mean()
        if len(grouped) < 4:
            continue
        try:
            sb = lowess(grouped["y"], grouped["x"], frac=frac, it=1, return_sorted=True)
        except Exception:
            continue
        if len(sb) < 4:
            continue
        curves.append(np.interp(grid, sb[:, 0], sb[:, 1], left=sb[0, 1], right=sb[-1, 1]))

    if len(curves) >= 40:
        stack = np.vstack(curves)
        lo = np.nanpercentile(stack, 10, axis=0)
        hi = np.nanpercentile(stack, 90, axis=0)
    else:
        lo = center
        hi = center
    return center, lo, hi


def draw_panel(
    ax,
    sub: pd.DataFrame,
    *,
    modality: str,
    title: str,
    frac: float,
    seed: int,
    show_ylabel: bool,
) -> None:
    colors = COLORS[modality]
    x = sub["effect_level"].to_numpy(float)
    y = sub["omega2"].to_numpy(float)
    xpad = 0.03 * (float(np.nanmax(x)) - float(np.nanmin(x)))
    grid = np.linspace(float(np.nanmin(x)) - xpad, float(np.nanmax(x)) + xpad, 500)
    center, lo, hi = smooth_with_band(x, y, grid, frac=frac, seed=seed)

    ax.fill_between(grid, lo, hi, color=colors["band"], alpha=0.55, lw=0, zorder=1)
    ax.plot(grid, center, color=colors["line"], lw=2.7, zorder=3)
    ax.scatter(x, y, s=36, color=colors["point"], alpha=0.72, edgecolor="white", linewidth=0.45, zorder=4)

    observed = sub.iloc[(sub["effect_level"].abs()).argsort().iloc[0]]
    ax.axvline(0, color="#7a8798", ls=":", lw=1.35, zorder=0)
    ax.scatter(
        [observed["effect_level"]],
        [observed["omega2"]],
        marker="D",
        s=96,
        color=colors["line"],
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xlabel("Effect level")
    ax.set_ylabel("true ω²" if show_ylabel else "")
    ax.set_xlim(grid[0], grid[-1])
    ymax = float(np.nanmax([np.nanmax(y), np.nanmax(hi)]))
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(True)
    ax.tick_params(axis="both", length=4, width=0.9, color="#444444")


def main() -> None:
    apply_style()
    df = pd.read_csv(DATA)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.65), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.045, h_pad=0.035, wspace=0.08, hspace=0.02)

    draw_panel(
        axes[0],
        df[df["modality"].eq("gene")],
        modality="gene",
        title="(a) Metagenomic PCAM",
        frac=0.24,
        seed=4100,
        show_ylabel=True,
    )
    draw_panel(
        axes[1],
        df[df["modality"].eq("protein")],
        modality="protein",
        title="(b) Metaproteomic MDC-TF-MC",
        frac=0.28,
        seed=8200,
        show_ylabel=False,
    )

    protein_colors = COLORS["protein"]
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=protein_colors["point"],
               markeredgecolor="white", markersize=7.5, label="Synthetic pool"),
        Line2D([0], [0], color=protein_colors["line"], lw=2.7, label="Smoothed trend"),
        Patch(facecolor=protein_colors["band"], edgecolor="none", alpha=0.55, label="Local trend band"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor=protein_colors["line"],
               markeredgewidth=1.0, markersize=8, linestyle="None", label="Observed setting"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right", handlelength=2.8)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
