#!/usr/bin/env python3
"""Redraw manuscript Figure 7 as a 3 x 2 panel figure.

Rows are baseline omega-squared, baseline power, and refitted power curves.
Columns are metagenomics/Gemelli and metaproteomics/PhyloFunc.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import figstyle  # noqa: E402
from logistic_fit import fit_logistic, logistic_curve  # noqa: E402

INDIR = ROOT / "data" / "archived_runs" / "fig5_rerun_20260701"
OUTDIR = ROOT / "figures" / "output"
OUT_PNG = OUTDIR / "fig7_tree_error_3x2_abcdef.png"
OUT_PDF = OUTDIR / "fig7_tree_error_3x2_abcdef.pdf"

CMAP_O2 = LinearSegmentedColormap.from_list("o2_light", ["#ffffff", figstyle.SYN])
CMAP_PW = LinearSegmentedColormap.from_list("pw_light", ["#ffffff", figstyle.GROUP[0]])

MODALITIES = [
    ("gene", "Metagenomics", "Gemelli", 0.60),
    ("protein", "Metaproteomics", "PhyloFunc", 0.18),
]


def apply_style() -> None:
    figstyle.apply_style()
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 15.0,
            "axes.titlesize": 17.5,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 13.8,
            "ytick.labelsize": 13.8,
            "legend.fontsize": 11.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def merged_heatmap(modality: str, key: str) -> np.ndarray:
    sigmas = np.linspace(0.0, 1.0, 20)
    nnis = np.linspace(0.0, 1.0, 20)
    out = np.full((len(nnis), len(sigmas)), np.nan)
    for path in sorted((INDIR / "_hm").glob(f"{modality}_*.npz")):
        arr = np.load(path)[key]
        out = np.where(np.isfinite(arr), arr, out)
    return out


def draw_heatmap(ax, data: np.ndarray, title: str, cmap, fig) -> None:
    sigmas = np.linspace(0.0, 1.0, data.shape[1])
    nnis = np.linspace(0.0, 1.0, data.shape[0])
    im = ax.imshow(
        np.ma.masked_invalid(data),
        cmap=cmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ticks = np.linspace(0, len(sigmas) - 1, 6, dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{sigmas[i]:.2f}" for i in ticks])
    ax.set_yticklabels([f"{nnis[i]:.2f}" for i in ticks])
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$p_{\mathrm{NNI}}$")
    ax.set_title(title, fontweight="bold", color="black", pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.76, pad=0.025)
    cb.ax.tick_params(labelsize=12.2)


def draw_curves(ax, curves: pd.DataFrame, modality: str, title: str, xmax: float) -> None:
    colors, _, _ = figstyle.seq_colors(sorted(curves["combo_index"].unique()))
    xg = np.linspace(0, xmax, 300)
    for i, combo in enumerate(sorted(curves["combo_index"].unique())):
        sub = curves[(curves["modality"].eq(modality)) & (curves["combo_index"].eq(combo))].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("true_omega2")
        sigma = float(sub["sigma"].iloc[0])
        nni = float(sub["nni"].iloc[0])
        color = colors[i % len(colors)]
        ax.scatter(sub["true_omega2"], sub["power"], s=13, color=color, alpha=0.34, zorder=2)
        fit = fit_logistic(sub[["true_omega2", "power"]].copy(), alpha=0.05)
        params = fit.get("params")
        if params:
            y = logistic_curve(xg, params["k"], params["x0"], 0.05)
            lw = 2.65 if sigma == 0.0 and nni == 0.0 else 1.75
            ax.plot(xg, y, color=color, lw=lw, label=rf"$\sigma$={sigma:.2f}, $p_{{NNI}}$={nni:.2f}")
        else:
            ax.plot(sub["true_omega2"], sub["power"], color=color, lw=1.6, label=rf"$\sigma$={sigma:.2f}, $p_{{NNI}}$={nni:.2f}")
    ax.axhline(0.8, color="#cc3333", ls="--", lw=1.2, alpha=0.72)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("ω²")
    ax.set_ylabel("Power")
    ax.set_title(title, fontweight="bold", color="black", pad=8)
    ax.grid(True, color="#e5e7eb", lw=0.65, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
        fontweight="bold",
        color="black",
    )


def main() -> None:
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(INDIR / "fig7_tree_perturbation_curves_rerun.csv")

    fig, axes = plt.subplots(3, 2, figsize=(16.7, 9.6), squeeze=False)
    for c, (modality, omics_label, metric_label, xmax) in enumerate(MODALITIES):
        draw_heatmap(
            axes[0, c],
            merged_heatmap(modality, "O"),
            f"{omics_label} baseline ω² ({metric_label})",
            CMAP_O2,
            fig,
        )
        draw_heatmap(
            axes[1, c],
            merged_heatmap(modality, "P"),
            f"{omics_label} baseline power ({metric_label})",
            CMAP_PW,
            fig,
        )
        draw_curves(
            axes[2, c],
            curves,
            modality,
            f"{omics_label} refitted power curves ({metric_label})",
            xmax,
        )

    for letter, ax in zip("abcdef", axes.ravel()):
        add_panel_label(ax, letter)

    fig.tight_layout(rect=(0.022, 0.0, 0.998, 1.0), w_pad=2.35, h_pad=0.9)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
