#!/usr/bin/env python3
"""Redraw the Type-I calibration figure as a 2 x 3 panel layout.

Panel order is designed so manuscript references can follow natural order:
  (a,b) QQ plots, (c,d) empirical-vs-nominal alpha, (e,f) Type-I vs sample size.

Within each pair, metagenomics is shown before metaproteomics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from scipy.stats import beta

import figstyle

ROOT = Path(__file__).resolve().parent
FIGDATA = ROOT / "figdata"

figstyle.apply_style()

for _font_path in [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
]:
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16.0,
    "axes.titlesize": 17.5,
    "axes.labelsize": 16.0,
    "xtick.labelsize": 14.2,
    "ytick.labelsize": 14.2,
    "legend.fontsize": 11.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

GENE = "#756bb1"
PROTEIN = "#159c76"
REAL = "#22313f"
ACCENT = "#cc3333"
NEUTRAL = "#6b7280"
GRID = "#e5e7eb"
BAND = "#e9ecef"


def _blend(color: str, other: str = "#ffffff", amount: float = 0.5) -> tuple[float, float, float]:
    a = np.asarray(to_rgb(color), dtype=float)
    b = np.asarray(to_rgb(other), dtype=float)
    return tuple((1.0 - amount) * a + amount * b)


def _eval_colors(eval_ns: list[int], base: str) -> dict[int, tuple[float, float, float]]:
    steps = np.linspace(0.66, 0.0, len(eval_ns))
    return {int(en): _blend(base, "#ffffff", float(step)) for en, step in zip(sorted(eval_ns), steps)}


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.tick_params(length=3.5, width=0.85, colors="#4b5563")
    ax.grid(color=GRID, lw=0.65, alpha=0.72)


def _panel_title(ax, letter: str, title: str, color: str) -> None:
    ax.text(-0.12, 1.018, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=30, fontweight="bold", color="black")
    ax.text(0.035, 1.025, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=17.5, fontweight="bold", color="black")


def _load_cache(cache: Path, eval_ns: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    df = pd.read_csv(cache)
    out = {}
    for modality in ["gene", "protein"]:
        dat = {}
        for en, sub in df[df["modality"].eq(modality)].groupby("eval_n"):
            en_i = int(en)
            if en_i in eval_ns:
                dat[en_i] = sub["pvalue"].to_numpy(float)
        out[modality] = dat
    return out["gene"], out["protein"]


def plot_qq(ax, pvals_by_n: dict[int, np.ndarray], qq_n: int, letter: str, title: str, color: str) -> None:
    pv = np.sort(pvals_by_n[qq_n])
    b = len(pv)
    expc = (np.arange(1, b + 1) - 0.5) / b
    lo = beta.ppf(0.025, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    hi = beta.ppf(0.975, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    ax.fill_between(expc, lo, hi, color=BAND, alpha=0.72, linewidth=0)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.15)
    ax.plot(expc, pv, ".", ms=2.4, color=color, alpha=0.78)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected null $P$")
    ax.set_ylabel("Observed null $P$")
    _panel_title(ax, letter, f"{title} QQ (n={qq_n})", color)
    _style_axis(ax)


def plot_alpha(ax, pvals_by_n: dict[int, np.ndarray], letter: str, title: str, color: str, show_legend: bool) -> None:
    eval_ns = sorted(pvals_by_n)
    color_of = _eval_colors(eval_ns, color)
    grid = np.linspace(0.001, 0.2, 120)
    pooled = []
    for en in eval_ns:
        vals = np.asarray([np.mean(pvals_by_n[en] < a) for a in grid], dtype=float)
        pooled.append(vals)
        ax.plot(grid, vals, color=color_of[en], lw=1.75, alpha=0.98, label=f"n={en}")
    if pooled:
        ax.plot(grid, np.mean(np.vstack(pooled), axis=0), color=REAL, lw=2.8, label="mean")
    ax.plot([0, 0.2], [0, 0.2], color=ACCENT, ls="--", lw=1.25)
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("Nominal $\\alpha$")
    ax.set_ylabel("Empirical Type I error")
    _panel_title(ax, letter, f"{title} alpha calibration", color)
    if show_legend:
        ax.legend(ncol=2, title="sample size", loc="upper left", frameon=False,
                  handlelength=2.15, columnspacing=1.15)
    _style_axis(ax)


def plot_size(ax, pvals_by_n: dict[int, np.ndarray], letter: str, title: str, color: str) -> dict[int, float]:
    eval_ns = sorted(pvals_by_n)
    xs = eval_ns
    ys = [float(np.mean(pvals_by_n[en] < 0.05)) for en in eval_ns]
    es = [1.96 * np.sqrt(y * (1 - y) / len(pvals_by_n[en])) for y, en in zip(ys, eval_ns)]
    ax.axhspan(0.025, 0.075, color=BAND, alpha=0.72, linewidth=0, zorder=0)
    ax.axhline(0.05, color=ACCENT, ls="--", lw=1.25, zorder=1)
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, ecolor=REAL, lw=2.05,
                ms=5.8, capsize=3.2, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Empirical Type I error at $\\alpha$=0.05")
    ax.set_ylim(0, 0.12)
    _panel_title(ax, letter, f"{title} sample-size check", color)
    ax.text(0.98, 0.69, "Bradley range", transform=ax.transAxes, ha="right", va="center",
            fontsize=12.0, color="#6b7280")
    _style_axis(ax)
    return {int(en): float(y) for en, y in zip(eval_ns, ys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=FIGDATA / "fig1_null_pvalues.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "fig2_typeI_2x3_gene_first")
    ap.add_argument("--eval-ns", default="6,10,17,30,50")
    ap.add_argument("--qq-n", type=int, default=30)
    args = ap.parse_args()

    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    gene, protein = _load_cache(args.cache, eval_ns)
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.9), squeeze=False)
    plot_qq(axes[0, 0], gene, args.qq_n, "a", "Metagenomics", GENE)
    plot_qq(axes[0, 1], protein, args.qq_n, "b", "Metaproteomics", PROTEIN)
    plot_alpha(axes[0, 2], gene, "c", "Metagenomics", GENE, show_legend=True)
    plot_alpha(axes[1, 0], protein, "d", "Metaproteomics", PROTEIN, show_legend=True)
    summary = {
        "metagenomics": plot_size(axes[1, 1], gene, "e", "Metagenomics", GENE),
        "metaproteomics": plot_size(axes[1, 2], protein, "f", "Metaproteomics", PROTEIN),
    }

    fig.suptitle("Type-I error calibration", y=0.995, fontsize=15.8, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.965), w_pad=2.65, h_pad=2.55)
    fig.savefig(args.out / "fig2_typeI_2x3_gene_first.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "fig2_typeI_2x3_gene_first.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.out / "fig2_typeI_2x3_gene_first_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(args.out / "fig2_typeI_2x3_gene_first.png")


if __name__ == "__main__":
    main()
