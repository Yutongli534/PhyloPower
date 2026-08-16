#!/usr/bin/env python3
"""Redraw manuscript Figure 6 as a 3 x 2 panel figure.

Rows are effect settings; columns are metagenomics and metaproteomics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import psutil

    if psutil.cpu_count() is None:
        psutil.cpu_count = lambda logical=True: 4
except Exception:
    pass

import fig6

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "fig6_metric_3x2_abcdef"
OUT_PNG = OUTDIR / "fig6_metric_3x2_abcdef.png"
OUT_PDF = OUTDIR / "fig6_metric_3x2_abcdef.pdf"

MODALITIES = [("gene", "Metagenomics"), ("protein", "Metaproteomics")]


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.135,
        1.03,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=25,
        fontweight="bold",
        color="black",
    )


def draw_one(ax, modality: str, modality_label: str, scenario_title: str, scenario: str, m: int, seed: int, stats_rows: list[dict]) -> None:
    print(f"[fig6-3x2] start {modality_label} {scenario_title}", flush=True)
    d = fig6.P.load_modality(modality)
    table, group_map = fig6._scenario_table(d, modality, scenario, M=m, seed=seed)
    dms = fig6._metric_distances(d, modality, table)
    production_metric = "Gemelli" if modality == "gene" else "PhyloFunc"
    omega = max(0.0, float(fig6.core.compute_omega2(dms[production_metric], group_map)))
    title = f"{modality_label} {scenario_title}\nω²={omega:.3f}"
    fig6._draw_panel(ax, title, dms, group_map, stats_rows)
    ax.set_title(title, fontsize=15.0, fontweight="bold", color="black", pad=10)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    print(f"[fig6-3x2] done {modality_label} {scenario_title}", flush=True)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 13.2,
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.5,
        }
    )
    scenarios = [
        ("Original effect", "original"),
        ("Effect enhanced", "high"),
        ("Effect diluted", "low"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16.8, 9.35), squeeze=False)
    stats_rows: list[dict] = []
    base_seed = 20260616
    for r, (scenario_title, scenario) in enumerate(scenarios):
        for c, (modality, modality_label) in enumerate(MODALITIES):
            seed = base_seed + r * 1000 + (0 if modality == "gene" else 10000)
            draw_one(axes[r, c], modality, modality_label, scenario_title, scenario, 100, seed, stats_rows)

    for letter, ax in zip("abcdef", axes.ravel()):
        add_panel_label(ax, letter)

    handles = [
        fig6.mpatches.Patch(facecolor=fig6.C_WITHIN, alpha=0.7, label="Within-group"),
        fig6.mpatches.Patch(facecolor=fig6.C_BETWEEN, alpha=0.7, label="Between-group"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=12.5)
    fig.tight_layout(rect=(0.025, 0.0, 0.998, 0.968), w_pad=2.25, h_pad=0.95)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=260)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(OUTDIR / "fig6_metric_3x2_stats.csv", index=False)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
