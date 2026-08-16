#!/usr/bin/env python3
"""Redraw manuscript Figure 5 as a 3 x 2 panel figure.

Rows follow the narrative order: study-size family, pilot consistency, and
pilot extrapolation. Columns are metagenomics followed by metaproteomics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import redraw_fig4_metagenomics_35 as gene_fig
import redraw_fig4_metaproteomics_35 as protein_fig

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "fig5_power_3x2_abcdef"
OUT_PNG = OUTDIR / "fig5_power_3x2_abcdef.png"
OUT_PDF = OUTDIR / "fig5_power_3x2_abcdef.pdf"


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.105,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=25,
        fontweight="bold",
        color="black",
    )


def main() -> None:
    gene_fig.apply_local_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 12.2,
            "legend.fontsize": 10.4,
        }
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    gene_df = pd.read_csv(gene_fig.DATA)
    gene_df = gene_df[gene_df["modality"].eq("gene")].copy()
    if gene_fig.PANEL_A_DENSE.exists():
        gene_panel_a = pd.read_csv(gene_fig.PANEL_A_DENSE)
    else:
        gene_panel_a = gene_df[
            (gene_df["panel"].eq("b"))
            & (gene_df["pilot"].eq(10))
            & (gene_df["eval_n"].isin(gene_fig.STUDY_SIZES))
        ].copy()
    if gene_fig.PANEL_A_4710_REFINED.exists():
        refined = pd.read_csv(gene_fig.PANEL_A_4710_REFINED)
        gene_panel_a = pd.concat(
            [gene_panel_a[~gene_panel_a["eval_n"].isin([4, 7, 10])], refined],
            ignore_index=True,
        )
    gene_panel_c = gene_df[
        (gene_df["panel"].eq("c"))
        & (gene_df["pilot"].isin(gene_fig.PILOTS))
        & (gene_df["eval_n"].eq(80))
    ].copy()
    gene_panel_c_keys = gene_fig.PILOTS
    gene_panel_c_colors = gene_fig.PILOT_COLORS
    if gene_fig.PANEL_C_EXTRA.exists():
        gene_panel_c = pd.concat([gene_panel_c, pd.read_csv(gene_fig.PANEL_C_EXTRA)], ignore_index=True)
        gene_panel_c_keys = gene_fig.PILOTS_EXTENDED
        gene_panel_c_colors = gene_fig.PILOT_EXTENDED_COLORS

    protein_df = pd.read_csv(protein_fig.DATA)

    fig, axes = plt.subplots(3, 2, figsize=(16.7, 9.35), squeeze=False)

    gene_fig.draw_points_panel(
        axes[0, 0],
        gene_panel_a[gene_panel_a["eval_n"].isin(gene_fig.STUDY_SIZES)],
        by="eval_n",
        keys=gene_fig.STUDY_SIZES,
        colors=gene_fig.SIZE_COLORS,
        title="Metagenomics study-size family (pilot n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_points_panel(
        axes[0, 1],
        protein_df[(protein_df["pilot_n"] == 17) & (protein_df["eval_n"].isin(protein_fig.PILOT_KEYS))],
        by="eval_n",
        keys=protein_fig.PILOT_KEYS,
        title="Metaproteomics study-size family (pilot n=17)",
        bin_width=0.004,
    )

    gene_fig.draw_band_panel(
        axes[1, 0],
        gene_df[(gene_df["panel"].eq("a")) & (gene_df["pilot"].isin(gene_fig.PILOTS)) & (gene_df["eval_n"].eq(10))],
        by="pilot",
        keys=gene_fig.PILOTS,
        colors=gene_fig.PILOT_COLORS,
        title="Metagenomics pilot consistency (eval n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[1, 1],
        protein_df[(protein_df["pilot_n"].isin([7, 10, 17])) & (protein_df["eval_n"] == 17)],
        by="pilot_n",
        keys=[7, 10, 17],
        title="Metaproteomics pilot consistency (eval n=17)",
        bin_width=0.004,
        seed_base=3500,
    )

    gene_fig.draw_band_panel(
        axes[2, 0],
        gene_panel_c[gene_panel_c["pilot"].isin(gene_fig.PILOTS_EXTENDED)],
        by="pilot",
        keys=gene_panel_c_keys,
        colors=gene_panel_c_colors,
        title="Metagenomics pilot extrapolation (eval n=80)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[2, 1],
        protein_df[(protein_df["pilot_n"].isin(protein_fig.PILOT_KEYS)) & (protein_df["eval_n"] == 80)],
        by="pilot_n",
        keys=protein_fig.PILOT_KEYS,
        title="Metaproteomics pilot extrapolation (eval n=80)",
        bin_width=0.003,
        seed_base=7600,
    )

    for letter, ax in zip("abcdef", axes.ravel()):
        add_panel_label(ax, letter)

    fig.tight_layout(rect=(0.018, 0.0, 0.998, 1.0), w_pad=2.4, h_pad=0.85)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
