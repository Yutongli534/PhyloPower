#!/usr/bin/env python3
"""Redraw the tree-error sample-size supplement in the Figure 5 layout."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import redraw_fig4_metagenomics_35 as gene_fig
import redraw_fig4_metaproteomics_35 as protein_fig

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "tree_error_fig5_like_quick"
FULL25_DIR = ROOT / "tree_error_fig5_like_full25"
OUTDIR = DATA_DIR / "figure"
OUT_PNG = OUTDIR / "tree_error_fig5_like_3x2.png"
OUT_PDF = OUTDIR / "tree_error_fig5_like_3x2.pdf"


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


def load_protein_extrapolation() -> pd.DataFrame:
    combined = DATA_DIR / "protein_pilot_extrapolation_refit" / "protein_power_curves_raw_combined.csv"
    if combined.exists():
        return pd.read_csv(combined)
    frames = []
    for pn in protein_fig.PILOT_KEYS:
        path = DATA_DIR / "protein_pilot_extrapolation_refit" / f"p{pn}" / "protein_power_curves_raw.csv"
        frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(combined, index=False)
    return out


def with_eval_n(df: pd.DataFrame, eval_n: int) -> pd.DataFrame:
    out = df.copy()
    if "eval_n" not in out.columns:
        out["eval_n"] = int(eval_n)
    return out


def append_fixed_midpoints(df: pd.DataFrame, modality: str, pilots: list[int]) -> pd.DataFrame:
    frames = [df.copy()]
    root = DATA_DIR / "fixed_midpoint_supplement" / modality
    for pn in pilots:
        path = root / f"p{pn}" / "fixed_midpoint_rows.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    if "pilot" in out.columns and "pilot_n" not in out.columns:
        out["pilot_n"] = out["pilot"]
    out = out.drop_duplicates(["pilot_n", "scale"], keep="last").sort_values(["pilot_n", "scale"])
    return out


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

    gene_study = pd.read_csv(DATA_DIR / "gene_study_size" / "gene_evalsweep_raw.csv")
    protein_study = pd.read_csv(DATA_DIR / "protein_study_size" / "protein_evalsweep_raw.csv")
    gene_consistency = with_eval_n(pd.read_csv(DATA_DIR / "gene_pilot_consistency" / "gene_power_curves_raw.csv"), 10)
    protein_consistency = with_eval_n(
        pd.read_csv(DATA_DIR / "protein_pilot_consistency" / "protein_power_curves_raw.csv"), 17
    )
    gene_full25 = FULL25_DIR / "gene_pilot_extrapolation" / "gene_power_curves_raw.csv"
    protein_full25 = FULL25_DIR / "protein_pilot_extrapolation" / "protein_power_curves_raw.csv"
    if gene_full25.exists():
        gene_extrapolation = with_eval_n(pd.read_csv(gene_full25), 80)
    else:
        gene_extrapolation = with_eval_n(pd.read_csv(DATA_DIR / "gene_pilot_extrapolation" / "gene_power_curves_raw.csv"), 80)
        gene_extrapolation = append_fixed_midpoints(gene_extrapolation, "gene", gene_fig.PILOTS_EXTENDED)
    if protein_full25.exists():
        protein_extrapolation = with_eval_n(pd.read_csv(protein_full25), 80)
    else:
        protein_extrapolation = with_eval_n(load_protein_extrapolation(), 80)
        protein_extrapolation = append_fixed_midpoints(protein_extrapolation, "protein", protein_fig.PILOT_KEYS)

    gene_consistency = gene_consistency.rename(columns={"pilot_n": "pilot"})
    gene_extrapolation = gene_extrapolation.rename(columns={"pilot_n": "pilot"})

    fig, axes = plt.subplots(3, 2, figsize=(16.7, 9.35), squeeze=False)

    gene_fig.draw_points_panel(
        axes[0, 0],
        gene_study[gene_study["eval_n"].isin(gene_fig.STUDY_SIZES)],
        by="eval_n",
        keys=gene_fig.STUDY_SIZES,
        colors=gene_fig.SIZE_COLORS,
        title="Metagenomics study-size family (pilot n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_points_panel(
        axes[0, 1],
        protein_study[protein_study["eval_n"].isin(protein_fig.PILOT_KEYS)],
        by="eval_n",
        keys=protein_fig.PILOT_KEYS,
        title="Metaproteomics study-size family (pilot n=17)",
        bin_width=0.004,
    )

    gene_fig.draw_band_panel(
        axes[1, 0],
        gene_consistency[(gene_consistency["pilot"].isin(gene_fig.PILOTS)) & (gene_consistency["eval_n"].eq(10))],
        by="pilot",
        keys=gene_fig.PILOTS,
        colors=gene_fig.PILOT_COLORS,
        title="Metagenomics pilot consistency (eval n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[1, 1],
        protein_consistency[
            (protein_consistency["pilot_n"].isin([7, 10, 17])) & (protein_consistency["eval_n"].eq(17))
        ],
        by="pilot_n",
        keys=[7, 10, 17],
        title="Metaproteomics pilot consistency (eval n=17)",
        bin_width=0.004,
        seed_base=3500,
    )

    gene_fig.draw_band_panel(
        axes[2, 0],
        gene_extrapolation[
            (gene_extrapolation["pilot"].isin(gene_fig.PILOTS_EXTENDED)) & (gene_extrapolation["eval_n"].eq(80))
        ],
        by="pilot",
        keys=gene_fig.PILOTS_EXTENDED,
        colors=gene_fig.PILOT_EXTENDED_COLORS,
        title="Metagenomics pilot extrapolation (eval n=80)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[2, 1],
        protein_extrapolation[
            (protein_extrapolation["pilot_n"].isin(protein_fig.PILOT_KEYS)) & (protein_extrapolation["eval_n"].eq(80))
        ],
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
