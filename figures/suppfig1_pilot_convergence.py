#!/usr/bin/env python3
"""Supplementary Figure S1 - pilot-information convergence in both data types.

Four panels: metagenomic (a,b) from data/pilot_information_supplement/ and
metaproteomic PXD069517 (c,d) from validation_datasets/results/
PXD069517_pilot_information/. Only plots archived metrics tables.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENE_DATA = ROOT / "data" / "pilot_information_supplement" / "pilot_information_curve_metrics.csv"
PROT_DATA = (
    ROOT / "validation_datasets" / "results" / "PXD069517_pilot_information" / "pilot_information_curve_metrics.csv"
)
OUT_PNG = ROOT / "figures" / "output" / "suppfig1_pilot_convergence.png"
OUT_PDF = ROOT / "figures" / "output" / "suppfig1_pilot_convergence.pdf"
PILOTS = {"gene": [4, 7, 10], "protein": [5, 7, 10]}
COLORS = {4: "#4b006e", 5: "#4b006e", 7: "#35679a", 10: "#ffbf00"}


def apply_style() -> None:
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
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box_and_points(ax, data: pd.DataFrame, column: str, pilots: list[int]) -> None:
    values = [data.loc[data["pilot_n"].eq(n), column].dropna().to_numpy() for n in pilots]
    boxes = ax.boxplot(values, positions=np.arange(len(pilots)), widths=0.54, patch_artist=True)
    for patch, n in zip(boxes["boxes"], pilots):
        patch.set_facecolor(COLORS[n])
        patch.set_alpha(0.38)
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.6)
    for key in ("whiskers", "caps", "medians"):
        for artist in boxes[key]:
            artist.set_color("#444444")
            artist.set_linewidth(0.7)
    rng = np.random.default_rng(20260713)
    for i, (n, vals) in enumerate(zip(pilots, values)):
        ax.scatter(
            i + rng.uniform(-0.10, 0.10, len(vals)),
            vals,
            s=8,
            color=COLORS[n],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(pilots)), [str(n) for n in pilots])
    ax.set_xlabel("Pilot size per group")


def main() -> None:
    apply_style()
    gene = pd.read_csv(GENE_DATA)
    prot = pd.read_csv(PROT_DATA)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06, wspace=0.08, hspace=0.10)

    rows = [("gene", gene, "Metagenomic (DPRS)"), ("protein", prot, "Metaproteomic (PXD069517)")]
    for r, (kind, data, label) in enumerate(rows):
        pilots = PILOTS[kind]
        box_and_points(axes[r, 0], data, "mean_abs_curve_error", pilots)
        axes[r, 0].set_title(f"{label}\nCurve-to-reference disagreement")
        axes[r, 0].set_ylabel("Mean absolute power difference")

        box_and_points(axes[r, 1], data, "omega2_at_80_power", pilots)
        reference = float(data["reference_omega2_at_80_power"].dropna().iloc[0])
        axes[r, 1].axhline(reference, color="#202020", lw=1.4, ls=":", label="Full-cohort reference")
        axes[r, 1].set_title("Stability of the 80% power threshold")
        axes[r, 1].set_ylabel("ω² required for 80% power")
        axes[r, 1].legend(loc="upper right", frameon=False)

    for letter, ax in zip("abcd", axes.flat):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(str(OUT_PDF.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
