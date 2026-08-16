#!/usr/bin/env python3
"""Export the two diagnostic panels used as Supplementary Figure S1."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "pilot_information_supplement" / "pilot_information_curve_metrics.csv"
OUT_PNG = ROOT / "pilot_information_supplement" / "pilot_information_two_panel.png"
OUT_PDF = ROOT / "pilot_information_supplement" / "pilot_information_two_panel.pdf"
PILOTS = [4, 7, 10]
COLORS = {4: "#4b006e", 7: "#35679a", 10: "#ffbf00"}


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
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
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


def box_and_points(ax, data: pd.DataFrame, column: str) -> None:
    values = [data.loc[data["pilot_n"].eq(n), column].dropna().to_numpy() for n in PILOTS]
    boxes = ax.boxplot(values, positions=np.arange(len(PILOTS)), widths=0.54, patch_artist=True)
    for patch, n in zip(boxes["boxes"], PILOTS):
        patch.set_facecolor(COLORS[n])
        patch.set_alpha(0.38)
    rng = np.random.default_rng(20260713)
    for i, (n, vals) in enumerate(zip(PILOTS, values)):
        ax.scatter(
            i + rng.uniform(-0.10, 0.10, len(vals)),
            vals,
            s=30,
            color=COLORS[n],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(PILOTS)), [str(n) for n in PILOTS])
    ax.set_xlabel("Pilot size per group")


def main() -> None:
    apply_style()
    data = pd.read_csv(DATA)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.55), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, wspace=0.10, hspace=0.02)

    box_and_points(axes[0], data, "mean_abs_curve_error")
    axes[0].set_title("Curve-to-reference disagreement")
    axes[0].set_ylabel("Mean absolute power difference")

    box_and_points(axes[1], data, "omega2_at_80_power")
    reference = float(data["reference_omega2_at_80_power"].dropna().iloc[0])
    axes[1].axhline(reference, color="#202020", lw=2.0, ls=":", label="Full-cohort reference")
    axes[1].set_title("Stability of the 80% power threshold")
    axes[1].set_ylabel("ω² required for 80% power")
    axes[1].legend(loc="upper right", frameon=False)

    for letter, ax in zip("ab", axes):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")

    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
