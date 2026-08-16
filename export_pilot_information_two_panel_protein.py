#!/usr/bin/env python3
"""Export the two diagnostic panels for the PXD069517 metaproteomic
pilot-information convergence analysis (counterpart of Supplementary Figure S1,
which covers the metagenomic side).

Style (Arial, palette, box+points layout) matches
``export_pilot_information_two_panel.py``; this copy is parameterized so the
original script's behaviour is untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "validation_datasets" / "results" / "PXD069517_pilot_information"
PALETTE = ["#4b006e", "#35679a", "#ffbf00", "#0f8b7c", "#b45309"]


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


def box_and_points(ax, data: pd.DataFrame, column: str, pilots: list[int], seed: int) -> None:
    values = [data.loc[data["pilot_n"].eq(n), column].dropna().to_numpy() for n in pilots]
    boxes = ax.boxplot(values, positions=np.arange(len(pilots)), widths=0.54, patch_artist=True)
    for patch, n in zip(boxes["boxes"], pilots):
        patch.set_facecolor(PALETTE[pilots.index(n) % len(PALETTE)])
        patch.set_alpha(0.38)
    rng = np.random.default_rng(seed)
    for i, (n, vals) in enumerate(zip(pilots, values)):
        ax.scatter(
            i + rng.uniform(-0.10, 0.10, len(vals)),
            vals,
            s=30,
            color=PALETTE[pilots.index(n) % len(PALETTE)],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(pilots)), [str(n) for n in pilots])
    ax.set_xlabel("Pilot size per group")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DIR / "pilot_information_curve_metrics.csv")
    parser.add_argument("--out-png", type=Path, default=DEFAULT_DIR / "pilot_information_two_panel.png")
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_DIR / "pilot_information_two_panel.pdf")
    parser.add_argument("--pilots", default="5,7,10")
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()
    args.pilots = [int(x) for x in args.pilots.split(",")]
    return args


def main() -> None:
    args = parse_args()
    apply_style()
    data = pd.read_csv(args.data)
    pilots = args.pilots
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.55), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, wspace=0.10, hspace=0.02)

    box_and_points(axes[0], data, "mean_abs_curve_error", pilots, args.seed)
    axes[0].set_title("Curve-to-reference disagreement")
    axes[0].set_ylabel("Mean absolute power difference")

    box_and_points(axes[1], data, "omega2_at_80_power", pilots, args.seed)
    reference = float(data["reference_omega2_at_80_power"].dropna().iloc[0])
    axes[1].axhline(reference, color="#202020", lw=2.0, ls=":", label="Full-cohort reference")
    axes[1].set_title("Stability of the 80% power threshold")
    axes[1].set_ylabel("ω² required for 80% power")
    axes[1].legend(loc="upper right", frameon=False)

    for letter, ax in zip("ab", axes):
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=320, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(args.out_png)
    print(args.out_pdf)


if __name__ == "__main__":
    main()
