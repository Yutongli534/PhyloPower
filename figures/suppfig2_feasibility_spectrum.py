#!/usr/bin/env python3
"""Supplementary Figure S2 - feasibility spectrum and null-cohort safety.

Core conclusion: across a ~100-fold realized-effect spectrum, the minimum
per-group sample size for 80% power spans two orders of magnitude, and a
true-null cohort yields no finite recommendation (the framework refuses
rather than manufacturing significance).

Panel (a): baseline realized omega2 vs minimum per-group n for 80% power.
  Primary datasets: PhyloPower archived baseline-scenario runs (pools at
  pilot-scale signal). Independent cohorts: empirical full-cohort
  subsampling truth (500 without-replacement draws, PERMANOVA 999
  permutations, alpha = 0.05, Bray-Curtis).
Panel (b): PXD069517 (no genuine group difference, realized omega2 ~ -0.007)
  null p-value behaviour from 10 independent null pools x 100 relabelings.

Writes figures/output/suppfig2_feasibility_spectrum.{png,pdf,svg} and
suppfig2_source_data.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "output"
PXD_PVALS = ROOT / "validation_datasets" / "results" / "PXD069517_typeI_null" / "typeI_null_pvalues.csv"

# (label, baseline omega2, min n for 80% power, source)
# min n for primary datasets is an upper bound: power jumps 0 -> 1.0 between
# n = 2 and n = 10 on the archived grid, so the boundary lies below 10.
POINTS = [
    ("DPRS Cd vs Ni (metagenomic)", 0.422, 10, "PhyloPower archived run (upper bound)"),
    ("Pediatric IBD TI CD vs Control (metaproteomic)", 0.225, 10, "PhyloPower archived run (upper bound)"),
    ("QinJ_2012 T2D vs control", 0.009, 65, "full-cohort subsampling truth"),
    ("YachidaS_2019 CRC vs control", 0.002, 180, "full-cohort subsampling truth"),
]
NULL_POINT = ("PXD069517 CD_only vs PolyAI_CD (null cohort)", 0.007)


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


def main() -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.04, wspace=0.12)

    # ---- panel (a): feasibility spectrum ----
    ax = axes[0]
    src_rows = []
    xs = [p[1] for p in POINTS]
    ys = [p[2] for p in POINTS]
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, FuncFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())  # minor log labels render below the 5 pt floor
    ax.yaxis.set_minor_formatter(NullFormatter())
    # plain tick labels: default log labels use mathtext exponents below the 5 pt floor
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.scatter(xs, ys, s=42, color="#35679a", zorder=3, edgecolor="white", linewidth=0.5)
    short = {"DPRS Cd vs Ni (metagenomic)": "DPRS",
             "Pediatric IBD TI CD vs Control (metaproteomic)": "IBD (TI)",
             "QinJ_2012 T2D vs control": "QinJ_2012",
             "YachidaS_2019 CRC vs control": "YachidaS_2019"}
    offsets = {"DPRS": (1.25, 0.72), "IBD (TI)": (1.3, 1.35), "QinJ_2012": (1.15, 0.62), "YachidaS_2019": (1.2, 0.7)}
    for label, w2, n, source in POINTS:
        dx, dy = offsets[short[label]]
        ax.annotate(short[label], (w2, n), xytext=(w2 * dx, n * dy), fontsize=7)
        src_rows.append({"dataset": label, "baseline_omega2": w2, "min_n_power80": n, "source": source})
    # null cohort: correctly refused
    ax.scatter([NULL_POINT[1]], [300], marker="v", s=52, color="#b2182b", zorder=3, edgecolor="white", linewidth=0.5)
    ax.annotate("PXD069517 (null)\nno finite n:\nrecommendation refused", (NULL_POINT[1], 300),
                xytext=(0.012, 260), fontsize=7, color="#b2182b")
    src_rows.append({"dataset": NULL_POINT[0], "baseline_omega2": -0.007, "min_n_power80": "none (refused)",
                     "source": "PhyloPower run on true-null cohort"})
    ax.set_ylim(4, 400)
    ax.set_xlabel("Baseline realized ω² of the dataset (log scale)")
    ax.set_ylabel("Minimum n per group for 80% power (log scale)")
    ax.set_title("Feasibility boundary across the effect spectrum")

    # ---- panel (b): null-cohort safety ----
    ax = axes[1]
    pvals = pd.read_csv(PXD_PVALS)
    for eval_n, color in ((14, "#35679a"), (80, "#ffbf00")):
        sub = pvals.loc[pvals["eval_n"].eq(eval_n), "p_value"].to_numpy()
        sub = np.sort(sub)
        theo = (np.arange(1, len(sub) + 1) - 0.5) / len(sub)
        ax.scatter(theo, sub, s=4, alpha=0.35, color=color,
                   label=f"n = {eval_n} (rejection at α=0.05: {(sub < 0.05).mean():.3f})")
        src_rows.append({"dataset": f"PXD069517 null pools, n={eval_n}",
                         "baseline_omega2": "", "min_n_power80": "",
                         "source": f"{len(sub)} null p-values; rejection {(sub < 0.05).mean():.3f}"})
    ax.plot([0, 1], [0, 1], color="#202020", lw=1.0, ls=":", label="Uniform(0,1) expectation")
    ax.set_xlabel("Expected Uniform(0,1) quantile")
    ax.set_ylabel("Observed null P value quantile")
    ax.set_title("True-null cohort: no manufactured significance")
    ax.legend(loc="upper left", frameon=False)

    for letter, ax in zip("ab", axes):
        ax.text(-0.16, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")

    pd.DataFrame(src_rows).to_csv(OUT / "suppfig2_source_data.csv", index=False)
    fig.savefig(OUT / "suppfig2_feasibility_spectrum.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUT / "suppfig2_feasibility_spectrum.pdf", bbox_inches="tight")
    fig.savefig(OUT / "suppfig2_feasibility_spectrum.svg", bbox_inches="tight")
    plt.close(fig)
    print(OUT / "suppfig2_feasibility_spectrum.png")


if __name__ == "__main__":
    main()
