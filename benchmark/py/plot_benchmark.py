#!/usr/bin/env python3
"""Combine the three tools' power curves, compute the minimum n per group for
80% power, and render the two-panel supplementary figure (Bray-Curtis /
weighted UniFrac; one curve per tool x effect tier; 0.8 reference line).

Inputs:  benchmark/results/power_{phylopower,micropower,mpress}.csv
Outputs: benchmark/results/power_all_tools.csv
         benchmark/results/min_n_power80.csv
         benchmark/figures/fig_benchmark_power.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmark" / "results"
FIGURES = REPO / "benchmark" / "figures"

TOOL_COLORS = {"PhyloPower": "#0072B2", "micropower": "#D55E00", "MPrESS": "#009E73"}
TOOL_ORDER = ["PhyloPower", "micropower", "MPrESS"]
METRIC_LABELS = {"braycurtis": "Bray-Curtis", "wunifrac": "Weighted UniFrac"}


def min_n_for_power(df: pd.DataFrame, target: float = 0.8) -> float:
    """Smallest n with power >= target (NaN if never reached)."""
    df = df.sort_values("n_per_group")
    hit = df[df["power"] >= target]
    return float(hit["n_per_group"].iloc[0]) if len(hit) else np.nan


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = []
    for tool, fname in [
        ("PhyloPower", "power_phylopower.csv"),
        ("micropower", "power_micropower.csv"),
        ("MPrESS", "power_mpress.csv"),
    ]:
        path = RESULTS / fname
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"WARNING: missing {path}", file=sys.stderr)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(RESULTS / "power_all_tools.csv", index=False)

    min_rows = []
    for (tool, metric, scale), sub in all_df.groupby(["tool", "metric", "between_scale"]):
        min_rows.append(
            {
                "tool": tool,
                "metric": metric,
                "between_scale": scale,
                "omega2": sub["omega2"].iloc[0],
                "min_n_power80": min_n_for_power(sub),
                "max_power": sub["power"].max(),
                "n_grid_max": sub["n_per_group"].max(),
            }
        )
    min_df = pd.DataFrame(min_rows).sort_values(["metric", "between_scale", "tool"])
    min_df.to_csv(RESULTS / "min_n_power80.csv", index=False)
    print(min_df.to_string(index=False))

    # ---- figure ----
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        plt.rcParams["font.family"] = "Arial"
    except ValueError:
        print("WARNING: Arial not found, falling back to default font", file=sys.stderr)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    tiers = sorted(all_df["between_scale"].unique())
    # shade intensity increases with realized omega2 (per metric, normalized)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, metric in zip(axes, ["braycurtis", "wunifrac"]):
        sub_m = all_df[all_df["metric"] == metric]
        omin, omax = sub_m["omega2"].min(), sub_m["omega2"].max()
        for tool in TOOL_ORDER:
            sub_t = sub_m[sub_m["tool"] == tool]
            for scale in tiers:
                sub = sub_t[sub_t["between_scale"] == scale].sort_values("n_per_group")
                if sub.empty:
                    continue
                omega2 = sub["omega2"].iloc[0]
                alpha_line = 0.30 + 0.70 * (omega2 - omin) / max(omax - omin, 1e-9)
                ax.plot(
                    sub["n_per_group"],
                    sub["power"],
                    color=TOOL_COLORS[tool],
                    alpha=alpha_line,
                    linewidth=1.0,
                    marker="o",
                    markersize=2.4,
                    markeredgewidth=0,
                )
        ax.axhline(0.8, color="black", linestyle="--", linewidth=0.7, zorder=0)
        ax.set_title(METRIC_LABELS[metric], fontsize=9)
        ax.set_xlabel("Samples per group (n)")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(sorted(all_df["n_per_group"].unique()))
        ax.tick_params(width=0.6, length=2.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("PERMANOVA power (alpha = 0.05)")

    tool_handles = [
        Line2D([0], [0], color=TOOL_COLORS[t], linewidth=1.4, label=t) for t in TOOL_ORDER
    ]
    fig.legend(
        handles=tool_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=8,
    )
    fig.text(
        0.5,
        -0.02,
        "Line opacity encodes the effect tier: more opaque = larger realized PERMANOVA omega-squared.",
        ha="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(FIGURES / "fig_benchmark_power.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_benchmark_power.pdf", bbox_inches="tight")
    print("figure written to", FIGURES)


if __name__ == "__main__":
    main()
