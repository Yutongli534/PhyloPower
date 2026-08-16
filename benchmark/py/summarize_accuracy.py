#!/usr/bin/env python3
"""Route A: merge truth + tool estimates, compute bias / |bias| / RMSE / SD per
(tool, metric, tier, n) cell, and render the accuracy figure.

Inputs:  benchmark/results/accuracy_truth/truth_*.csv
         benchmark/results/accuracy_estimates/{phylopower,micropower,mpress}_*.csv
Outputs: benchmark/results/accuracy_truth.csv
         benchmark/results/accuracy_estimates_long.csv
         benchmark/results/accuracy_summary.csv
         benchmark/figures/fig_accuracy.{png,pdf}
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
TIER_LABELS = {0.0: "near-null tier (scale 0)", 0.5: "dilution tier (scale 0.5)",
               1.0: "baseline tier (scale 1.0)"}


def main() -> None:
    truth = pd.concat(
        [pd.read_csv(p) for p in sorted((RESULTS / "accuracy_truth").glob("truth_*.csv"))],
        ignore_index=True,
    ).drop_duplicates(subset=["metric", "between_scale", "n_per_group"], keep="last")
    truth.to_csv(RESULTS / "accuracy_truth.csv", index=False)

    est = pd.concat(
        [pd.read_csv(p) for p in sorted((RESULTS / "accuracy_estimates").glob("*.csv"))],
        ignore_index=True,
    ).drop_duplicates(subset=["tool", "metric", "between_scale", "n_per_group", "rep"], keep="last")
    est.to_csv(RESULTS / "accuracy_estimates_long.csv", index=False)

    merged = est.merge(truth, on=["metric", "between_scale", "n_per_group"], how="left")
    if merged["truth_power"].isna().any():
        missing = merged[merged["truth_power"].isna()][
            ["metric", "between_scale", "n_per_group"]
        ].drop_duplicates()
        print("WARNING: cells without truth:\n", missing, file=sys.stderr)
    merged["error"] = merged["power_est"] - merged["truth_power"]

    summary = (
        merged.groupby(["tool", "metric", "between_scale", "n_per_group"])
        .agg(
            truth_power=("truth_power", "first"),
            mean_est=("power_est", "mean"),
            sd_est=("power_est", "std"),
            bias=("error", "mean"),
            abs_bias=("error", lambda e: float(np.mean(np.abs(e)))),
            rmse=("error", lambda e: float(np.sqrt(np.mean(e**2)))),
            n_reps=("rep", "count"),
        )
        .reset_index()
        .sort_values(["metric", "between_scale", "n_per_group", "tool"])
    )
    summary.to_csv(RESULTS / "accuracy_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    # ---- figure: per-cell bias, rep-level points + mean ----
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        plt.rcParams["font.family"] = "Arial"
    except ValueError:
        print("WARNING: Arial not found, falling back", file=sys.stderr)
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    plot_df = merged.dropna(subset=["truth_power"]).copy()
    tiers = sorted(plot_df["between_scale"].unique())
    metrics = ["braycurtis", "wunifrac"]
    fig, axes = plt.subplots(
        len(tiers), len(metrics),
        figsize=(7.2, 2.4 * len(tiers)),
        sharex=False, squeeze=False,
    )
    rng = np.random.default_rng(7)
    for r, tier in enumerate(tiers):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            sub = plot_df[(plot_df["between_scale"] == tier) & (plot_df["metric"] == metric)]
            ns = sorted(sub["n_per_group"].unique())
            x_index = {n: i for i, n in enumerate(ns)}
            offsets = {"PhyloPower": -0.22, "micropower": 0.0, "MPrESS": 0.22}
            for tool in TOOL_ORDER:
                st = sub[sub["tool"] == tool]
                if st.empty:
                    continue
                xs = st["n_per_group"].map(x_index).to_numpy() + offsets[tool]
                xs = xs + rng.uniform(-0.05, 0.05, size=len(xs))
                ax.scatter(xs, st["error"], s=6, alpha=0.25, color=TOOL_COLORS[tool],
                           linewidths=0, zorder=2)
                mean_by_n = st.groupby("n_per_group")["error"].mean()
                sd_by_n = st.groupby("n_per_group")["error"].std()
                mx = mean_by_n.index.map(x_index).to_numpy() + offsets[tool]
                ax.errorbar(mx, mean_by_n.to_numpy(), yerr=sd_by_n.to_numpy(),
                            color=TOOL_COLORS[tool], marker="o", markersize=3.5,
                            linewidth=1.1, capsize=2, zorder=3, label=tool)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.7, zorder=1)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels([str(n) for n in ns])
            ax.set_title(f"{METRIC_LABELS[metric]} — {TIER_LABELS.get(tier, f'scale {tier}')}",
                         fontsize=8.5)
            ax.tick_params(width=0.6, length=2.5)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            if c == 0:
                ax.set_ylabel("estimated power - truth")
            if r == len(tiers) - 1:
                ax.set_xlabel("Samples per group (n)")
    handles = [Line2D([0], [0], color=TOOL_COLORS[t], marker="o", markersize=4,
                      linewidth=1.1, label=t) for t in TOOL_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_accuracy.png", dpi=450, bbox_inches="tight")
    fig.savefig(FIGURES / "fig_accuracy.pdf", bbox_inches="tight")
    print("figure written to", FIGURES)


if __name__ == "__main__":
    main()
