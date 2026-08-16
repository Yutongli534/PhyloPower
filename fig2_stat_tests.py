#!/usr/bin/env python3
"""Statistical-test version of Figure 2 fidelity validation.

This trial figure follows the MIDASim/SparseDOSSA2-style reporting more closely:
sample-size matched synthetic subsets are repeatedly drawn, and each fidelity
metric is summarized by KS statistic, KS p-value, and p>0.05 pass rate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import figstyle  # noqa: E402
from fig2 import (  # noqa: E402
    GENE,
    PROTEIN,
    REAL,
    _distance_metrics,
    _eigenspectrum,
    _feature_stats,
    _make_gene,
    _make_protein,
    _protein_topology,
    _safe_ratio,
    _within_between,
)
from phylopower import core  # noqa: E402

core.load_core_runtime()
figstyle.apply_style()

GRID = "#e7e7e7"


def _matched_syn_cols(real_sgm: pd.Series, syn_sgm: pd.Series, rng: np.random.Generator) -> list[str]:
    cols: list[str] = []
    for group, n in real_sgm.value_counts().items():
        members = syn_sgm[syn_sgm == group].index.to_numpy()
        replace = len(members) < int(n)
        cols.extend(rng.choice(members, size=int(n), replace=replace).tolist())
    return cols


def _nonzero_moments(tab: pd.DataFrame) -> dict[str, np.ndarray]:
    arr = tab.to_numpy(dtype=float)
    positive = arr > 0
    prev = positive.mean(axis=1)
    nz_mean = np.zeros(arr.shape[0], dtype=float)
    nz_var = np.zeros(arr.shape[0], dtype=float)
    for i in range(arr.shape[0]):
        vals = arr[i, positive[i]]
        if vals.size:
            nz_mean[i] = float(vals.mean())
            nz_var[i] = float(vals.var())
    return {"prevalence": prev, "nonzero_mean": nz_mean, "nonzero_variance": nz_var}


def _ks(real: np.ndarray, syn: np.ndarray) -> tuple[float, float]:
    real = np.asarray(real, dtype=float)
    syn = np.asarray(syn, dtype=float)
    real = real[np.isfinite(real)]
    syn = syn[np.isfinite(syn)]
    if real.size == 0 or syn.size == 0:
        return np.nan, np.nan
    res = ks_2samp(real, syn)
    return float(res.statistic), float(res.pvalue)


def _rank(real: np.ndarray, syn: np.ndarray) -> float:
    real = np.asarray(real, dtype=float)
    syn = np.asarray(syn, dtype=float)
    mask = np.isfinite(real) & np.isfinite(syn)
    if mask.sum() < 3:
        return np.nan
    rho = spearmanr(real[mask], syn[mask]).correlation
    return float(rho) if np.isfinite(rho) else np.nan


def _long_metric(rows: list[dict], modality: str, block: str, metric: str, d: float, p: float, rep: int) -> None:
    rows.append(
        {
            "modality": modality,
            "block": block,
            "metric": metric,
            "rep": int(rep),
            "ks_d": float(d),
            "p_value": float(p),
            "pass_p_gt_0.05": bool(np.isfinite(p) and p > 0.05),
        }
    )


def _bootstrap_modality(data: dict, modality: str, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    ratios: list[dict] = []

    real_stats = _feature_stats(data["real_tab"])
    real_nz = _nonzero_moments(data["real_tab"])
    real_dm = data["real_dm"]
    real_sgm = data["real_sgm"]
    real_within, real_between = _within_between(real_dm, real_sgm)
    real_spec = _eigenspectrum(real_dm, real_sgm)
    real_omega = max(0.0, float(core.compute_omega2(real_dm, real_sgm)))

    if modality == "protein":
        real_top, _ = data["topology"]
    else:
        real_top = None

    for rep in range(n_boot):
        cols = _matched_syn_cols(data["real_sgm"], data["syn_sgm"], rng)
        syn_tab = data["syn_tab"].loc[:, cols]
        syn_sgm = data["syn_sgm"].loc[cols]
        syn_stats = _feature_stats(syn_tab)
        syn_nz = _nonzero_moments(syn_tab)

        for metric in ["mean", "variance", "prevalence", "library_size", "sample_sparsity"]:
            d, p = _ks(real_stats[metric], syn_stats[metric])
            _long_metric(rows, modality, "feature", metric, d, p, rep)
        for metric in ["nonzero_mean", "nonzero_variance"]:
            d, p = _ks(real_nz[metric], syn_nz[metric])
            _long_metric(rows, modality, "feature", metric, d, p, rep)

        for metric in ["prevalence", "nonzero_mean", "nonzero_variance"]:
            ratios.append(
                {
                    "modality": modality,
                    "block": "rank",
                    "metric": metric,
                    "rep": rep,
                    "value": _rank(real_nz[metric], syn_nz[metric]),
                }
            )

        syn_dm = data["syn_dm"].loc[cols, cols]
        syn_within, syn_between = _within_between(syn_dm, syn_sgm)
        for metric, real_vals, syn_vals in [
            ("within_distance", real_within, syn_within),
            ("between_distance", real_between, syn_between),
        ]:
            d, p = _ks(real_vals, syn_vals)
            _long_metric(rows, modality, "distance", metric, d, p, rep)

        syn_spec = _eigenspectrum(syn_dm, syn_sgm)
        syn_omega = max(0.0, float(core.compute_omega2(syn_dm, syn_sgm)))
        ratios.extend(
            [
                {
                    "modality": modality,
                    "block": "geometry",
                    "metric": "omega2_ratio",
                    "rep": rep,
                    "value": _safe_ratio(syn_omega, real_omega),
                },
                {
                    "modality": modality,
                    "block": "geometry",
                    "metric": "deff_ratio",
                    "rep": rep,
                    "value": _safe_ratio(float(syn_spec["deff"]), float(real_spec["deff"])),
                },
            ]
        )

        if modality == "protein":
            _, syn_top = _protein_topology(data["d"], syn_tab, syn_sgm)
            for metric in ["taxon_degree", "function_degree", "edge_count"]:
                d, p = _ks(real_top[metric], syn_top[metric])
                _long_metric(rows, modality, "tf_topology", metric, d, p, rep)
            ratios.append(
                {
                    "modality": modality,
                    "block": "tf_topology",
                    "metric": "connectance_ratio",
                    "rep": rep,
                    "value": _safe_ratio(float(syn_top["connectance"]), float(real_top["connectance"])),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(ratios)


def _summary(tests: pd.DataFrame) -> pd.DataFrame:
    return (
        tests.groupby(["modality", "block", "metric"], as_index=False)
        .agg(
            median_ks_d=("ks_d", "median"),
            median_p=("p_value", "median"),
            pass_rate=("pass_p_gt_0.05", "mean"),
        )
        .sort_values(["modality", "block", "metric"])
    )


def _pivot(summary: pd.DataFrame, value: str, metrics: list[tuple[str, str, str]]) -> np.ndarray:
    arr = np.full((len(metrics), 2), np.nan)
    for i, (modality, block, metric) in enumerate(metrics):
        sub = summary[(summary["modality"] == modality) & (summary["block"] == block) & (summary["metric"] == metric)]
        if not sub.empty:
            arr[i, 0 if modality == "protein" else 1] = float(sub.iloc[0][value])
    return arr


def _heat(ax, data: np.ndarray, ylabels: list[str], title: str, cmap: str, vmin: float, vmax: float, fmt: str):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([0, 1], ["Protein", "Gene"])
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title(title, fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=7)
    ax.tick_params(axis="y", labelsize=7.4)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[:].set_visible(False)
    return im


def _ratio_panel(ax, ratios: pd.DataFrame):
    keep = ratios[ratios["metric"].isin(["prevalence", "nonzero_mean", "nonzero_variance", "omega2_ratio", "deff_ratio", "connectance_ratio"])]
    labels = []
    values = []
    colors = []
    for modality, color in [("protein", PROTEIN), ("gene", GENE)]:
        subm = keep[keep["modality"] == modality]
        for metric in ["prevalence", "nonzero_mean", "nonzero_variance", "omega2_ratio", "deff_ratio", "connectance_ratio"]:
            sub = subm[subm["metric"] == metric]
            if sub.empty:
                continue
            labels.append(f"{modality[:4]} {metric.replace('_', ' ')}")
            values.append(sub["value"].to_numpy(float))
            colors.append(color)
    pos = np.arange(len(labels))
    meds = [np.nanmedian(v) for v in values]
    lo = [np.nanpercentile(v, 10) for v in values]
    hi = [np.nanpercentile(v, 90) for v in values]
    ax.axvline(1.0, color="#444444", ls="--", lw=1.0)
    ax.errorbar(meds, pos, xerr=[np.asarray(meds) - np.asarray(lo), np.asarray(hi) - np.asarray(meds)],
                fmt="none", ecolor="#666666", lw=1.0, capsize=2, zorder=1)
    ax.scatter(meds, pos, c=colors, s=35, edgecolors="white", linewidths=0.6, zorder=2)
    ax.set_yticks(pos, labels, fontsize=7.2)
    ax.set_xlabel("rank correlation or ratio")
    ax.set_title("Power-relevant ratios and rank agreement", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bar_pass(ax, summary: pd.DataFrame, metrics: list[tuple[str, str, str]], labels: list[str]):
    y = np.arange(len(metrics))[::-1]
    vals = []
    cols = []
    for modality, block, metric in metrics:
        sub = summary[(summary["modality"] == modality) & (summary["block"] == block) & (summary["metric"] == metric)]
        vals.append(float(sub.iloc[0]["pass_rate"]) if not sub.empty else np.nan)
        cols.append(PROTEIN if modality == "protein" else GENE)
    ax.axvspan(0.8, 1.0, color="#e6f1e3", zorder=0)
    ax.barh(y, vals, color=cols, alpha=0.82)
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("pass rate: KS p > 0.05")
    ax.set_title("Sample-size matched indistinguishability", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(summary: pd.DataFrame, ratios: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("protein", "feature", "mean"),
        ("protein", "feature", "variance"),
        ("protein", "feature", "prevalence"),
        ("protein", "feature", "nonzero_mean"),
        ("protein", "feature", "nonzero_variance"),
        ("protein", "tf_topology", "taxon_degree"),
        ("protein", "tf_topology", "function_degree"),
        ("protein", "tf_topology", "edge_count"),
        ("protein", "distance", "within_distance"),
        ("protein", "distance", "between_distance"),
        ("gene", "feature", "mean"),
        ("gene", "feature", "variance"),
        ("gene", "feature", "prevalence"),
        ("gene", "feature", "nonzero_mean"),
        ("gene", "feature", "nonzero_variance"),
        ("gene", "feature", "library_size"),
        ("gene", "feature", "sample_sparsity"),
        ("gene", "distance", "within_distance"),
        ("gene", "distance", "between_distance"),
    ]
    labels = [f"{m[0][:4]} {m[2].replace('_', ' ')}" for m in metrics]
    heat_metrics = [
        ("protein", "feature", "mean"),
        ("protein", "feature", "variance"),
        ("protein", "feature", "prevalence"),
        ("protein", "tf_topology", "taxon_degree"),
        ("protein", "tf_topology", "function_degree"),
        ("protein", "tf_topology", "edge_count"),
        ("protein", "distance", "within_distance"),
        ("protein", "distance", "between_distance"),
        ("gene", "feature", "mean"),
        ("gene", "feature", "variance"),
        ("gene", "feature", "prevalence"),
        ("gene", "feature", "library_size"),
        ("gene", "feature", "sample_sparsity"),
        ("gene", "distance", "within_distance"),
        ("gene", "distance", "between_distance"),
    ]
    heat_labels = [f"{m[1].replace('_', ' ')}\n{m[2].replace('_', ' ')}" for m in heat_metrics]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    _bar_pass(axes[0, 0], summary, metrics, labels)
    im = _heat(
        axes[0, 1],
        _pivot(summary, "median_ks_d", heat_metrics),
        heat_labels,
        "Median KS statistic (D)",
        "YlGnBu_r",
        0,
        0.35,
        ".2f",
    )
    fig.colorbar(im, ax=axes[0, 1], shrink=0.75)
    im = _heat(
        axes[0, 2],
        _pivot(summary, "median_p", heat_metrics),
        heat_labels,
        "Median KS p-value",
        "YlGn",
        0,
        1,
        ".2f",
    )
    fig.colorbar(im, ax=axes[0, 2], shrink=0.75)
    _ratio_panel(axes[1, 0], ratios)

    dist = summary[summary["block"] == "distance"].copy()
    x = np.arange(len(dist))
    colors = [PROTEIN if m == "protein" else GENE for m in dist["modality"]]
    axes[1, 1].scatter(dist["median_ks_d"], dist["pass_rate"], c=colors, s=55, edgecolors="white", linewidths=0.7)
    for _, row in dist.iterrows():
        axes[1, 1].text(row["median_ks_d"] + 0.004, row["pass_rate"], f"{row['modality'][:4]} {row['metric'].split('_')[0]}", fontsize=7)
    axes[1, 1].set_xlabel("median KS D")
    axes[1, 1].set_ylabel("pass rate p>0.05")
    axes[1, 1].set_title("Downstream distance tests", fontweight="bold")
    axes[1, 1].set_xlim(0, max(0.35, float(dist["median_ks_d"].max()) * 1.2))
    axes[1, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].grid(color=GRID, lw=0.6)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.96,
        "Interpretation\n\n"
        "KS p-values are computed after matching synthetic\n"
        "sample counts to the real pilot in each group.\n\n"
        "Good fidelity: high pass rate, small KS D,\n"
        "rank correlations near 1, ratios near 1.\n\n"
        "Unlike abundance-only generators, the protein\n"
        "panel also tests taxon-function topology.",
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
    )
    fig.suptitle("Figure 2 trial - statistical fidelity tests with matched synthetic subsets", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.0)
    fig.savefig(out / "fig2_stat_tests.png", dpi=260, bbox_inches="tight")
    fig.savefig(out / "fig2_stat_tests.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("fig2_stat_tests"))
    parser.add_argument("--pool-M", type=int, default=220)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--protein-edge-fraction", default="1.25")
    parser.add_argument("--protein-residual-mode", choices=["random", "template"], default="template")
    parser.add_argument("--protein-marginal-strength", default="auto")
    parser.add_argument("--protein-eb-k", default="auto")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    print("[fig2-stat] generating protein pool...", flush=True)
    protein = _make_protein(
        args.pool_M,
        args.seed,
        args.protein_edge_fraction,
        args.protein_marginal_strength,
        args.protein_eb_k,
        args.protein_residual_mode,
        0.5,
        2.0,
        7,
    )
    print("[fig2-stat] generating gene pool...", flush=True)
    gene = _make_gene(args.pool_M, args.seed + 7)

    print("[fig2-stat] bootstrapping matched tests...", flush=True)
    p_tests, p_ratios = _bootstrap_modality(protein, "protein", args.n_boot, args.seed + 101)
    g_tests, g_ratios = _bootstrap_modality(gene, "gene", args.n_boot, args.seed + 202)
    tests = pd.concat([p_tests, g_tests], ignore_index=True)
    ratios = pd.concat([p_ratios, g_ratios], ignore_index=True)
    summary = _summary(tests)

    tests.to_csv(args.out / "fig2_stat_tests_long.csv", index=False)
    ratios.to_csv(args.out / "fig2_stat_ratios_long.csv", index=False)
    summary.to_csv(args.out / "fig2_stat_tests_summary.csv", index=False)
    plot(summary, ratios, args.out)
    print(f"[fig2-stat] done -> {args.out / 'fig2_stat_tests.png'}", flush=True)


if __name__ == "__main__":
    main()
