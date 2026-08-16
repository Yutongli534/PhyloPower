#!/usr/bin/env python3
"""Generator fidelity benchmark: whose population approximation is closer to truth?

Focused comparison on the ONE thing PhyloPower does differently:
  - PhyloPower: PCAM feature-level synthesis → recompute distance → population estimate
  - micropower: with-replacement bootstrap from observed distance matrix
  - MPrESS: without-replacement subsample from observed data

All tools use the SAME distance metric (Jaccard), SAME pilot, SAME pool size.
Ground truth = full-cohort distance matrix.

Metric: 2-sample KS statistic (within-group + between-group distance distributions).
Lower KS = generated/estimated distance structure closer to population truth.

Usage:
  python3 benchmark/py/run_generator_fidelity.py
  (requires QIIME 2 env for PhyloPower's gene workflow)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "analysis"))

# --- QIIME 2 compat for Gene workflow ---
try:
    import skbio.stats.distance as _skbio_distance
    _permanova_orig = _skbio_distance.permanova
    def _permanova_seeded(dm, grouping, column=None, permutations=999, seed=None):
        if seed is None:
            return _permanova_orig(dm, grouping, column=column, permutations=permutations)
        state = np.random.get_state()
        try:
            np.random.seed(int(seed) % (2**32))
            return _permanova_orig(dm, grouping, column=column, permutations=permutations)
        finally:
            np.random.set_state(state)
    _skbio_distance.permanova = _permanova_seeded
except ImportError:
    pass

from phylopower import core  # noqa: E402
core.load_core_runtime()
import semisynthetic_power as sp  # noqa: E402

SEED = 20260614
POOL_SIZE = 200  # samples per group
N_PILOT = 10     # pilot size per group
N_REPEATS = 5    # independent pilot draws
N_BOOTSTRAP = 200

# Data paths
QINJ_FULL = _REPO / "validation_datasets" / "processed" / "QinJ_2012_full"
QINJ_TREE = str(QINJ_FULL / "rooted-tree.nwk")
OUT = _REPO / "benchmark" / "results"
FIG = _REPO / "benchmark" / "figures"


def load_qinj_full():
    table = pd.read_csv(QINJ_FULL / "table.csv", index_col=0)
    grp = pd.read_csv(QINJ_FULL / "group.csv")
    gmap = pd.Series(grp["group_name"].values, index=grp["sample_id"].astype(str))
    return table, gmap


def compute_jaccard_dm(table: pd.DataFrame) -> pd.DataFrame:
    """Compute Jaccard distance from a taxon x sample table."""
    from skbio.diversity import beta_diversity
    counts = table.transpose().astype(float)
    dm = beta_diversity("jaccard", (counts > 0).to_numpy().astype(float),
                        ids=list(counts.index))
    return dm.to_data_frame()


def within_between_dists(dm_df: pd.DataFrame, gmap: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Extract within-group and between-group distance values."""
    common = dm_df.index.intersection(gmap.index)
    dm_df = dm_df.loc[common, common]
    gmap = gmap.loc[common]
    groups = sorted(gmap.unique())
    arr = dm_df.to_numpy()
    ids = list(dm_df.index)
    within, between = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if gmap[ids[i]] == gmap[ids[j]]:
                within.append(arr[i, j])
            else:
                between.append(arr[i, j])
    return np.array(within), np.array(between)


def draw_pilot(table, gmap, n_per_group, rng):
    """Draw a random pilot of n_per_group samples per group."""
    groups = sorted(gmap.unique())
    picked = []
    for g in groups:
        members = gmap[gmap == g].index.to_numpy()
        picked.extend(rng.choice(members, size=n_per_group, replace=False).tolist())
    return sorted(picked)


def evaluate_one_pilot(
    table_full: pd.DataFrame, gmap_full: pd.Series,
    pilot_samples: list[str], seed: int
) -> dict:
    """For ONE pilot, compare all three methods against full-cohort truth."""

    # --- Ground truth: full cohort Jaccard ---
    dm_truth = compute_jaccard_dm(table_full)
    w_true, b_true = within_between_dists(dm_truth, gmap_full)

    # --- Pilot data ---
    table_pilot = table_full[pilot_samples]
    gmap_pilot = gmap_full.loc[pilot_samples]
    dm_pilot = compute_jaccard_dm(table_pilot)

    results = {"pilot_seed": seed}

    # ===== PhyloPower: PCAM → recompute Jaccard =====
    pool, pool_gmap, _ = sp.generate_taxon_pool(
        table_pilot, gmap_pilot,
        pool_size_per_group=POOL_SIZE,
        random_seed=seed + 1000,
        between_scale=1.0,   # preserve pilot's effect
        residual_scale=1.0,
        noise_multiplier=0.10,
    )
    dm_phylopower = compute_jaccard_dm(pool)
    w_pp, b_pp = within_between_dists(dm_phylopower, pool_gmap)
    ks_w_pp = ks_2samp(w_true, w_pp).statistic
    ks_b_pp = ks_2samp(b_true, b_pp).statistic
    results["phylopower_ks_within"] = float(ks_w_pp)
    results["phylopower_ks_between"] = float(ks_b_pp)
    results["phylopower_ks_mean"] = float((ks_w_pp + ks_b_pp) / 2)

    # ===== micropower-style: with-replacement bootstrap from observed DM =====
    # micropower bootstraps WITH replacement from the observed distance matrix.
    # We replicate its mechanism: draw n samples/group with replacement from
    # the pilot, extract the corresponding distance submatrix, collect all
    # pairwise distances over N_BOOTSTRAP iterations.
    n_obs = len(pilot_samples) // 2
    rng = np.random.default_rng(seed + 2000)
    mp_within, mp_between = [], []
    groups = sorted(gmap_pilot.unique())
    g_indices = {g: [i for i, s in enumerate(pilot_samples) if gmap_pilot[s] == g]
                 for g in groups}
    dm_pilot_arr = dm_pilot.to_numpy()

    for _ in range(N_BOOTSTRAP):
        picked_idx = []
        for g in groups:
            picked_idx.extend(rng.choice(g_indices[g], size=n_obs, replace=True).tolist())
        sub_arr = dm_pilot_arr[np.ix_(picked_idx, picked_idx)]
        # Collect within/between pairs
        for i in range(len(picked_idx)):
            for j in range(i + 1, len(picked_idx)):
                gi = pilot_samples[picked_idx[i]]
                gj = pilot_samples[picked_idx[j]]
                if gmap_pilot[gi] == gmap_pilot[gj]:
                    mp_within.append(sub_arr[i, j])
                else:
                    mp_between.append(sub_arr[i, j])

    ks_w_mp = ks_2samp(w_true, mp_within).statistic
    ks_b_mp = ks_2samp(b_true, mp_between).statistic
    results["micropower_ks_within"] = float(ks_w_mp)
    results["micropower_ks_between"] = float(ks_b_mp)
    results["micropower_ks_mean"] = float((ks_w_mp + ks_b_mp) / 2)

    # ===== MPrESS-style: without-replacement subsample from observed DM =====
    # MPrESS subsamples WITHOUT replacement from the observed data (when n <= pool).
    rng = np.random.default_rng(seed + 3000)
    ms_within, ms_between = [], []
    dm_pilot_arr2 = dm_pilot.to_numpy()
    pilot_list = list(pilot_samples)

    for _ in range(N_BOOTSTRAP):
        picked_idx = []
        for g in groups:
            picked_idx.extend(rng.choice(g_indices[g], size=min(n_obs, len(g_indices[g])),
                                         replace=False).tolist())
        sub_arr = dm_pilot_arr2[np.ix_(picked_idx, picked_idx)]
        for i in range(len(picked_idx)):
            for j in range(i + 1, len(picked_idx)):
                gi = pilot_samples[picked_idx[i]]
                gj = pilot_samples[picked_idx[j]]
                if gmap_pilot[gi] == gmap_pilot[gj]:
                    ms_within.append(sub_arr[i, j])
                else:
                    ms_between.append(sub_arr[i, j])

    ks_w_ms = ks_2samp(w_true, ms_within).statistic
    ks_b_ms = ks_2samp(b_true, ms_between).statistic
    results["mpress_ks_within"] = float(ks_w_ms)
    results["mpress_ks_between"] = float(ks_b_ms)
    results["mpress_ks_mean"] = float((ks_w_ms + ks_b_ms) / 2)

    # Also: the observed pilot itself (baseline)
    w_pilot, b_pilot = within_between_dists(dm_pilot, gmap_pilot)
    ks_w_pilot = ks_2samp(w_true, w_pilot).statistic
    ks_b_pilot = ks_2samp(b_true, b_pilot).statistic
    results["pilot_ks_within"] = float(ks_w_pilot)
    results["pilot_ks_between"] = float(ks_b_pilot)
    results["pilot_ks_mean"] = float((ks_w_pilot + ks_b_pilot) / 2)

    return results


def apply_style():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 13, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def make_figure(all_results: list[dict]):
    apply_style()
    df = pd.DataFrame(all_results)

    methods = ["pilot", "micropower", "mpress", "phylopower"]
    labels = ["Observed pilot", "micropower\n(bootstrap DM)", "MPrESS\n(subsample DM)",
              "PhyloPower\n(PCAM + recompute)"]
    colors = ["#7a8798", "#1f77b4", "#2ca02c", "#d62728"]
    metrics = ["ks_within", "ks_between", "ks_mean"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), constrained_layout=True)
    titles = ["Within-group KS", "Between-group KS", "Mean KS"]
    bar_w = 0.55

    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]
        values = []
        for m in methods:
            col = f"{m}_{metric}"
            if col in df.columns:
                values.append(df[col].to_numpy())
            else:
                values.append(np.array([]))

        x_pos = np.arange(len(methods))
        for i, (vals, label, color) in enumerate(zip(values, labels, colors)):
            if len(vals) == 0:
                continue
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            ax.bar(i, mean_v, bar_w, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
            ax.errorbar(i, mean_v, yerr=std_v, fmt="none", ecolor="#333333",
                        capsize=4, lw=1.2)
            # Individual points
            rng_jitter = np.random.default_rng(42)
            ax.scatter(i + rng_jitter.uniform(-0.12, 0.12, len(vals)), vals,
                       s=35, color=color, alpha=0.6, edgecolor="white", linewidth=0.3,
                       zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("KS statistic (lower = better)")
        ax.set_title(title)
        ax.set_ylim(0, None)
        # Annotate best method
        means = [np.mean(v) if len(v) > 0 else 999 for v in values]
        best = np.argmin(means)
        ax.text(best, means[best] + 0.01, "best", ha="center", fontsize=9,
                fontweight="bold", color=colors[best])

    fig.suptitle("Generator Fidelity — Jaccard distance vs Full-Cohort Ground Truth\n"
                 f"(QinJ_2012, {N_REPEATS} pilots × {N_PILOT}/group, pool={POOL_SIZE}/group)",
                 fontsize=14, fontweight="bold")
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        dpi = 320 if ext == "png" else None
        fig.savefig(FIG / f"fig_generator_fidelity.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {FIG / 'fig_generator_fidelity.png'}", flush=True)


def make_table(all_results: list[dict]):
    df = pd.DataFrame(all_results)
    methods = [
        ("pilot", "Observed pilot (baseline)"),
        ("micropower", "micropower (bootstrap DM)"),
        ("mpress", "MPrESS (subsample DM)"),
        ("phylopower", "PhyloPower (PCAM + recompute)"),
    ]
    rows = []
    for prefix, label in methods:
        row = {"Method": label}
        for metric in ["ks_within", "ks_between", "ks_mean"]:
            col = f"{prefix}_{metric}"
            if col in df.columns:
                vals = df[col]
                row[metric] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            else:
                row[metric] = "—"
        rows.append(row)

    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "generator_fidelity.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\n→ {OUT / 'generator_fidelity.csv'}", flush=True)

    # Also save detailed per-pilot results
    df.to_csv(OUT / "generator_fidelity_detail.csv", index=False)

    # Relative improvement over pilot baseline
    pilot_mean = df["pilot_ks_mean"].mean()
    for prefix in ["micropower", "mpress", "phylopower"]:
        col = f"{prefix}_ks_mean"
        if col in df.columns:
            tool_mean = df[col].mean()
            improvement = (pilot_mean - tool_mean) / pilot_mean * 100
            print(f"{prefix}: mean KS {tool_mean:.3f} vs pilot {pilot_mean:.3f} "
                  f"({improvement:+.1f}%)", flush=True)


def main():
    table_full, gmap_full = load_qinj_full()
    print(f"Full cohort: {len(table_full.columns)} samples, {len(table_full)} species")

    rng_main = np.random.default_rng(SEED)
    all_results = []
    for rep in range(N_REPEATS):
        pilot_seed = int(rng_main.integers(2**31 - 1))
        pilot = draw_pilot(table_full, gmap_full, N_PILOT,
                           np.random.default_rng(pilot_seed))
        print(f"\nPilot {rep+1}/{N_REPEATS} (seed={pilot_seed}): "
              f"{len(pilot)} samples", flush=True)

        res = evaluate_one_pilot(table_full, gmap_full, pilot, pilot_seed)
        all_results.append(res)
        print(f"  PhyloPower:  within={res['phylopower_ks_within']:.3f}  "
              f"between={res['phylopower_ks_between']:.3f}  "
              f"mean={res['phylopower_ks_mean']:.3f}", flush=True)
        print(f"  micropower:  within={res['micropower_ks_within']:.3f}  "
              f"between={res['micropower_ks_between']:.3f}  "
              f"mean={res['micropower_ks_mean']:.3f}", flush=True)
        print(f"  MPrESS:      within={res['mpress_ks_within']:.3f}  "
              f"between={res['mpress_ks_between']:.3f}  "
              f"mean={res['mpress_ks_mean']:.3f}", flush=True)
        print(f"  Pilot only:  within={res['pilot_ks_within']:.3f}  "
              f"between={res['pilot_ks_between']:.3f}  "
              f"mean={res['pilot_ks_mean']:.3f}", flush=True)

    make_figure(all_results)
    make_table(all_results)
    print("\nGENERATOR FIDELITY BENCHMARK DONE", flush=True)


if __name__ == "__main__":
    main()
