#!/usr/bin/env python3
"""Type-I calibration check for the PXD069517 metaproteomic workflow.

Mirrors the main-text Figure 1 null design (raw-pool null -> recompute the
real PhyloFunc distance -> relabel bootstrap PERMANOVA p-values), specialized
to PRIDE PXD069517 via the PCAM generator:

- ``pcam_gen.pcam_null_pool`` builds null pools (donors drawn from the pooled
  cohort, no group deviation) that preserve sparsity, library sizes, clade
  blocks, and donor coherence without importing a group signal.
- For each independent null pool, p-values come from bootstrap draws WITH
  replacement from the whole pool combined with RANDOM group labels, so a
  single pool's fixed realized null effect cannot accumulate with sample size.

Outputs a p-value CSV, a JSON summary (rejection rates at alpha = 0.05 and
uniformity tests), and a QQ plot per evaluation size.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phylopower-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import pcam_gen as pcam
from run_pilot_information_supplement_protein_pxd import load_pxd_protein, PXD


_WORKER_DATA = None


def worker_init(group_file: str, table_file: str, tree_file: str) -> None:
    global _WORKER_DATA
    _WORKER_DATA = load_pxd_protein(Path(group_file), Path(table_file), Path(tree_file))


def null_pool_task(job: tuple) -> list[dict]:
    """One null pool: one PhyloFunc distance, then relabel-bootstrap p-values
    for every evaluation size."""
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    pool_seed, pool_size, eval_ns, n_reps, perms = job
    table, group_map = pcam.pcam_null_pool(_WORKER_DATA, pool_size, pool_seed, ndon=1)
    dm = pcam.recompute_distance(_WORKER_DATA, table)
    ids = list(dm.index)
    arr = dm.loc[ids, ids].to_numpy()
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    n = len(ids)
    rng = np.random.default_rng(pool_seed * 99991 + 7)
    records = []
    for eval_n in eval_ns:
        for rep in range(n_reps):
            pick = rng.choice(n, 2 * eval_n, replace=True)
            grp = np.array(["A"] * eval_n + ["B"] * eval_n)
            rng.shuffle(grp)
            sub = arr[np.ix_(pick, pick)]
            try:
                p = float(permanova(DistanceMatrix(sub), grp, permutations=perms)["p-value"])
                if not np.isfinite(p):
                    p = 1.0
            except Exception:
                p = 1.0
            records.append(
                {"pool_seed": pool_seed, "eval_n": eval_n, "rep": rep, "p_value": p}
            )
    return records


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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def summarize_and_plot(df: pd.DataFrame, args: argparse.Namespace) -> None:
    from scipy.stats import beta, kstest

    apply_style()
    eval_ns = sorted(df["eval_n"].unique())
    summary = {"dataset": "PRIDE PXD069517 (CD_only vs PolyAI_CD)", "alpha": args.alpha,
               "pools": int(df["pool_seed"].nunique()), "permutations": args.permutations,
               "eval_sizes": {}}
    fig, axes = plt.subplots(1, len(eval_ns), figsize=(5.6 * len(eval_ns), 5.0),
                             constrained_layout=True)
    if len(eval_ns) == 1:
        axes = [axes]
    for ax, eval_n in zip(axes, eval_ns):
        pvals = np.sort(df.loc[df["eval_n"].eq(eval_n), "p_value"].to_numpy())
        m = len(pvals)
        expected = (np.arange(1, m + 1) - 0.5) / m
        lo = beta.ppf(0.025, np.arange(1, m + 1), m - np.arange(1, m + 1) + 1)
        hi = beta.ppf(0.975, np.arange(1, m + 1), m - np.arange(1, m + 1) + 1)
        ax.fill_between(expected, lo, hi, color="#9ecae1", alpha=0.45, lw=0, label="95% band")
        ax.plot([0, 1], [0, 1], color="#202020", lw=1.4, ls=":", label="Uniform")
        ax.plot(expected, pvals, color="#35679a", lw=2.0, label="Observed null p")
        reject = float(np.mean(pvals < args.alpha))
        se = float(np.sqrt(reject * (1 - reject) / m))
        ks = kstest(pvals, "uniform")
        ax.set_title(f"Null calibration, n={eval_n} per group")
        ax.set_xlabel("Expected null p")
        ax.set_ylabel("Observed null p")
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        summary["eval_sizes"][str(eval_n)] = {
            "n_pvalues": m,
            "rejection_rate_at_alpha": reject,
            "rejection_rate_se": se,
            "ks_uniform_statistic": float(ks.statistic),
            "ks_uniform_p": float(ks.pvalue),
        }
    fig.suptitle("Type-I calibration — Metaproteomics (PXD069517)",
                 fontsize=14, fontweight="bold")
    fig.savefig(args.out / "typeI_null_qq.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "typeI_null_qq.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.out / "typeI_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-file", type=Path, default=PXD / "group.csv")
    parser.add_argument("--table-file", type=Path, default=PXD / "protein_taxon_function.csv")
    parser.add_argument("--tree-file", type=Path, default=PXD / "rooted-tree.nwk")
    parser.add_argument("--eval-ns", default="14,80")
    parser.add_argument("--pool-size", type=int, default=200)
    parser.add_argument("--n-pool-seeds", type=int, default=10)
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path("validation_datasets/results/PXD069517_typeI_null"),
    )
    args = parser.parse_args()
    args.eval_ns = [int(x) for x in args.eval_ns.split(",")]
    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "typeI_null_pvalues.csv"
    if args.plot_only:
        df = pd.read_csv(csv_path)
    else:
        jobs = [
            (args.seed + 7000 + s, args.pool_size, tuple(args.eval_ns), args.n_reps, args.permutations)
            for s in range(args.n_pool_seeds)
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(str(args.group_file), str(args.table_file), str(args.tree_file)),
        ) as executor:
            parts = list(executor.map(null_pool_task, jobs))
        df = pd.DataFrame.from_records([r for part in parts for r in part])
        df.to_csv(csv_path, index=False)
    summarize_and_plot(df, args)
    print(args.out / "typeI_null_qq.png", flush=True)


if __name__ == "__main__":
    main()
