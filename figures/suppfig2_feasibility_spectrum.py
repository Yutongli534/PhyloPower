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

Default mode (no arguments) only plots, from the PhyloPower archived runs and
the archived PXD069517 null p-value CSV at
``validation_datasets/results/PXD069517_typeI_null/typeI_null_pvalues.csv``.
It runs in the base environment (matplotlib/numpy/pandas only). If that CSV
is missing (e.g. a release checkout without the archived results), default
mode automatically falls back to the ``--compute`` path below and prints a
notice::

    python3 figures/suppfig2_feasibility_spectrum.py

``--compute`` mode first recomputes the PXD069517 type-I null calibration
VERBATIM as the retired producer ``analysis/run_pxd069517_typeI.py`` did (now
archived in ``_archive_scripts/``): PCAM null pools -> the real PhyloFunc
distance recomputed per pool -> relabel-bootstrap PERMANOVA p-values. It
writes ``typeI_null_pvalues.csv``, ``typeI_summary.json`` and
``typeI_null_qq.{png,pdf}`` to ``--out`` (same filenames and CSV columns as
the producer), then plots exactly as in default mode. All producer CLI knobs
are carried over with identical defaults (``--group-file``, ``--table-file``,
``--tree-file``, ``--eval-ns``, ``--pool-size``, ``--n-pool-seeds``,
``--n-reps``, ``--permutations``, ``--alpha``, ``--seed``, ``--workers``,
``--plot-only``, ``--out``). Compute mode needs the QIIME2 metagenome
environment (scikit-bio, scipy, phylopower core runtime)::

    /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python \
        figures/suppfig2_feasibility_spectrum.py --compute --workers 4

Writes figures/output/suppfig2_feasibility_spectrum.{png,pdf,svg} and
suppfig2_source_data.csv.
"""
import argparse
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


# ---------------------------------------------------------------------------
# --compute mode: PXD069517 type-I null calibration, carried over verbatim
# from the retired producer analysis/run_pxd069517_typeI.py (now archived in
# _archive_scripts/). Computational logic (seeds, defaults, output filenames,
# CSV columns) is unchanged; only the heavy imports were made lazy and the
# helper functions prefixed so the default plotting mode keeps working in the
# base environment.

_COMPUTE_WORKER_DATA = None


def _compute_worker_init(group_file: str, table_file: str, tree_file: str) -> None:
    global _COMPUTE_WORKER_DATA
    import sys

    for path in (str(ROOT), str(ROOT / "analysis"), str(ROOT / "figures")):
        if path not in sys.path:
            sys.path.insert(0, path)
    # load_pxd_protein lives verbatim in suppfig1_pilot_convergence.py (both
    # retired producers shared the same helper); bind its lazy analysis
    # modules before calling it.
    import suppfig1_pilot_convergence as _suppfig1

    _suppfig1._load_analysis_modules()
    load_pxd_protein = _suppfig1.load_pxd_protein

    _COMPUTE_WORKER_DATA = load_pxd_protein(Path(group_file), Path(table_file), Path(tree_file))


def _compute_null_pool_task(job: tuple) -> list[dict]:
    """One null pool: one PhyloFunc distance, then relabel-bootstrap p-values
    for every evaluation size."""
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova
    import pcam_gen as pcam

    pool_seed, pool_size, eval_ns, n_reps, perms = job
    table, group_map = pcam.pcam_null_pool(_COMPUTE_WORKER_DATA, pool_size, pool_seed, ndon=1)
    dm = pcam.recompute_distance(_COMPUTE_WORKER_DATA, table)
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


def _compute_apply_style() -> None:
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


def _compute_summarize_and_plot(df: pd.DataFrame, args) -> None:
    import json

    from scipy.stats import beta, kstest

    _compute_apply_style()
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


def run_compute(args) -> None:
    """Recompute the PXD069517 type-I null calibration (verbatim producer
    logic), writing typeI_null_pvalues.csv, typeI_summary.json and
    typeI_null_qq.{png,pdf} to ``args.out``."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    import sys

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phylopower-mpl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    for path in (str(ROOT), str(ROOT / "analysis")):
        if path not in sys.path:
            sys.path.insert(0, path)

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
            initializer=_compute_worker_init,
            initargs=(str(args.group_file), str(args.table_file), str(args.tree_file)),
        ) as executor:
            parts = list(executor.map(_compute_null_pool_task, jobs))
        df = pd.DataFrame.from_records([r for part in parts for r in part])
        df.to_csv(csv_path, index=False)
    _compute_summarize_and_plot(df, args)
    print(args.out / "typeI_null_qq.png", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compute", action="store_true",
                        help="recompute the PXD069517 type-I null calibration "
                             "before plotting (needs the QIIME2 metagenome env)")
    # Producer knobs carried over from analysis/run_pxd069517_typeI.py with
    # identical defaults (anchored at the repo root so any cwd works).
    pxd = ROOT / "validation_datasets" / "processed" / "PXD069517"
    parser.add_argument("--group-file", type=Path, default=pxd / "group.csv")
    parser.add_argument("--table-file", type=Path, default=pxd / "protein_taxon_function.csv")
    parser.add_argument("--tree-file", type=Path, default=pxd / "rooted-tree.nwk")
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
        default=ROOT / "validation_datasets" / "results" / "PXD069517_typeI_null",
    )
    args = parser.parse_args()
    args.eval_ns = [int(x) for x in args.eval_ns.split(",")]
    return args


def main() -> None:
    args = parse_args()
    if not args.compute and not PXD_PVALS.exists():
        # Fallback: the release ships no archived results, so default mode
        # recomputes the null calibration first instead of dying with
        # FileNotFoundError.
        print(
            f"[suppfig2] archived data not found ({PXD_PVALS}); computing from scratch "
            "(--compute with default knobs; this can take a while and needs the QIIME2 "
            "metagenome env) ...",
            flush=True,
        )
        args.compute = True
    if args.compute:
        run_compute(args)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
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
