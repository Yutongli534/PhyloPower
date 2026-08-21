#!/usr/bin/env python3
"""Supplementary Figure S1 - pilot-information convergence in both data types.

Four panels: metagenomic (a,b) from data/pilot_information_supplement/ and
metaproteomic PXD069517 (c,d) from validation_datasets/results/
PXD069517_pilot_information/. Only plots archived metrics tables.

Default (no arguments): plot only, from the archived metrics tables.  Works in
the base Python 3 environment.  If a metrics table is missing (e.g. a release
checkout without the archived data), default mode automatically falls back to
recomputing the missing side(s) as with ``--compute``, and prints a notice.

``--compute gene`` / ``--compute protein`` / ``--compute all`` first recompute
the corresponding archive(s) exactly as the retired producers
``analysis/run_pilot_information_supplement.py`` (gene side) and
``analysis/run_pilot_information_supplement_protein_pxd.py`` (protein side,
now both in ``_archive_scripts/``) did — same seeds, defaults, effect grids,
cache files, output filenames, and CSV columns — then plot.  Each producer's
CLI knobs are carried over with identical defaults; names that collide between
the two producers are prefixed with ``--gene-`` / ``--protein-``.  The heavy
analysis imports are lazy inside the compute path, so base-env default
plotting keeps working.

Environment needs:

- Default plotting: base Python 3 environment.
- ``--compute gene``: the QIIME 2 / Gemelli environment
  (/opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python); the gene
  distance recomputation imports gemelli/biom/scikit-bio.
- ``--compute protein``: base Python 3 environment (PhyloFunc runs
  in-process).

Run from the repository root.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

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


# ---------------------------------------------------------------------------
# Compute path: lazy imports of the analysis/ helpers.
#
# The heavy modules (pcam_gen, _fig4_curve_plotting, and through them
# phylopower/semisynthetic_power) are imported lazily so the default plotting
# mode never needs them.  Worker initializers call _load_analysis_modules()
# again inside each spawned child process, which re-establishes sys.path and
# the module globals there.
# ---------------------------------------------------------------------------

pcam = None
binned_monotone = None
fit_binned_null_hill = None


def _compute_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phylopower-mpl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _load_analysis_modules() -> None:
    global pcam, binned_monotone, fit_binned_null_hill
    if pcam is not None:
        return
    for path in (str(ROOT), str(ROOT / "analysis")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import pcam_gen as _pcam
    from _fig4_curve_plotting import binned_monotone as _bm
    from _fig4_curve_plotting import fit_binned_null_hill as _fbh

    pcam = _pcam
    binned_monotone = _bm
    fit_binned_null_hill = _fbh


# ---------------------------------------------------------------------------
# Shared producer helpers (identical in both retired producers).
# ---------------------------------------------------------------------------

_WORKER_DATA = None

PALETTE = ["#4b006e", "#35679a", "#ffcf24", "#0f8b7c", "#b45309"]


def _producer_apply_style() -> None:
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
            "font.size": 11.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fit_curve(rows: pd.DataFrame, x: np.ndarray, bin_width: float) -> np.ndarray | None:
    rows = rows[["true_omega2", "power"]].dropna().sort_values("true_omega2")
    if len(rows) < 4 or rows["true_omega2"].nunique() < 3:
        return None
    binned = binned_monotone(rows, bin_width)
    y, _ = fit_binned_null_hill(binned, x)
    if y is not None and np.all(np.isfinite(y)):
        return np.clip(y, 0.0, 1.0)
    sx = rows["true_omega2"].clip(lower=0).to_numpy(float)
    sy = np.maximum.accumulate(rows["power"].clip(0, 1).to_numpy(float))
    ux, idx = np.unique(sx, return_index=True)
    if len(ux) < 2:
        return None
    return np.interp(x, ux, sy[idx], left=sy[idx][0], right=sy[idx][-1])


def crossing(x: np.ndarray, y: np.ndarray, target: float = 0.8) -> float:
    hit = np.flatnonzero(y >= target)
    if not len(hit):
        return np.nan
    i = int(hit[0])
    if i == 0:
        return float(x[0])
    x0, x1 = float(x[i - 1]), float(x[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 <= y0:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def sorted_pilot_view(data: dict, pilot_n: int, seed: int) -> dict:
    """Return a pilot view with deterministic donor ordering.

    Sorting is essential for a common-random-number experiment: if a pilot
    contains every available sample, changing the pilot seed must not merely
    permute donor positions and create artificial between-replicate variation.
    """
    rng = np.random.default_rng(seed)
    groups = data["groups"]
    selected = {
        group: np.sort(
            rng.choice(indices, min(pilot_n, len(indices)), replace=False)
        )
        for group, indices in data["gs"].items()
    }
    pooled = np.concatenate([selected[group] for group in groups])
    grand = data["L"][:, pooled].mean(axis=1)
    deviations = {
        group: data["L"][:, selected[group]].mean(axis=1) - grand
        for group in groups
    }
    view = dict(data)
    view["gs"] = selected
    view["dev"] = deviations
    view["libs"] = data["abund"][:, pooled].sum(axis=0)
    return view


# ---------------------------------------------------------------------------
# Gene-side compute (verbatim from analysis/run_pilot_information_supplement.py).
#
# The empirical cohort, effect grid, evaluation size, and random numbers used
# to generate each effect point are held fixed.  Only the pilot subset and its
# size change.  Curves are compared with the curve obtained from the full
# empirical cohort, so decreasing curve-to-reference error with pilot size is
# evidence for finite-pilot information as the source of disagreement.
# ---------------------------------------------------------------------------

GENE_EFFECT_GRID = {
    "gene": [
        (0.50, 1.0), (0.60, 1.0), (0.68, 1.0), (0.75, 1.0),
        (0.82, 1.0), (0.88, 1.0), (0.93, 1.0), (0.97, 1.0),
        (1.00, 1.0), (1.00, 1.3), (1.00, 1.7),
    ],
    "protein": [
        (0.50, 1.0), (0.60, 1.0), (0.68, 1.0), (0.75, 1.0),
        (0.82, 1.0), (0.88, 1.0), (0.93, 1.0), (0.97, 1.0),
        (1.00, 1.0), (1.00, 1.3), (1.00, 1.7), (1.00, 2.1),
    ],
}


def gene_worker_init(modality: str) -> None:
    global _WORKER_DATA
    _load_analysis_modules()
    _WORKER_DATA = pcam.load_modality(modality)


def gene_worker_task(job: tuple) -> tuple:
    from phylopower import core
    from semisynthetic_power import summarize_distance_metrics_with_replacement

    core.load_core_runtime()
    pi, scale, pool_size, generation_seed, eval_n, boot, pilot_n, pilot_seed = job
    pilot = sorted_pilot_view(_WORKER_DATA, pilot_n, pilot_seed)
    table, group_map = pcam.pcam_pool(
        pilot, pool_size, generation_seed, pi, scale, ndon=1
    )
    distance = pcam.recompute_distance(pilot, table)
    omega2 = max(0.0, float(core.compute_omega2(distance, group_map)))
    metrics = summarize_distance_metrics_with_replacement(
        dm=distance,
        group_map=group_map,
        boot_number=boot,
        alpha=0.05,
        n_jobs=1,
        random_seed=7,
        n_per_group=eval_n,
        permutations=99,
        omega2_floor=0.0,
    )
    return pilot_n, pilot_seed, pi, scale, omega2, float(metrics["power"])


def gene_simulate_one_size(
    modality: str,
    pilot_n: int,
    reps: int,
    eval_n: int,
    pool_size: int,
    boot: int,
    seed: int,
    workers: int,
) -> pd.DataFrame:
    jobs = []
    for rep in range(reps):
        pilot_seed = seed + pilot_n * 1009 + rep * 131
        for gi, (pi, scale) in enumerate(GENE_EFFECT_GRID[modality]):
            # Common random numbers across pilot sizes/repetitions reduce Monte
            # Carlo noise unrelated to the pilot subset itself.
            generation_seed = seed + 50_000 + gi * 9973
            jobs.append(
                (pi, scale, pool_size, generation_seed, eval_n, boot, pilot_n, pilot_seed)
            )
    with ProcessPoolExecutor(
        max_workers=workers, initializer=gene_worker_init, initargs=(modality,)
    ) as executor:
        result = list(executor.map(gene_worker_task, jobs))
    records = []
    seed_to_rep = {
        seed + pilot_n * 1009 + rep * 131: rep for rep in range(reps)
    }
    for pn, pilot_seed, pi, scale, omega2, power in result:
        records.append(
            {
                "modality": modality,
                "pilot_n": pn,
                "rep": seed_to_rep[pilot_seed],
                "pi": pi,
                "scale": scale,
                "true_omega2": omega2,
                "power": power,
                "eval_n": eval_n,
            }
        )
    return pd.DataFrame.from_records(records)


def gene_run_simulation(args: argparse.Namespace) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    parts = []
    for pilot_n in args.pilots:
        cache = args.out / f"raw_pilot_{pilot_n}.csv"
        if cache.exists() and not args.force:
            part = pd.read_csv(cache)
            print(f"[pilot-info] reused {cache}", flush=True)
        else:
            print(
                f"[pilot-info] {args.modality}: pilot n={pilot_n}, "
                f"reps={args.reps}, points={len(GENE_EFFECT_GRID[args.modality])}",
                flush=True,
            )
            part = gene_simulate_one_size(
                args.modality, pilot_n, args.reps, args.eval_n,
                args.pool_size, args.boot, args.seed, args.workers,
            )
            part.to_csv(cache, index=False)
        parts.append(part)

    # A full-cohort reference.  A pilot_n much larger than either group causes
    # pilot_view() to retain every empirical sample.
    ref_cache = args.out / "raw_full_cohort_reference.csv"
    if ref_cache.exists() and not args.force:
        reference = pd.read_csv(ref_cache)
    else:
        print("[pilot-info] full-cohort reference", flush=True)
        reference = gene_simulate_one_size(
            args.modality, 999, args.reference_reps, args.eval_n,
            # Keep the generation seed identical to the pilot runs.  Because
            # the full cohort is deterministically ordered, this makes the
            # largest possible pilot an exact common-random-number match to
            # the reference and isolates information loss from Monte Carlo
            # variation.
            args.pool_size, args.boot, args.seed, args.workers,
        )
        reference.to_csv(ref_cache, index=False)
    parts.append(reference)

    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(args.out / "pilot_information_raw.csv", index=False)
    return raw


def gene_summarize_and_plot(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

    _producer_apply_style()
    xmax = args.xmax
    x = np.linspace(0.0, xmax, 700)
    bin_width = args.bin_width

    ref_all = raw[raw["pilot_n"].eq(999)]
    # Common-random-number reference replicates are intentionally identical.
    # Fit one copy so duplicated rows do not change the curve fitter's weights.
    ref_rows = ref_all[ref_all["rep"].eq(ref_all["rep"].min())]
    reference = fit_curve(ref_rows, x, bin_width)
    if reference is None:
        raise RuntimeError("Could not fit the full-cohort reference curve")

    curves: dict[int, list[tuple[int, np.ndarray]]] = {}
    summary = []
    for pilot_n in args.pilots:
        curves[pilot_n] = []
        subset = raw[raw["pilot_n"].eq(pilot_n)]
        for rep, rows in subset.groupby("rep"):
            curve = fit_curve(rows, x, bin_width)
            if curve is None:
                continue
            curves[pilot_n].append((int(rep), curve))
            summary.append(
                {
                    "pilot_n": pilot_n,
                    "rep": int(rep),
                    "mean_abs_curve_error": float(np.trapz(np.abs(curve - reference), x) / xmax),
                    "omega2_at_80_power": crossing(x, curve, 0.8),
                    "reference_omega2_at_80_power": crossing(x, reference, 0.8),
                }
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out / "pilot_information_curve_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(args.pilots)}

    # A: median curves and repeated-pilot envelopes for every pilot size.
    ax = axes[0, 0]
    ax.plot(x, reference, color="#202020", lw=3.0, label="Full-cohort reference", zorder=5)
    for pilot_n in args.pilots:
        stack = np.vstack([curve for _, curve in curves[pilot_n]])
        color = colors[pilot_n]
        ax.fill_between(
            x, np.percentile(stack, 10, axis=0), np.percentile(stack, 90, axis=0),
            color=color, alpha=0.16, lw=0,
        )
        ax.plot(x, np.median(stack, axis=0), color=color, lw=2.0, label=f"pilot n={pilot_n}")
    ax.set_title("Repeated-pilot power curves")
    ax.set_xlabel("Realized effect size (ω²)")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", fontsize=9)

    # B: individual curves at the smallest and largest pilot sizes.
    ax = axes[0, 1]
    small, large = min(args.pilots), max(args.pilots)
    for pilot_n, linestyle in ((small, "-"), (large, "--")):
        color = colors[pilot_n]
        for j, (_, curve) in enumerate(curves[pilot_n]):
            ax.plot(
                x, curve, color=color, lw=1.0, alpha=0.38, ls=linestyle,
                label=f"pilot n={pilot_n}" if j == 0 else None,
            )
    ax.plot(x, reference, color="#202020", lw=3.0, label="Full-cohort reference", zorder=5)
    ax.set_title("Between-pilot disagreement")
    ax.set_xlabel("Realized effect size (ω²)")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", fontsize=9)

    # C: the primary quantitative test.
    ax = axes[1, 0]
    values = [
        summary_df.loc[summary_df["pilot_n"].eq(n), "mean_abs_curve_error"].dropna().to_numpy()
        for n in args.pilots
    ]
    bp = ax.boxplot(values, positions=np.arange(len(args.pilots)), widths=0.55, patch_artist=True)
    for patch, n in zip(bp["boxes"], args.pilots):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.40)
    rng = np.random.default_rng(args.seed)
    for i, (n, vals) in enumerate(zip(args.pilots, values)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=24, color=colors[n], alpha=0.80)
    ax.set_xticks(np.arange(len(args.pilots)), [str(n) for n in args.pilots])
    ax.set_title("Curve-to-reference disagreement")
    ax.set_xlabel("Pilot size per group")
    ax.set_ylabel("Mean absolute power difference")

    # D: decision-relevant stability of the 80% power threshold.
    ax = axes[1, 1]
    threshold_values = [
        summary_df.loc[summary_df["pilot_n"].eq(n), "omega2_at_80_power"].dropna().to_numpy()
        for n in args.pilots
    ]
    bp = ax.boxplot(
        threshold_values, positions=np.arange(len(args.pilots)), widths=0.55, patch_artist=True
    )
    for patch, n in zip(bp["boxes"], args.pilots):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.40)
    for i, (n, vals) in enumerate(zip(args.pilots, threshold_values)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=24, color=colors[n], alpha=0.80)
    ref_cross = crossing(x, reference, 0.8)
    ax.axhline(ref_cross, color="#202020", lw=2.0, ls=":", label="Full-cohort reference")
    ax.set_xticks(np.arange(len(args.pilots)), [str(n) for n in args.pilots])
    ax.set_title("Stability of the 80% power threshold")
    ax.set_xlabel("Pilot size per group")
    ax.set_ylabel("ω² required for 80% power")
    ax.legend(loc="best", fontsize=9)

    for letter, ax in zip("abcd", axes.ravel()):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")
    for ax in axes.ravel()[:2]:
        ax.axhline(0.8, color="#7a8798", lw=1.1, ls=":")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.03)

    title_modality = "Metagenomics (Gemelli)" if args.modality == "gene" else "Metaproteomics"
    fig.suptitle(
        f"Finite-pilot disagreement and convergence — {title_modality}",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(args.out / "pilot_information_supplement.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "pilot_information_supplement.pdf", bbox_inches="tight")
    plt.close(fig)

    medians = summary_df.groupby("pilot_n", as_index=False).median(numeric_only=True)
    rho, rho_p = spearmanr(summary_df["pilot_n"], summary_df["mean_abs_curve_error"])
    paired = summary_df.pivot(
        index="rep", columns="pilot_n", values="mean_abs_curve_error"
    )
    friedman = friedmanchisquare(*(paired[n] for n in args.pilots))
    pairwise = []
    for smaller, larger in zip(args.pilots[:-1], args.pilots[1:]):
        test = wilcoxon(
            paired[smaller], paired[larger], alternative="greater"
        )
        pairwise.append(
            {
                "smaller_pilot_n": smaller,
                "larger_pilot_n": larger,
                "paired_w": float(test.statistic),
                "one_sided_p": float(test.pvalue),
            }
        )
    report = {
        "modality": args.modality,
        "eval_n": args.eval_n,
        "reps_requested": args.reps,
        "reference_omega2_at_80_power": ref_cross,
        "spearman_pilot_n_vs_curve_error": {
            "rho": float(rho),
            "p": float(rho_p),
        },
        "friedman_repeated_pilot_size_test": {
            "chi_square": float(friedman.statistic),
            "p": float(friedman.pvalue),
        },
        "adjacent_paired_wilcoxon_greater": pairwise,
        "median_metrics_by_pilot_n": medians.to_dict(orient="records"),
    }
    (args.out / "pilot_information_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def compute_gene(cli: argparse.Namespace) -> None:
    _compute_env()
    _load_analysis_modules()
    args = argparse.Namespace(
        modality=cli.modality,
        pilots=[int(x) for x in cli.gene_pilots.split(",")],
        reps=cli.gene_reps,
        reference_reps=cli.gene_reference_reps,
        eval_n=cli.gene_eval_n,
        pool_size=cli.gene_pool_size,
        boot=cli.gene_boot,
        workers=cli.gene_workers,
        seed=cli.gene_seed,
        xmax=cli.gene_xmax,
        bin_width=cli.gene_bin_width,
        force=cli.force,
        plot_only=cli.plot_only,
        out=cli.gene_out,
    )
    raw_path = args.out / "pilot_information_raw.csv"
    if args.plot_only:
        raw = pd.read_csv(raw_path)
    else:
        raw = gene_run_simulation(args)
    gene_summarize_and_plot(raw, args)
    print(args.out / "pilot_information_supplement.png", flush=True)


# ---------------------------------------------------------------------------
# Protein-side compute (verbatim from
# analysis/run_pilot_information_supplement_protein_pxd.py).
#
# Metaproteomic counterpart for PRIDE PXD069517 (CD_only vs PolyAI_CD, 14
# samples per group).  The PXD069517 cohort shows essentially no observed group
# separation (omega-squared of the unmodified distance matrix is about
# -0.007), so a small pilot's apparent group difference is dominated by
# subsampling noise.  Results are cached per pilot size and per repeated pilot
# draw, so an interrupted run resumes where it stopped.
# ---------------------------------------------------------------------------

PXD = Path("validation_datasets/processed/PXD069517")


def load_pxd_protein(group_file: Path, table_file: Path, tree_file: Path, K_prot=16) -> dict:
    """Mirror of ``pcam_gen.load_modality('protein', ...)`` specialized to the
    PXD069517 identifier quirk: the table's ``Taxon`` ids use an underscore
    (``taxon_006d...``).  Bio.Phylo (used by ``phylofunc_fast``) preserves the
    underscore when parsing the tree, but scikit-bio (used by
    ``pcam_gen.clade_assign``) converts it to a space per Newick convention.
    The table is therefore kept unchanged, and taxa are converted to the
    scikit-bio spelling only for the clade-assignment step.  Kept here because
    pcam_gen.py must not be modified."""
    from phylopower import core

    core.load_core_runtime()
    from semisynthetic_power import _read_group_map
    from skbio import TreeNode

    gm = _read_group_map(Path(group_file))
    df = pd.read_csv(table_file)
    samples = [c for c in df.columns if c in gm.index]
    df = df[["Taxon", "Function"] + samples]
    df[samples] = df[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    unit = df["Taxon"].astype(str).to_numpy()
    abund = df[samples].to_numpy(float)
    taxa = pd.unique(unit)
    uid = pd.Series(np.arange(len(taxa)), index=taxa).loc[unit].to_numpy()
    tree = TreeNode.read(str(tree_file))
    tip_names = {n.name for n in tree.tips()}
    taxa_skbio = np.array([t.replace("_", " ", 1) for t in taxa])
    missing = sorted(set(taxa_skbio) - tip_names)
    if missing:
        raise ValueError(f"{len(missing)} taxa missing from the tree, e.g. {missing[:3]}")
    K = pcam._resolve_auto_blocks(K_prot, len(taxa), lo=6, hi=32)
    grp = gm.loc[samples].to_numpy()
    groups = list(pd.unique(grp))
    gs = {g: np.where(grp == g)[0] for g in groups}
    rows = [np.where(pcam.clade_assign(tree, taxa_skbio, K)[uid] == c)[0] for c in range(K)]
    L = np.log1p(abund)
    pall = np.concatenate([gs[g] for g in groups])
    grand = L[:, pall].mean(1)
    dev = {g: L[:, gs[g]].mean(1) - grand for g in groups}
    return dict(
        modality="protein", abund=abund, L=L, dev=dev, unit=unit, uid=uid,
        rows=rows, gs=gs, groups=groups,
        # Complement donor pool per group (indices), matching the current
        # pcam_gen.load_modality convention; for two groups this is exactly
        # gs[<other group>], so the two-group RNG stream is unchanged.
        other={g: np.concatenate([gs[h] for h in groups if h != g]) for g in groups},
        libs=abund[:, pall].sum(0),
        meta=df[["Taxon", "Function"]].reset_index(drop=True),
        tree_path=str(tree_file), post=None,
    )


def pxd_worker_init(group_file: str, table_file: str, tree_file: str) -> None:
    global _WORKER_DATA
    _load_analysis_modules()
    _WORKER_DATA = load_pxd_protein(Path(group_file), Path(table_file), Path(tree_file))


def pxd_worker_task(job: tuple) -> tuple:
    from phylopower import core
    from semisynthetic_power import summarize_distance_metrics_with_replacement

    core.load_core_runtime()
    pi, scale, pool_size, generation_seed, eval_n, boot, permutations, pilot_n, pilot_seed = job
    pilot = sorted_pilot_view(_WORKER_DATA, pilot_n, pilot_seed)
    table, group_map = pcam.pcam_pool(
        pilot, pool_size, generation_seed, pi, scale, ndon=1
    )
    distance = pcam.recompute_distance(pilot, table)
    omega2 = max(0.0, float(core.compute_omega2(distance, group_map)))
    metrics = summarize_distance_metrics_with_replacement(
        dm=distance,
        group_map=group_map,
        boot_number=boot,
        alpha=0.05,
        n_jobs=1,
        random_seed=7,
        n_per_group=eval_n,
        permutations=permutations,
        omega2_floor=0.0,
    )
    return pilot_n, pilot_seed, pi, scale, omega2, float(metrics["power"])


def pxd_simulate_rep(executor, args, pilot_n: int, rep: int) -> pd.DataFrame:
    pilot_seed = args.seed + pilot_n * 1009 + rep * 131
    jobs = []
    for gi, (pi, scale) in enumerate(args.grid):
        # Common random numbers across pilot sizes/repetitions reduce Monte
        # Carlo noise unrelated to the pilot subset itself.
        generation_seed = args.seed + 50_000 + gi * 9973
        jobs.append(
            (pi, scale, args.pool_size, generation_seed, args.eval_n,
             args.boot, args.permutations, pilot_n, pilot_seed)
        )
    records = []
    for pn, pseed, pi, scale, omega2, power in executor.map(pxd_worker_task, jobs):
        records.append(
            {
                "modality": "protein",
                "pilot_n": pn,
                "rep": rep,
                "pi": pi,
                "scale": scale,
                "true_omega2": omega2,
                "power": power,
                "eval_n": args.eval_n,
            }
        )
    return pd.DataFrame.from_records(records)


def pxd_simulate_one_size(executor, args, pilot_n: int, reps: int, cache: Path) -> pd.DataFrame:
    done = pd.DataFrame()
    if cache.exists():
        if args.force:
            cache.unlink()
        else:
            done = pd.read_csv(cache)
    done_reps = set(done["rep"].astype(int)) if len(done) else set()
    for rep in range(reps):
        if rep in done_reps:
            continue
        t0 = time.time()
        part = pxd_simulate_rep(executor, args, pilot_n, rep)
        part.to_csv(cache, mode="a", header=not cache.exists(), index=False)
        done_reps.add(rep)
        print(
            f"[pilot-info-pxd] pilot n={pilot_n} rep={rep} done in "
            f"{time.time() - t0:.0f}s ({len(done_reps)}/{reps})",
            flush=True,
        )
    out = pd.read_csv(cache)
    return out[out["rep"].astype(int) < reps]


def pxd_run_simulation(args: argparse.Namespace) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=pxd_worker_init,
        initargs=(str(args.group_file), str(args.table_file), str(args.tree_file)),
    ) as executor:
        parts = []
        for pilot_n in args.pilots:
            cache = args.out / f"raw_pilot_{pilot_n}.csv"
            print(
                f"[pilot-info-pxd] pilot n={pilot_n}, reps={args.reps}, "
                f"points={len(args.grid)}",
                flush=True,
            )
            parts.append(pxd_simulate_one_size(executor, args, pilot_n, args.reps, cache))

        # A full-cohort reference.  A pilot_n much larger than either group
        # causes sorted_pilot_view() to retain every empirical sample; the
        # generation seeds are identical to the pilot runs, so the reference
        # is an exact common-random-number match.
        ref_cache = args.out / "raw_full_cohort_reference.csv"
        parts.append(
            pxd_simulate_one_size(executor, args, 999, args.reference_reps, ref_cache)
        )

    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(args.out / "pilot_information_raw.csv", index=False)
    return raw


def pxd_summarize_and_plot(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

    _producer_apply_style()
    xmax = args.xmax
    x = np.linspace(0.0, xmax, 700)
    bin_width = args.bin_width

    ref_all = raw[raw["pilot_n"].eq(999)]
    # Common-random-number reference replicates are intentionally identical.
    # Fit one copy so duplicated rows do not change the curve fitter's weights.
    ref_rows = ref_all[ref_all["rep"].eq(ref_all["rep"].min())]
    reference = fit_curve(ref_rows, x, bin_width)
    if reference is None:
        raise RuntimeError("Could not fit the full-cohort reference curve")

    curves: dict[int, list[tuple[int, np.ndarray]]] = {}
    summary = []
    for pilot_n in args.pilots:
        curves[pilot_n] = []
        subset = raw[raw["pilot_n"].eq(pilot_n)]
        for rep, rows in subset.groupby("rep"):
            curve = fit_curve(rows, x, bin_width)
            if curve is None:
                continue
            curves[pilot_n].append((int(rep), curve))
            summary.append(
                {
                    "pilot_n": pilot_n,
                    "rep": int(rep),
                    "mean_abs_curve_error": float(np.trapezoid(np.abs(curve - reference), x) / xmax),
                    "omega2_at_80_power": crossing(x, curve, 0.8),
                    "reference_omega2_at_80_power": crossing(x, reference, 0.8),
                }
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out / "pilot_information_curve_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(args.pilots)}

    ax = axes[0, 0]
    ax.plot(x, reference, color="#202020", lw=3.0, label="Full-cohort reference", zorder=5)
    for pilot_n in args.pilots:
        stack = np.vstack([curve for _, curve in curves[pilot_n]])
        color = colors[pilot_n]
        ax.fill_between(
            x, np.percentile(stack, 10, axis=0), np.percentile(stack, 90, axis=0),
            color=color, alpha=0.16, lw=0,
        )
        ax.plot(x, np.median(stack, axis=0), color=color, lw=2.0, label=f"pilot n={pilot_n}")
    ax.set_title("Repeated-pilot power curves")
    ax.set_xlabel("Realized effect size (ω²)")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[0, 1]
    small, large = min(args.pilots), max(args.pilots)
    for pilot_n, linestyle in ((small, "-"), (large, "--")):
        color = colors[pilot_n]
        for j, (_, curve) in enumerate(curves[pilot_n]):
            ax.plot(
                x, curve, color=color, lw=1.0, alpha=0.38, ls=linestyle,
                label=f"pilot n={pilot_n}" if j == 0 else None,
            )
    ax.plot(x, reference, color="#202020", lw=3.0, label="Full-cohort reference", zorder=5)
    ax.set_title("Between-pilot disagreement")
    ax.set_xlabel("Realized effect size (ω²)")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1, 0]
    values = [
        summary_df.loc[summary_df["pilot_n"].eq(n), "mean_abs_curve_error"].dropna().to_numpy()
        for n in args.pilots
    ]
    bp = ax.boxplot(values, positions=np.arange(len(args.pilots)), widths=0.55, patch_artist=True)
    for patch, n in zip(bp["boxes"], args.pilots):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.40)
    rng = np.random.default_rng(args.seed)
    for i, (n, vals) in enumerate(zip(args.pilots, values)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=24, color=colors[n], alpha=0.80)
    ax.set_xticks(np.arange(len(args.pilots)), [str(n) for n in args.pilots])
    ax.set_title("Curve-to-reference disagreement")
    ax.set_xlabel("Pilot size per group")
    ax.set_ylabel("Mean absolute power difference")

    ax = axes[1, 1]
    threshold_values = [
        summary_df.loc[summary_df["pilot_n"].eq(n), "omega2_at_80_power"].dropna().to_numpy()
        for n in args.pilots
    ]
    bp = ax.boxplot(
        threshold_values, positions=np.arange(len(args.pilots)), widths=0.55, patch_artist=True
    )
    for patch, n in zip(bp["boxes"], args.pilots):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.40)
    for i, (n, vals) in enumerate(zip(args.pilots, threshold_values)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=24, color=colors[n], alpha=0.80)
    ref_cross = crossing(x, reference, 0.8)
    ax.axhline(ref_cross, color="#202020", lw=2.0, ls=":", label="Full-cohort reference")
    ax.set_xticks(np.arange(len(args.pilots)), [str(n) for n in args.pilots])
    ax.set_title("Stability of the 80% power threshold")
    ax.set_xlabel("Pilot size per group")
    ax.set_ylabel("ω² required for 80% power")
    ax.legend(loc="best", fontsize=9)

    for letter, ax in zip("abcd", axes.ravel()):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")
    for ax in axes.ravel()[:2]:
        ax.axhline(0.8, color="#7a8798", lw=1.1, ls=":")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.03)

    fig.suptitle(
        "Finite-pilot disagreement and convergence — Metaproteomics (PXD069517)",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(args.out / "pilot_information_supplement.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "pilot_information_supplement.pdf", bbox_inches="tight")
    plt.close(fig)

    medians = summary_df.groupby("pilot_n", as_index=False).median(numeric_only=True)
    rho, rho_p = spearmanr(summary_df["pilot_n"], summary_df["mean_abs_curve_error"])
    paired = summary_df.pivot(
        index="rep", columns="pilot_n", values="mean_abs_curve_error"
    )
    friedman = friedmanchisquare(*(paired[n] for n in args.pilots))
    pairwise = []
    for smaller, larger in zip(args.pilots[:-1], args.pilots[1:]):
        test = wilcoxon(
            paired[smaller], paired[larger], alternative="greater"
        )
        pairwise.append(
            {
                "smaller_pilot_n": smaller,
                "larger_pilot_n": larger,
                "paired_w": float(test.statistic),
                "one_sided_p": float(test.pvalue),
            }
        )
    report = {
        "modality": "protein",
        "dataset": "PRIDE PXD069517 (CD_only vs PolyAI_CD)",
        "eval_n": args.eval_n,
        "reps_requested": args.reps,
        "reference_omega2_at_80_power": ref_cross,
        "spearman_pilot_n_vs_curve_error": {
            "rho": float(rho),
            "p": float(rho_p),
        },
        "friedman_repeated_pilot_size_test": {
            "chi_square": float(friedman.statistic),
            "p": float(friedman.pvalue),
        },
        "adjacent_paired_wilcoxon_greater": pairwise,
        "median_metrics_by_pilot_n": medians.to_dict(orient="records"),
    }
    (args.out / "pilot_information_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def compute_protein(cli: argparse.Namespace) -> None:
    _compute_env()
    _load_analysis_modules()
    pis = [float(x) for x in cli.pis.split(",")]
    scales = [float(x) for x in cli.scales.split(",")]
    args = argparse.Namespace(
        group_file=cli.group_file,
        table_file=cli.table_file,
        tree_file=cli.tree_file,
        pilots=[int(x) for x in cli.protein_pilots.split(",")],
        reps=cli.protein_reps,
        reference_reps=cli.protein_reference_reps,
        eval_n=cli.protein_eval_n,
        pool_size=cli.protein_pool_size,
        boot=cli.protein_boot,
        permutations=cli.permutations,
        workers=cli.protein_workers,
        seed=cli.protein_seed,
        xmax=cli.protein_xmax,
        bin_width=cli.protein_bin_width,
        force=cli.force,
        plot_only=cli.plot_only,
        out=cli.protein_out,
        grid=[(pi, 1.0) for pi in pis] + [(1.0, s) for s in scales if s != 1.0],
    )
    raw_path = args.out / "pilot_information_raw.csv"
    if args.plot_only:
        raw = pd.read_csv(raw_path)
    else:
        raw = pxd_run_simulation(args)
    pxd_summarize_and_plot(raw, args)
    print(args.out / "pilot_information_supplement.png", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplementary Figure S1: plot archived pilot-information "
        "metrics, optionally recomputing the archives first (--compute)."
    )
    parser.add_argument(
        "--compute",
        choices=["gene", "protein", "all"],
        default=None,
        help="Recompute the corresponding pilot-information archive(s) "
        "verbatim, then plot. Default: plot only.",
    )
    # Shared compute flags (identical semantics in both retired producers).
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    # Gene-side knobs (defaults identical to the retired
    # analysis/run_pilot_information_supplement.py).
    parser.add_argument("--modality", choices=["gene", "protein"], default="gene")
    parser.add_argument("--gene-pilots", default="4,7,10")
    parser.add_argument("--gene-reps", type=int, default=4)
    parser.add_argument("--gene-reference-reps", type=int, default=2)
    parser.add_argument("--gene-eval-n", type=int, default=80)
    parser.add_argument("--gene-pool-size", type=int, default=100)
    parser.add_argument("--gene-boot", type=int, default=40)
    parser.add_argument("--gene-workers", type=int, default=6)
    parser.add_argument("--gene-seed", type=int, default=20260713)
    parser.add_argument("--gene-xmax", type=float, default=0.22)
    parser.add_argument("--gene-bin-width", type=float, default=0.008)
    parser.add_argument("--gene-out", type=Path, default=Path("data/pilot_information_supplement"))
    # Protein-side knobs (defaults identical to the retired
    # analysis/run_pilot_information_supplement_protein_pxd.py).
    parser.add_argument("--group-file", type=Path, default=PXD / "group.csv")
    parser.add_argument("--table-file", type=Path, default=PXD / "protein_taxon_function.csv")
    parser.add_argument("--tree-file", type=Path, default=PXD / "rooted-tree.nwk")
    parser.add_argument("--protein-pilots", default="5,7,10")
    parser.add_argument("--protein-reps", type=int, default=20)
    parser.add_argument("--protein-reference-reps", type=int, default=1)
    parser.add_argument("--protein-eval-n", type=int, default=14)
    parser.add_argument("--protein-pool-size", type=int, default=200)
    parser.add_argument("--protein-boot", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument(
        "--pis",
        default="0.5,0.6,0.7,0.8,0.9,1",
        help="Donor-mixing grid at scale=1 (pi=0.5 is the true null point).",
    )
    parser.add_argument(
        "--scales",
        default="1.15,1.3,1.5,1.7,2",
        help="Deviation-amplification grid at pi=1 (scale>1 amplifies the pilot's apparent difference).",
    )
    parser.add_argument("--protein-workers", type=int, default=4)
    parser.add_argument("--protein-seed", type=int, default=20260614)
    parser.add_argument("--protein-xmax", type=float, default=0.1)
    parser.add_argument("--protein-bin-width", type=float, default=0.005)
    parser.add_argument(
        "--protein-out",
        type=Path,
        default=Path("validation_datasets/results/PXD069517_pilot_information"),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Supplementary Figure S1 plotting (unchanged default behavior).
# ---------------------------------------------------------------------------


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


def plot() -> None:
    apply_style()
    gene = pd.read_csv(GENE_DATA)
    prot = pd.read_csv(PROT_DATA)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
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


def main() -> None:
    args = parse_args()
    if args.compute is None:
        # Fallback: the release ships no archived metrics tables, so when a
        # required plotting input is missing, recompute that side first (same
        # as --compute with default knobs) instead of dying with
        # FileNotFoundError.
        missing = [p for p in (GENE_DATA, PROT_DATA) if not p.exists()]
        if missing:
            args.compute = (
                "all" if len(missing) == 2 else ("gene" if missing[0] == GENE_DATA else "protein")
            )
            # The compute-side --gene-out/--protein-out defaults are
            # cwd-relative; anchor them so compute refreshes the exact files
            # plot() reads even when invoked outside the repo root.
            args.gene_out = GENE_DATA.parent
            args.protein_out = PROT_DATA.parent
            print(
                f"[suppfig1] archived data not found ({', '.join(str(m) for m in missing)}); "
                f"computing from scratch (--compute {args.compute}; this can take a while"
                + (" and needs the QIIME 2 env for gene" if args.compute in ("gene", "all") else "")
                + ") ...",
                flush=True,
            )
    if args.compute in ("gene", "all"):
        compute_gene(args)
    if args.compute in ("protein", "all"):
        compute_protein(args)
    plot()


if __name__ == "__main__":
    main()
