#!/usr/bin/env python3
"""Manuscript Figure 2 - Type-I error calibration (3 rows x 2 columns).

Rows are analysis questions; columns are data types. This lets the manuscript
cite panels in order: (a,b) QQ plots, (c,d) empirical-vs-nominal alpha,
(e,f) Type-I vs sample size, with metagenomics before metaproteomics in every
pair.

Default mode only plots from the archived null p-value table
(data/figdata/fig1_null_pvalues.csv); no simulations are run. If that table
is missing (e.g. a release checkout without the archived data), default mode
automatically falls back to the --compute path below and prints a notice.

With --compute, the null p-value simulations are re-run first (ported verbatim
from the retired producer analysis/produce_typeI_null_pvalues.py): gene uses
the PCAM raw-pool null and recomputes Gemelli; protein uses the MDC-TF-MC
raw-pool null and recomputes PhyloFunc; null p-values come from relabel
bootstrap PERMANOVA on the recomputed distance matrix. The long-form table
(modality, eval_n, pvalue) is written to data/figdata/fig1_null_pvalues.csv,
summary stats to <out>/fig1_summary.json, and the same 3x2 figure is then
plotted from the freshly computed p-values. Compute mode needs the QIIME 2
env for gene (Gemelli):
    /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig2_typeI.py --compute
On macOS, if the forked gene-side workers die with an Objective-C
"initialize may have been in progress in another thread when fork() was
called" crash, launch with OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from scipy.stats import beta

import figstyle

ROOT = Path(__file__).resolve().parents[1]
FIGDATA = ROOT / "data" / "figdata"
OUTDIR = ROOT / "figures" / "output"

figstyle.apply_style()

for _font_path in [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
]:
    if _font_path.exists():
        font_manager.fontManager.addfont(str(_font_path))

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16.0,
    "axes.titlesize": 17.5,
    "axes.labelsize": 16.0,
    "xtick.labelsize": 14.2,
    "ytick.labelsize": 14.2,
    "legend.fontsize": 11.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

GENE = "#756bb1"
PROTEIN = "#159c76"
REAL = "#22313f"
ACCENT = "#cc3333"
NEUTRAL = "#6b7280"
GRID = "#e5e7eb"
BAND = "#e9ecef"


def _blend(color: str, other: str = "#ffffff", amount: float = 0.5) -> tuple[float, float, float]:
    a = np.asarray(to_rgb(color), dtype=float)
    b = np.asarray(to_rgb(other), dtype=float)
    return tuple((1.0 - amount) * a + amount * b)


def _eval_colors(eval_ns: list[int], base: str) -> dict[int, tuple[float, float, float]]:
    steps = np.linspace(0.66, 0.0, len(eval_ns))
    return {int(en): _blend(base, "#ffffff", float(step)) for en, step in zip(sorted(eval_ns), steps)}


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.tick_params(length=3.5, width=0.85, colors="#4b5563")
    ax.grid(color=GRID, lw=0.65, alpha=0.72)


def _panel_title(ax, letter: str, title: str, color: str) -> None:
    ax.text(-0.12, 1.018, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=30, fontweight="bold", color="black")
    ax.text(0.035, 1.025, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=17.5, fontweight="bold", color="black")


def _load_cache(cache: Path, eval_ns: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    df = pd.read_csv(cache)
    out = {}
    for modality in ["gene", "protein"]:
        dat = {}
        for en, sub in df[df["modality"].eq(modality)].groupby("eval_n"):
            en_i = int(en)
            if en_i in eval_ns:
                dat[en_i] = sub["pvalue"].to_numpy(float)
        out[modality] = dat
    return out["gene"], out["protein"]


# ---------------------------------------------------------------------------
# Compute path (--compute): null p-value simulations, ported verbatim from the
# retired producer analysis/produce_typeI_null_pvalues.py. The old producer's
# plotting code (_row, _compact_row, its _panel_title/_eval_linestyles and
# suptitle layout) was discarded; only the computation below was carried over.
# Heavy deps (phylopower core finder, pcam_gen/Gemelli, MDC-TF-MC, skbio) are
# imported lazily so the default base-env plotting path never touches QIIME.
# ---------------------------------------------------------------------------

P = None  # pcam_gen, bound by _load_compute_deps()
mdctf_mc_pool = None  # from _protein_mdctf_mc, bound by _load_compute_deps()


def _load_compute_deps() -> None:
    """Bind the heavy compute-only modules (same idiom as the retired producer)."""
    global P, mdctf_mc_pool
    # Same wiring note as fig5_power_curves.compute_gene_power_curves: fork is
    # required so pcam_gen's ProcessPoolExecutor workers inherit the
    # fully-initialized modules; under macOS's default spawn start method the
    # gene-side pool dies with BrokenProcessPool.
    import multiprocessing as mp
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "figures"))  # shared figstyle
    from phylopower import core  # import first: installs the embedded-module finder
    core.load_core_runtime()
    import pcam_gen as _pcam_gen  # PCAM generator + parallel null-pvalue evaluator
    from _protein_mdctf_mc import mdctf_mc_pool as _pool

    P = _pcam_gen
    mdctf_mc_pool = _pool


def _pvals_for_gene(eval_ns, M, n_reps, n_pool_seeds, perms, seed0, n_workers) -> Dict[int, np.ndarray]:
    """Gene raw-pool null: PCAM pi=0.5 -> recompute Gemelli -> relabel bootstrap p-values."""
    out = P.null_pvalues("gene", eval_ns, M=M, n_pool_seeds=n_pool_seeds, n_reps=n_reps,
                         perms=perms, seed0=seed0, n_workers=n_workers)
    for en in eval_ns:
        print(f"   eval_n={en}: type-I@0.05 = {np.mean(out[en] < 0.05):.4f} (n={len(out[en])})", flush=True)
    return out


def _null_chunk(args) -> Dict[int, np.ndarray]:
    # Local import: ProcessPoolExecutor workers (spawn) re-import this module
    # without the compute-only deps, so skbio is imported here, not at top level.
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    arr, eval_ns, perms, n_chunk, seed = args
    rng = np.random.default_rng(int(seed))
    out = {int(en): [] for en in eval_ns}
    for en in eval_ns:
        labels = np.array(["A"] * int(en) + ["B"] * int(en), dtype=object)
        for rep in range(int(n_chunk)):
            pick = rng.choice(arr.shape[0], size=2 * int(en), replace=True)
            sub = arr[np.ix_(pick, pick)]
            sub = (sub + sub.T) / 2.0
            np.fill_diagonal(sub, 0.0)
            grp = labels.copy()
            rng.shuffle(grp)
            if np.allclose(sub, 0):
                out[int(en)].append(1.0)
                continue
            try:
                ids = [f"n{en}_{rep}_{i}" for i in range(sub.shape[0])]
                p = float(permanova(DistanceMatrix(sub, ids=ids), grp, permutations=perms)["p-value"])
                out[int(en)].append(1.0 if not np.isfinite(p) else p)
            except Exception:
                out[int(en)].append(1.0)
    return {en: np.asarray(vals, dtype=float) for en, vals in out.items()}


def _pvals_for_protein_mdctf(
    eval_ns, M, n_reps, n_pool_seeds, perms, seed0, edge_fraction,
    marginal_strength, eb_k, residual_mode, n_workers
) -> Dict[int, np.ndarray]:
    """Protein raw-pool null: MDC-TF-MC strength=0 -> recompute PhyloFunc -> relabel bootstrap p-values."""
    parts = []
    for ps in range(int(n_pool_seeds)):
        pool_seed = int(seed0 + ps * 7919)
        print(f"   protein null pool {ps + 1}/{n_pool_seeds} seed={pool_seed}", flush=True)
        d = P.load_modality("protein")
        tab, sgm = mdctf_mc_pool(
            d,
            M,
            pool_seed,
            0.0,
            edge_fraction=edge_fraction,
            marginal_strength=marginal_strength,
            eb_k=eb_k,
            residual_mode=residual_mode,
        )
        dm = P.recompute_distance(d, tab)
        ids = list(dm.index)
        arr = dm.loc[ids, ids].to_numpy(dtype=float)
        arr = (arr + arr.T) / 2.0
        np.fill_diagonal(arr, 0.0)
        n_workers = max(1, int(n_workers))
        chunks = [n_reps // n_workers] * n_workers
        for i in range(n_reps % n_workers):
            chunks[i] += 1
        jobs = [(arr, tuple(eval_ns), perms, n, pool_seed * 17 + 1009 * i) for i, n in enumerate(chunks) if n > 0]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            parts.extend(list(ex.map(_null_chunk, jobs)))
    out = {int(en): np.concatenate([p[int(en)] for p in parts]) for en in eval_ns}
    for en in eval_ns:
        print(f"   eval_n={en}: type-I@0.05 = {np.mean(out[en] < 0.05):.4f} (n={len(out[en])})", flush=True)
    return out


def _compute_null_pvalues(args) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray] | None]:
    """Re-run the null simulations and rewrite the long-form p-value table.

    Ported from the retired producer's main(): protein first (MDC-TF-MC null),
    then gene (PCAM null) unless --skip-gene, then the long-form CSV
    (modality, eval_n, pvalue).
    """
    _load_compute_deps()
    p_ev = [int(x) for x in args.protein_eval_ns.split(",") if x.strip()]
    g_ev = [int(x) for x in args.gene_eval_ns.split(",") if x.strip()]
    print("[fig2] protein (MDC-TF-MC null, PhyloFunc)...", flush=True)
    pdat = _pvals_for_protein_mdctf(
        p_ev, args.pool_M, args.n_reps, args.n_pool_seeds, args.permutations, args.seed,
        args.protein_edge_fraction, args.protein_marginal_strength, args.protein_eb_k,
        args.protein_residual_mode, args.n_workers,
    )
    gdat = None
    if not args.skip_gene:
        print("[fig2] gene (PCAM, Gemelli)...", flush=True)
        gdat = _pvals_for_gene(g_ev, args.pool_M, args.n_reps, args.n_pool_seeds,
                               args.permutations, args.seed, args.n_workers)
    # save the data table behind the figure (long form: modality, eval_n, pvalue) for re-use
    rows = []
    for modn, dat in [("protein", pdat), ("gene", gdat)]:
        if dat is None:
            continue
        for en, pv in dat.items():
            rows.append(pd.DataFrame({"modality": modn, "eval_n": en, "pvalue": pv}))
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.csv_out, index=False)
    print(f"[fig2] data table -> {args.csv_out}", flush=True)
    return pdat, gdat


def plot_qq(ax, pvals_by_n: dict[int, np.ndarray], qq_n: int, letter: str, title: str, color: str) -> None:
    pv = np.sort(pvals_by_n[qq_n])
    b = len(pv)
    expc = (np.arange(1, b + 1) - 0.5) / b
    lo = beta.ppf(0.025, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    hi = beta.ppf(0.975, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    ax.fill_between(expc, lo, hi, color=BAND, alpha=0.72, linewidth=0)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.15)
    ax.plot(expc, pv, ".", ms=2.4, color=color, alpha=0.78)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected null $P$")
    ax.set_ylabel("Observed null $P$")
    _panel_title(ax, letter, f"{title} QQ (n={qq_n})", color)
    _style_axis(ax)


def plot_alpha(ax, pvals_by_n: dict[int, np.ndarray], letter: str, title: str, color: str, show_legend: bool) -> None:
    eval_ns = sorted(pvals_by_n)
    color_of = _eval_colors(eval_ns, color)
    grid = np.linspace(0.001, 0.2, 120)
    pooled = []
    for en in eval_ns:
        vals = np.asarray([np.mean(pvals_by_n[en] < a) for a in grid], dtype=float)
        pooled.append(vals)
        ax.plot(grid, vals, color=color_of[en], lw=1.75, alpha=0.98, label=f"n={en}")
    if pooled:
        ax.plot(grid, np.mean(np.vstack(pooled), axis=0), color=REAL, lw=2.8, label="mean")
    ax.plot([0, 0.2], [0, 0.2], color=ACCENT, ls="--", lw=1.25)
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("Nominal $\\alpha$")
    ax.set_ylabel("Empirical Type I error")
    _panel_title(ax, letter, f"{title} alpha calibration", color)
    if show_legend:
        ax.legend(ncol=2, title="sample size", loc="upper left", frameon=False,
                  handlelength=2.15, columnspacing=1.15)
    _style_axis(ax)


def plot_size(ax, pvals_by_n: dict[int, np.ndarray], letter: str, title: str, color: str) -> dict[int, float]:
    eval_ns = sorted(pvals_by_n)
    xs = eval_ns
    ys = [float(np.mean(pvals_by_n[en] < 0.05)) for en in eval_ns]
    es = [1.96 * np.sqrt(y * (1 - y) / len(pvals_by_n[en])) for y, en in zip(ys, eval_ns)]
    ax.axhspan(0.025, 0.075, color=BAND, alpha=0.72, linewidth=0, zorder=0)
    ax.axhline(0.05, color=ACCENT, ls="--", lw=1.25, zorder=1)
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, ecolor=REAL, lw=2.05,
                ms=5.8, capsize=3.2, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Empirical Type I error at $\\alpha$=0.05")
    ax.set_ylim(0, 0.12)
    _panel_title(ax, letter, f"{title} sample-size check", color)
    ax.text(0.98, 0.69, "Bradley range", transform=ax.transAxes, ha="right", va="center",
            fontsize=12.0, color="#6b7280")
    _style_axis(ax)
    return {int(en): float(y) for en, y in zip(eval_ns, ys)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 2: type-I calibration, both modalities.")
    ap.add_argument("--cache", type=Path, default=FIGDATA / "fig1_null_pvalues.csv")
    ap.add_argument("--out", type=Path, default=OUTDIR)
    ap.add_argument("--eval-ns", default="6,10,17,30,50")
    ap.add_argument("--qq-n", type=int, default=30)
    ap.add_argument("--compute", action="store_true",
                    help="Re-run the null p-value simulations before plotting "
                         "(needs the QIIME 2 env for gene).")
    # Null-simulation knobs, ported verbatim (identical defaults) from the
    # retired producer analysis/produce_typeI_null_pvalues.py; only used with
    # --compute.
    ap.add_argument("--protein-eval-ns", default="6,10,17,30,50")
    ap.add_argument("--gene-eval-ns", default="6,10,17,30,50")
    ap.add_argument("--pool-M", type=int, default=120)
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--n-pool-seeds", type=int, default=6)
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--protein-edge-fraction", type=float, default=1.0)
    ap.add_argument("--protein-marginal-strength", default="auto")
    ap.add_argument("--protein-eb-k", default="auto")
    ap.add_argument("--protein-residual-mode", choices=["random", "template"], default="random")
    ap.add_argument("--n-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--skip-gene", action="store_true")
    ap.add_argument("--csv-out", type=Path, default=FIGDATA / "fig1_null_pvalues.csv",
                    help=argparse.SUPPRESS)  # test override; default matches the producer
    args = ap.parse_args()

    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    if not args.compute and not args.cache.exists():
        # Fallback: the release ships no archived CSVs, so default mode
        # recomputes first instead of dying with FileNotFoundError.
        print(
            f"[fig2] archived data not found ({args.cache}); computing from scratch "
            "(this can take a while and needs the QIIME 2 env for gene) ...",
            flush=True,
        )
        args.compute = True
    if args.compute:
        pdat, gdat = _compute_null_pvalues(args)
        fig1_summary = {
            "protein": {int(en): float(np.mean(pv < 0.05)) for en, pv in pdat.items()},
        }
        if gdat is not None:
            fig1_summary["gene"] = {
                int(en): float(np.mean(pv < 0.05)) for en, pv in gdat.items()
            }
        protein = {int(en): pv for en, pv in pdat.items() if int(en) in eval_ns}
        if gdat is not None:
            gene = {int(en): pv for en, pv in gdat.items() if int(en) in eval_ns}
        else:
            print("[fig2] --skip-gene: gene panels drawn from --cache", flush=True)
            gene, _ = _load_cache(args.cache, eval_ns)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "fig1_summary.json").write_text(
            json.dumps(fig1_summary, indent=2), encoding="utf-8"
        )
    else:
        gene, protein = _load_cache(args.cache, eval_ns)
        args.out.mkdir(parents=True, exist_ok=True)

    # Wider, manuscript-friendly 3 x 2 layout: each panel gets a wide aspect
    # ratio while the panel order keeps metagenomics before metaproteomics.
    fig, axes = plt.subplots(3, 2, figsize=(16.6, 9.4), squeeze=False)
    plot_qq(axes[0, 0], gene, args.qq_n, "a", "Metagenomics", GENE)
    plot_qq(axes[0, 1], protein, args.qq_n, "b", "Metaproteomics", PROTEIN)
    plot_alpha(axes[1, 0], gene, "c", "Metagenomics", GENE, show_legend=True)
    plot_alpha(axes[1, 1], protein, "d", "Metaproteomics", PROTEIN, show_legend=True)
    summary = {
        "metagenomics": plot_size(axes[2, 0], gene, "e", "Metagenomics", GENE),
        "metaproteomics": plot_size(axes[2, 1], protein, "f", "Metaproteomics", PROTEIN),
    }

    fig.tight_layout(rect=(0.035, 0.0, 0.995, 1.0), w_pad=3.0, h_pad=0.05)
    fig.savefig(args.out / "fig2_typeI_3x2_gene_first_wider_flat.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "fig2_typeI_3x2_gene_first_wider_flat.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.out / "fig2_typeI_3x2_gene_first_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(args.out / "fig2_typeI_3x2_gene_first_wider_flat.png")


if __name__ == "__main__":
    main()
