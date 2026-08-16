#!/usr/bin/env python3
"""Figure 1 — Type-I error calibration of the raw-pool power engine for BOTH modalities.

A single combined figure (2 rows x 3 panels):
    rows  = Protein (PhyloFunc, vst) | Gene (Gemelli)
    cols  = (a) null p-value QQ vs Uniform   (b) empirical vs nominal alpha   (c) type-I vs sample size

Under the null, a valid level-alpha test rejects with probability alpha. Gene uses the PCAM
raw-pool null and recomputes Gemelli. Protein uses the MDC-TF-MC raw-pool null and recomputes
PhyloFunc. Null p-values are obtained by relabel bootstrap on the recomputed distance matrix.

Run inside the QIIME 2 env (gene needs Gemelli); protein is pure Python:
    PATH=/opt/miniconda3/bin:$PATH /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python fig1.py \
        --out fig1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from scipy.stats import beta
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pcam_gen as P  # noqa: E402  (PCAM generator + parallel null-pvalue evaluator)
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
from phylopower import core  # noqa: E402
import figstyle  # noqa: E402

core.load_core_runtime()
figstyle.apply_style()
FIGDATA = ROOT / "figdata"

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
    "font.size": 12.2,
    "axes.titlesize": 13.0,
    "axes.labelsize": 12.2,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "legend.fontsize": 9.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

PROTEIN = "#159c76"
GENE = "#756bb1"
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


def _eval_linestyles(eval_ns: list[int]) -> dict[int, str | tuple[int, tuple[float, ...]]]:
    return {int(en): "-" for en in sorted(eval_ns)}


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(0.95)
    ax.spines["bottom"].set_linewidth(0.95)
    ax.tick_params(length=3.5, width=0.85, colors="#4b5563")
    ax.grid(color=GRID, lw=0.65, alpha=0.72)


def _panel_title(ax, letter: str, title: str) -> None:
    ax.text(-0.10, 1.045, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=17, fontweight="bold", color="black")
    ax.text(0.03, 1.055, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13.2, fontweight="bold", color="black")


def _pvals_for_gene(eval_ns, M, n_reps, n_pool_seeds, perms, seed0, n_workers) -> Dict[int, np.ndarray]:
    """Gene raw-pool null: PCAM pi=0.5 -> recompute Gemelli -> relabel bootstrap p-values."""
    out = P.null_pvalues("gene", eval_ns, M=M, n_pool_seeds=n_pool_seeds, n_reps=n_reps,
                         perms=perms, seed0=seed0, n_workers=n_workers)
    for en in eval_ns:
        print(f"   eval_n={en}: type-I@0.05 = {np.mean(out[en] < 0.05):.4f} (n={len(out[en])})", flush=True)
    return out


def _null_chunk(args) -> Dict[int, np.ndarray]:
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


def _compact_row(axes, pvals_by_n, qq_n, row_label, letters):
    eval_ns = sorted(pvals_by_n)
    ax = axes[0]
    pv = np.sort(pvals_by_n[qq_n])
    b = len(pv)
    expc = (np.arange(1, b + 1) - 0.5) / b
    lo = beta.ppf(0.025, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    hi = beta.ppf(0.975, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    ax.fill_between(expc, lo, hi, color=figstyle.BAND, alpha=0.65, linewidth=0)
    ax.plot([0, 1], [0, 1], color="#555555", ls="--", lw=1.0)
    ax.plot(expc, pv, ".", ms=2.1, color=figstyle.REAL, alpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected p under Uniform(0,1)")
    ax.set_ylabel("Observed null p")
    ax.set_title(f"{letters[0]}  QQ plot")
    ax.text(-0.18, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center",
            fontsize=11, fontweight="bold")

    ax = axes[1]
    xs = np.asarray(eval_ns, dtype=float)
    ys = np.asarray([float(np.mean(pvals_by_n[en] < 0.05)) for en in eval_ns])
    ns = np.asarray([len(pvals_by_n[en]) for en in eval_ns], dtype=float)
    es = 1.96 * np.sqrt(np.maximum(ys * (1 - ys), 1e-9) / ns)
    ax.axhspan(0.025, 0.075, color=figstyle.BAND, alpha=0.65, linewidth=0, zorder=0)
    ax.axhline(0.05, color="#555555", ls="--", lw=1.0, zorder=1)
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=figstyle.REAL, lw=1.8, ms=5, capsize=3, zorder=3)
    ax.set_xlabel("Target sample size per group")
    ax.set_ylabel("Empirical type-I at alpha=0.05")
    ax.set_ylim(0, 0.12)
    ax.set_title(f"{letters[1]}  Type-I calibration")
    ax.text(0.98, 0.66, "Bradley range", transform=ax.transAxes, ha="right", va="center",
            fontsize=7, color="#666666")
    return {int(en): float(np.mean(pvals_by_n[en] < 0.05)) for en in eval_ns}


def _row(axes, pvals_by_n, qq_n, row_label, letters, *, color: str, show_legend=False):
    eval_ns = sorted(pvals_by_n)
    color_of = _eval_colors(eval_ns, color)
    linestyle_of = _eval_linestyles(eval_ns)
    # (a) QQ
    ax = axes[0]
    pv = np.sort(pvals_by_n[qq_n]); b = len(pv)
    expc = (np.arange(1, b + 1) - 0.5) / b
    lo = beta.ppf(0.025, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    hi = beta.ppf(0.975, np.arange(1, b + 1), b - np.arange(1, b + 1) + 1)
    ax.fill_between(expc, lo, hi, color=BAND, alpha=0.72, linewidth=0, label="95% band")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1.15)
    ax.plot(expc, pv, ".", ms=2.4, color=color, alpha=0.78)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected null p")
    ax.set_ylabel("Observed null p")
    _panel_title(ax, letters[0], f"Null p-value QQ (n={qq_n})")
    ax.text(-0.23, 0.5, row_label, transform=ax.transAxes, rotation=90, ha="center", va="center",
            fontsize=12.6, fontweight="bold", color=color)
    _style_axis(ax)
    # (b) empirical vs nominal
    ax = axes[1]
    grid = np.linspace(0.001, 0.2, 120)
    pooled = []
    for en in eval_ns:
        vals = np.asarray([np.mean(pvals_by_n[en] < a) for a in grid], dtype=float)
        pooled.append(vals)
        ax.plot(grid, vals, color=color_of[en], ls=linestyle_of[en], lw=1.75, alpha=0.98, label=f"n={en}")
    if pooled:
        ax.plot(grid, np.mean(np.vstack(pooled), axis=0), color=REAL, lw=2.8, label="mean")
    ax.plot([0, 0.2], [0, 0.2], color=ACCENT, ls="--", lw=1.25)
    ax.set_xlim(0, 0.2); ax.set_ylim(0, 0.2)
    ax.set_xlabel("Nominal alpha")
    ax.set_ylabel("Empirical type-I")
    _panel_title(ax, letters[1], "Empirical vs nominal alpha")
    if show_legend:
        ax.legend(fontsize=8.7, ncol=2, title="sample size", loc="upper left", frameon=False,
                  handlelength=2.25, columnspacing=1.2)
    _style_axis(ax)
    # (c) type-I @0.05 vs eval_n with Bradley band
    ax = axes[2]
    xs = eval_ns
    ys = [float(np.mean(pvals_by_n[en] < 0.05)) for en in eval_ns]
    es = [1.96 * np.sqrt(y * (1 - y) / len(pvals_by_n[en])) for y, en in zip(ys, eval_ns)]
    ax.axhspan(0.025, 0.075, color=BAND, alpha=0.72, linewidth=0, zorder=0)
    ax.axhline(0.05, color=ACCENT, ls="--", lw=1.25, zorder=1)
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, ecolor=REAL, lw=2.05,
                ms=5.8, capsize=3.2, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Empirical type-I at alpha=0.05")
    ax.set_ylim(0, 0.12)
    _panel_title(ax, letters[2], "Type-I vs sample size")
    ax.text(0.98, 0.69, "Bradley range", transform=ax.transAxes, ha="right", va="center",
            fontsize=9.0, color="#6b7280")
    _style_axis(ax)
    return {int(en): float(np.mean(pvals_by_n[en] < 0.05)) for en in eval_ns}


def main(argv=None):
    p = argparse.ArgumentParser(description="Figure 1: type-I calibration, both modalities.")
    p.add_argument("--qiime-env", default="qiime2-metagenome-2024.10")
    p.add_argument("--protein-eval-ns", default="6,10,17,30,50")
    p.add_argument("--gene-eval-ns", default="6,10,17,30,50")
    p.add_argument("--protein-qq-n", type=int, default=30)
    p.add_argument("--gene-qq-n", type=int, default=30)
    p.add_argument("--pool-M", type=int, default=120)
    p.add_argument("--n-reps", type=int, default=200)
    p.add_argument("--n-pool-seeds", type=int, default=6)
    p.add_argument("--permutations", type=int, default=199)
    p.add_argument("--protein-edge-fraction", type=float, default=1.0)
    p.add_argument("--protein-marginal-strength", default="auto")
    p.add_argument("--protein-eb-k", default="auto")
    p.add_argument("--protein-residual-mode", choices=["random", "template"], default="random")
    p.add_argument("--n-workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--skip-gene", action="store_true")
    p.add_argument("--from-cache", action="store_true",
                   help="Redraw the figure from figdata/fig1_null_pvalues.csv without rerunning null simulations.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True); FIGDATA.mkdir(parents=True, exist_ok=True)
    p_ev = [int(x) for x in args.protein_eval_ns.split(",") if x.strip()]
    g_ev = [int(x) for x in args.gene_eval_ns.split(",") if x.strip()]

    if args.from_cache:
        cache_path = FIGDATA / "fig1_null_pvalues.csv"
        print(f"[fig1] redrawing from cache -> {cache_path}", flush=True)
        cache = pd.read_csv(cache_path)
        pdat = {
            int(en): sub["pvalue"].to_numpy(float)
            for en, sub in cache[cache["modality"].eq("protein")].groupby("eval_n")
            if int(en) in p_ev
        }
        gdat = None if args.skip_gene else {
            int(en): sub["pvalue"].to_numpy(float)
            for en, sub in cache[cache["modality"].eq("gene")].groupby("eval_n")
            if int(en) in g_ev
        }
    else:
        print("[fig1] protein (MDC-TF-MC null, PhyloFunc)...", flush=True)
        pdat = _pvals_for_protein_mdctf(
            p_ev, args.pool_M, args.n_reps, args.n_pool_seeds, args.permutations, args.seed,
            args.protein_edge_fraction, args.protein_marginal_strength, args.protein_eb_k,
            args.protein_residual_mode, args.n_workers
        )
        gdat = None
        if not args.skip_gene:
            print("[fig1] gene (PCAM, Gemelli)...", flush=True)
            gdat = _pvals_for_gene(g_ev, args.pool_M, args.n_reps, args.n_pool_seeds,
                                   args.permutations, args.seed, args.n_workers)
        # save the data table behind the figure (long form: modality, eval_n, pvalue) for re-use
        rows = []
        for modn, dat in [("protein", pdat), ("gene", gdat)]:
            if dat is None: continue
            for en, pv in dat.items():
                rows.append(pd.DataFrame({"modality": modn, "eval_n": en, "pvalue": pv}))
        pd.concat(rows, ignore_index=True).to_csv(FIGDATA / "fig1_null_pvalues.csv", index=False)
        print(f"[fig1] data table -> {FIGDATA}/fig1_null_pvalues.csv", flush=True)

    nrows = 1 if gdat is None else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(14.2, 4.15 * nrows), squeeze=False)
    summary = {
        "protein": _row(
            axes[0], pdat, args.protein_qq_n, "Protein\nMDC-TF-MC + PhyloFunc",
            ["a", "b", "c"], color=PROTEIN, show_legend=True,
        )
    }
    if gdat is not None:
        summary["gene"] = _row(
            axes[1], gdat, args.gene_qq_n, "Gene\nPCAM + Gemelli",
            ["d", "e", "f"], color=GENE, show_legend=True,
        )
    fig.suptitle("Type-I error calibration", y=1.0, fontsize=15.5, fontweight="bold")
    fig.tight_layout(rect=(0.025, 0, 1, 0.97), w_pad=2.2, h_pad=2.4)
    fig.savefig(args.out / "fig1.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "fig1.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.out / "fig1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[fig1] done -> {args.out}/fig1.png\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
