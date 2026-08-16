#!/usr/bin/env python3
"""Figure 4 — Power-curve consistency, extrapolation, and sample-size planning (PCAM pipeline).

Per modality (rows), 3 panels (same form as before, now PCAM: feature-space pool -> recomputed REAL
distance; effect swept via pi then scale):
  (a) omega^2-power at eval_n = N_truth, banded over pilot subsamples -> small pilot recovers the
      full-data curve (consistency).
  (b) omega^2-power by study size n (from the full pilot) -> the 0.8 crossing shifts left as n grows.
  (c) omega^2-power at eval_n=100, banded over pilots -> extrapolation to a large study.

    PATH=/opt/miniconda3/bin:$PATH /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python fig4.py --out fig4
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pcam_gen as P  # noqa: E402
from logistic_fit import fit_logistic, logistic_curve  # noqa: E402
import figstyle  # noqa: E402
figstyle.apply_style()
FIGDATA = ROOT / "figdata"

GRID = {"gene": [(0.5, 1), (0.6, 1), (0.68, 1), (0.75, 1), (0.82, 1), (0.88, 1), (0.93, 1), (0.97, 1),
                 (1.0, 1), (1.0, 1.3), (1.0, 1.7)],
        "protein": [(0.5, 1), (0.6, 1), (0.68, 1), (0.75, 1), (0.82, 1), (0.88, 1), (0.93, 1), (0.97, 1),
                    (1.0, 1), (1.0, 1.3), (1.0, 1.7), (1.0, 2.1)]}


def _band(ax, x, curves, color, label):
    C = np.vstack(curves); ax.plot(x, np.median(C, 0), color=color, lw=2, zorder=3, label=label)
    if C.shape[0] > 1:
        ax.fill_between(x, np.percentile(C, 10, 0), np.percentile(C, 90, 0), color=color, alpha=0.18, zorder=1)


def _fit_curve(rows, xg, floor=0.0):    # free fit (floor=0): curve starts where the data honestly is, not forced
    df = pd.DataFrame(rows, columns=["true_omega2", "power"]).dropna()
    if len(df) < 4: return None
    fit = fit_logistic(df, floor=floor); pr = fit.get("params")
    return logistic_curve(xg, pr["k"], pr["x0"], floor=floor) if pr else None


def run_modality(axes, modality, pilots, eval_truth, study_sizes, xmax, c_xmax, n_sub, seed0, M, boot):
    grid = GRID[modality]; cof, _, _ = figstyle.seq_colors(pilots)
    # ---- build all jobs ----
    jobs = []
    # (a)+(c): each pilot subsample, sweep grid, eval at (truth, 80)
    for pn in pilots:
        for r in range(n_sub):
            pseed = seed0 + pn * 1009 + r * 131
            for gi, (pi, sc) in enumerate(grid):
                jobs.append((pi, sc, M, pseed + 9000, (eval_truth, 80), boot, pn, pseed))
    # (b): full pilot (one subsample), sweep grid, eval at study sizes
    full = max(pilots); fseed = seed0 + full * 1009; bseed = fseed + 777  # distinct key so (b) doesn't collide with (a)+(c)
    for gi, (pi, sc) in enumerate(grid):
        jobs.append((pi, sc, M, bseed + 7000, tuple(study_sizes), boot, full, bseed))
    print(f"[fig4-{modality}] {len(jobs)} jobs...", flush=True)
    res = P.eval_pilot(modality, jobs, n_workers=6)

    # ---- assemble ----
    # index results: key (pn,pseed) -> list of (om, powers dict)
    from collections import defaultdict
    byp = defaultdict(list)
    for pn, pseed, pi, sc, om, powers in res:
        byp[(pn, pseed)].append((om, powers))
    xg = np.linspace(0, xmax, 300); xgc = np.linspace(0, c_xmax, 300)
    saved = []

    # (a) consistency @ eval_truth
    for pn in pilots:
        curves = []
        for r in range(n_sub):
            pseed = seed0 + pn * 1009 + r * 131
            rows = [(om, pw[eval_truth]) for om, pw in byp[(pn, pseed)]]
            c = _fit_curve(rows, xg)
            if c is not None: curves.append(c)
            for om, pw in byp[(pn, pseed)]:
                saved.append({"modality": modality, "panel": "a", "pilot": pn, "eval_n": eval_truth,
                              "true_omega2": om, "power": pw[eval_truth]})
        if curves: _band(axes[0], xg, curves, cof[pn], f"pilot {pn}")
        print(f"[fig4-{modality}] (a) pilot {pn} done", flush=True)
    axes[0].set_title(f"{modality} (a) consistency @ eval_n={eval_truth}")
    axes[0].set_xlabel("true omega^2"); axes[0].set_ylabel("power"); axes[0].set_xlim(0, xmax); axes[0].set_ylim(-.02, 1.03)
    axes[0].legend(fontsize=7)

    # (b) study-size family from full pilot
    full = max(pilots); fseed = seed0 + full * 1009; bseed = fseed + 777
    cofe, _, _ = figstyle.seq_colors(study_sizes)
    # collect (b) rows from the dedicated full-pilot jobs (keyed by bseed, separate from (a)+(c))
    brows = defaultdict(list)
    for pn, pseed, pi, sc, om, powers in res:
        if pn == full and pseed == bseed:
            for en in study_sizes:
                if en in powers: brows[en].append((om, powers[en]))
    for en in study_sizes:
        c = _fit_curve(brows[en], xg)
        if c is not None: axes[1].plot(xg, c, color=cofe[en], lw=2, label=f"n={en}")
        for om, pw in brows[en]:
            saved.append({"modality": modality, "panel": "b", "pilot": full, "eval_n": en, "true_omega2": om, "power": pw})
    axes[1].axhline(0.8, color=figstyle.NEUTRAL, ls=":", lw=1)
    axes[1].set_title(f"{modality} (b) power curves by study size n (pilot {full})")
    axes[1].set_xlabel("true omega^2"); axes[1].set_ylabel("power"); axes[1].set_xlim(0, xmax); axes[1].set_ylim(-.02, 1.03)
    axes[1].legend(fontsize=7, title="study n")

    # (c) eval_n=80 extrapolation, banded over pilots
    for pn in pilots:
        curves = []
        for r in range(n_sub):
            pseed = seed0 + pn * 1009 + r * 131
            rows = [(om, pw[80]) for om, pw in byp[(pn, pseed)]]
            c = _fit_curve(rows, xgc)
            if c is not None: curves.append(c)
            for om, pw in byp[(pn, pseed)]:
                saved.append({"modality": modality, "panel": "c", "pilot": pn, "eval_n": 80,
                              "true_omega2": om, "power": pw[80]})
        if curves: _band(axes[2], xgc, curves, cof[pn], f"pilot {pn}")
        print(f"[fig4-{modality}] (c) pilot {pn} done", flush=True)
    axes[2].axhline(0.8, color=figstyle.NEUTRAL, ls=":", lw=1)
    axes[2].set_title(f"{modality} (c) omega^2-power @ eval_n=80")
    axes[2].set_xlabel("true omega^2"); axes[2].set_ylabel("power"); axes[2].set_xlim(0, c_xmax); axes[2].set_ylim(-.02, 1.03)
    axes[2].legend(fontsize=7)
    return saved


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--protein-pilots", default="7,10,19")
    p.add_argument("--protein-eval-truth", type=int, default=17)
    p.add_argument("--protein-sizes", default="10,17,30,50,80")
    p.add_argument("--protein-xmax", type=float, default=0.30)
    p.add_argument("--protein-c-xmax", type=float, default=0.08)
    p.add_argument("--gene-pilots", default="4,7,10")
    p.add_argument("--gene-eval-truth", type=int, default=10)
    p.add_argument("--gene-sizes", default="6,10,30,50,80")
    p.add_argument("--gene-xmax", type=float, default=0.50)
    p.add_argument("--gene-c-xmax", type=float, default=0.18)
    p.add_argument("--n-sub", type=int, default=1)     # one pilot draw = a real study's single realization
    p.add_argument("--pool-M", type=int, default=300)  # large pool -> with-replacement bootstrap ~ i.i.d. at eval_n<=80
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--skip-gene", action="store_true")
    p.add_argument("--skip-protein", action="store_true")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True); FIGDATA.mkdir(parents=True, exist_ok=True)

    mods = []
    if not args.skip_protein:
        mods.append(("protein", [int(x) for x in args.protein_pilots.split(",")], args.protein_eval_truth,
                     [int(x) for x in args.protein_sizes.split(",")], args.protein_xmax, args.protein_c_xmax))
    if not args.skip_gene:
        mods.append(("gene", [int(x) for x in args.gene_pilots.split(",")], args.gene_eval_truth,
                     [int(x) for x in args.gene_sizes.split(",")], args.gene_xmax, args.gene_c_xmax))

    nrows = len(mods); fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.6 * nrows), squeeze=False)
    allrows = []
    for r, (modality, pilots, etr, sizes, xmax, cxmax) in enumerate(mods):
        allrows += run_modality(axes[r], modality, pilots, etr, sizes, xmax, cxmax,
                                args.n_sub, args.seed, args.pool_M, args.boot)
    fig.suptitle("Figure 4 — Power-curve consistency, extrapolation, and sample-size planning (PCAM)",
                 y=1.005, fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(args.out / "fig4.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(allrows).to_csv(FIGDATA / "fig4_power_curves.csv", index=False)
    print(f"[fig4] data table -> {FIGDATA}/fig4_power_curves.csv", flush=True)
    print(f"[fig4] done -> {args.out}/fig4.png", flush=True)


if __name__ == "__main__":
    main()
