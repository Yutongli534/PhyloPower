#!/usr/bin/env python3
"""Figure 5 — Phylogenetic-uncertainty sensitivity of power (tree perturbation).

Same layout as the original tree-perturbation figures, re-run on the NEW pipeline
(protein_power_workflow / gene_power_workflow) with the new figstyle colours:

  Figure 5  (2x2 heatmaps):  rows = Gene (Gemelli) / Protein (PhyloFunc);
            cols = baseline omega^2 | baseline power, over a sigma x p_NNI grid.
            sigma = lognormal branch-length jitter; p_NNI = NNI topology-flip probability.
            Per grid point the tree is perturbed, the pilot distance is recomputed through the
            production pipeline, and omega^2 / power (scale=1, observed n) are evaluated.

  Figure 6  (2x1 curves):    a few representative (sigma, p_NNI) combos -> full omega^2-power
            sweep (pilot_curve) overlaid, so tree uncertainty's effect on the curve is visible.

Run in the QIIME env (gene needs Gemelli):
    PATH=/opt/miniconda3/bin:$PATH /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python fig5.py --out fig5
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from semisynthetic_power import (  # noqa: E402
    ID_COLS, _read_group_map, _read_protein_long_table, _read_taxon_feature_table,
    _build_ordination_model, _effective_pool_size, _ordination_pool_dm,
    _ordination_pool_group_map, summarize_distance_metrics_with_replacement, _subsample_group_map,
)
from phylopower import core
from logistic_fit import fit_logistic, logistic_curve  # noqa: E402
import protein_power_workflow as ppw  # noqa: E402  (NEW pipeline)
import gene_power_workflow as gpw      # noqa: E402  (NEW pipeline)
import figstyle  # noqa: E402

core.load_core_runtime()
figstyle.apply_style()

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
# light colormaps consistent with the figstyle palette (pale white -> figstyle orange / blue)
CMAP_O2 = LinearSegmentedColormap.from_list("o2_light", ["#ffffff", figstyle.SYN])
CMAP_PW = LinearSegmentedColormap.from_list("pw_light", ["#ffffff", figstyle.GROUP[0]])

PROTEIN_TABLE = core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv"
PROTEIN_GROUP = core.DATAPRO_DIR / "group.csv"
PROTEIN_TREE = core.DATAPRO_DIR / "rooted-tree.nwk"

REPRESENTATIVE = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.25), (0.75, 0.5), (1.0, 1.0)]


def _args(tree, sigma, nni, **over):
    a = SimpleNamespace(
        alpha=0.05, pool_size_per_group=1000, boot_number=80, permutations=99, n_jobs=1,
        omega2_floor=0.0, effect_grid="power-uniform", engine="ordination",
        ordination_enhance_max=3.0, omega_calibrated_max_scale=20.0, adaptive_reference_n=17.0,
        power_preview_boot_number=20, power_preview_permutations=49, omega_grid_candidates=40,
        fit_power_min=0.15, increase_num=12, decrease_num=12,
        center_mode="omega-calibrated", cov_estimator="ledoit-wolf", embed_dim=None,
        pool_cov=False, cov_eb_pool=False, pool_dist="student-t", pool_df="auto",
        protein_transform="vst", lowrank_rank=5, agg_clades=300,
        tree=tree, tree_jitter_sigma=float(sigma), tree_nni_prob=float(nni),
        tree_support_threshold=None, random_seed=20260616, use_phylogeny=True,
        qiime_env="qiime2-metagenome-2024.10", taxonomy=core.DATAGENE_DIR / "taxonomy.csv",
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _pilot_distance(modality, data, args, stem):
    if modality == "protein":
        return ppw.protein_pilot_distance(data, args)
    return gpw.gene_pilot_distance(data, args, stem)


def _baseline_omega2_power(modality, data, group_map, sigma, nni, qiime_env, seed):
    """Perturb tree -> pipeline distance -> baseline omega^2 and power (scale=1, observed n)."""
    tree = PROTEIN_TREE if modality == "protein" else core.DATAGENE_DIR / "rooted-tree.nwk"
    a = _args(tree, sigma, nni, qiime_env=qiime_env)
    try:
        dm = _pilot_distance(modality, data, a, f"hm_{sigma:.2f}_{nni:.2f}")
    except Exception:
        return sigma, nni, np.nan, np.nan
    common = dm.index.intersection(group_map.index)
    gm = group_map.loc[common]
    om = float(max(0.0, core.compute_omega2(dm.loc[common, common], gm)))
    obs_n = int(gm.value_counts().min())
    try:
        model = _build_ordination_model(dm, gm, center_mode="omega-calibrated",
            cov_estimator="ledoit-wolf", pool_dist="student-t", pool_df="auto")
        psize = _effective_pool_size(1000, obs_n)
        pgm = _ordination_pool_group_map(model, psize)
        pooldm = _ordination_pool_dm(model, 1.0, psize, seed)
        m = summarize_distance_metrics_with_replacement(dm=pooldm, group_map=pgm, boot_number=100,
            alpha=0.05, n_jobs=1, random_seed=seed, n_per_group=obs_n, permutations=99, omega2_floor=0.0)
        pw = float(m["power"])
    except Exception:
        pw = np.nan
    return sigma, nni, om, pw


def heatmap_grid(modality, data, group_map, sigmas, nnis, qiime_env, n_jobs, shard=None):
    """Compute the omega^2/power heatmap. shard=(k, N) computes only grid indices i%N==k (others NaN)
    so the gene grid can be split across independent PROCESSES (QIIME is not loky/thread-safe; each
    shard runs serially in its own process)."""
    grid = [(s, n) for s in sigmas for n in nnis]
    idxs = [i for i in range(len(grid)) if (shard is None or i % shard[1] == shard[0])]
    backend = "loky" if modality == "protein" else "threading"
    res = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_baseline_omega2_power)(modality, data, group_map, grid[i][0], grid[i][1], qiime_env, 7000 + i)
        for i in idxs)
    O = np.full((len(nnis), len(sigmas)), np.nan)
    P = np.full((len(nnis), len(sigmas)), np.nan)
    si = {round(s, 6): i for i, s in enumerate(sigmas)}
    ni = {round(n, 6): i for i, n in enumerate(nnis)}
    for s, n, om, pw in res:
        O[ni[round(n, 6)], si[round(s, 6)]] = om
        P[ni[round(n, 6)], si[round(s, 6)]] = pw
    print(f"[fig5] {modality} heatmap shard {shard} done ({len(idxs)} pts)", flush=True)
    return O, P


def _draw_heatmap(ax, data, title, cmap, fig, sigmas, nnis):
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto", origin="lower",
                   interpolation="nearest")
    step = max(1, len(sigmas) // 6)
    ax.set_xticks(range(0, len(sigmas), step)); ax.set_yticks(range(0, len(nnis), step))
    ax.set_xticklabels([f"{sigmas[i]:.2f}" for i in range(0, len(sigmas), step)], fontsize=8)
    ax.set_yticklabels([f"{nnis[i]:.2f}" for i in range(0, len(nnis), step)], fontsize=8)
    ax.set_xlabel(r"$\sigma$ (branch-length jitter)"); ax.set_ylabel(r"$p_{\mathrm{NNI}}$ (topology flip)")
    ax.set_title(title, fontweight="bold", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03); cb.ax.tick_params(labelsize=8)


def curves_panel(ax, modality, data, group_map, eval_n, qiime_env, xmax):
    cof, _, _ = figstyle.seq_colors(list(range(len(REPRESENTATIVE))))
    xg = np.linspace(0, xmax, 300)
    for li, (s, n) in enumerate(REPRESENTATIVE):
        a = _args(PROTEIN_TREE if modality == "protein" else core.DATAGENE_DIR / "rooted-tree.nwk",
                  s, n, qiime_env=qiime_env)
        try:
            dm = _pilot_distance(modality, data, a, f"cv_{s:.2f}_{n:.2f}")
            model = _build_ordination_model(dm, group_map, center_mode="omega-calibrated",
                cov_estimator="ledoit-wolf", pool_dist="student-t", pool_df="auto")
            df = (ppw if modality == "protein" else gpw).pilot_curve(model, a, 700 + li, eval_n)
        except Exception as e:
            print(f"[fig5] {modality} curve ({s},{n}) failed: {e}", flush=True); continue
        ax.scatter(df["true_omega2"], df["power"], s=10, color=cof[li], alpha=0.3, zorder=1)
        fit = fit_logistic(df.copy(), alpha=0.05); pr = fit.get("params")
        lw = 2.6 if (s == 0 and n == 0) else 1.5
        if pr:
            ax.plot(xg, logistic_curve(xg, pr["k"], pr["x0"], 0.05), color=cof[li],
                    lw=lw, zorder=2, label=f"σ={s:.2f}, p={n:.2f}")
        print(f"[fig5] {modality} curve σ={s:.2f} p={n:.2f} done", flush=True)
    ax.axhline(0.8, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlim(0, xmax); ax.set_ylim(-.03, 1.05)
    ax.set_xlabel(r"effect size ($\omega^2$)"); ax.set_ylabel("power")
    ax.set_title(f"{'Metaproteomics' if modality=='protein' else 'Metagenomics'} — tree-perturbation curves",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)


def load_modality(modality, protein_n, gene_n, seed):
    """Subsample the real data to the pilot size the original figure used (protein 7v7, gene 4v4)."""
    if modality == "protein":
        gm = _read_group_map(PROTEIN_GROUP)
        ldf, agm = _read_protein_long_table(PROTEIN_TABLE, gm)
        sub = _subsample_group_map(agm, protein_n, seed)
        return ldf[ID_COLS + list(sub.index)].copy(), sub
    ggm = _read_group_map(core.DATAGENE_DIR / "group.csv")
    gtbl, gagm = _read_taxon_feature_table(core.DATAGENE_DIR / "table.csv", ggm)
    sub = _subsample_group_map(gagm, gene_n, seed)
    return gtbl[list(sub.index)], sub


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--qiime-env", default="qiime2-metagenome-2024.10")
    p.add_argument("--grid", type=int, default=20, help="sigma/pNNI grid size (GxG).")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--protein-n", type=int, default=7)
    p.add_argument("--gene-n", type=int, default=4)
    p.add_argument("--protein-xmax", type=float, default=0.18)
    p.add_argument("--gene-xmax", type=float, default=0.60)
    p.add_argument("--seed", type=int, default=20260616)
    p.add_argument("--task", choices=["heatmap", "render", "all"], default="all")
    p.add_argument("--modality", choices=["protein", "gene"])
    p.add_argument("--shard", default=None, help="k/N: compute only grid indices i%%N==k.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_hm").mkdir(exist_ok=True)
    sigmas = np.linspace(0.0, 1.0, args.grid).tolist()
    nnis = np.linspace(0.0, 1.0, args.grid).tolist()

    # ---- TASK: heatmap shard (compute + save npz, no figure) ----
    if args.task == "heatmap":
        m = args.modality
        shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
        dat, gmap = load_modality(m, args.protein_n, args.gene_n, args.seed)
        O, P = heatmap_grid(m, dat, gmap, sigmas, nnis, args.qiime_env, args.n_jobs, shard=shard)
        tag = f"{shard[0]}_{shard[1]}" if shard else "full"
        np.savez(args.out / "_hm" / f"{m}_{tag}.npz", O=O, P=P)
        print(f"[fig5] saved shard {m} {tag}", flush=True)
        return

    # ---- TASK: render (merge shards + curves, assemble final figures) ----
    mods = ["gene", "protein"]

    def merged(m):
        O = np.full((len(nnis), len(sigmas)), np.nan); P = O.copy()
        for f in sorted((args.out / "_hm").glob(f"{m}_*.npz")):
            d = np.load(f)
            O = np.where(np.isfinite(d["O"]), d["O"], O)
            P = np.where(np.isfinite(d["P"]), d["P"], P)
        return O, P

    fig5, ax5 = plt.subplots(2, 2, figsize=(11, 9.6))
    for r, m in enumerate(mods):
        O, P = merged(m)
        lab = "Metagenomics (Gemelli)" if m == "gene" else "Metaproteomics (PhyloFunc)"
        _draw_heatmap(ax5[r][0], O, rf"{lab} — baseline $\omega^2$", CMAP_O2, fig5, sigmas, nnis)
        _draw_heatmap(ax5[r][1], P, f"{lab} — baseline power", CMAP_PW, fig5, sigmas, nnis)
        print(f"[fig5] {m} merged valid={np.isfinite(O).sum()}/{O.size}", flush=True)
    fig5.suptitle("Figure 5 — Phylogenetic-uncertainty sensitivity (tree perturbation)",
                  y=1.005, fontweight="bold", fontsize=13)
    fig5.tight_layout()
    fig5.savefig(args.out / "Figure5_heatmaps.png", bbox_inches="tight", dpi=200)
    plt.close(fig5)

    fig6, ax6 = plt.subplots(2, 1, figsize=(6.0, 9.0), squeeze=False)
    for r, m in enumerate(mods):
        dat, gmap = load_modality(m, args.protein_n, args.gene_n, args.seed)
        eval_n = int(gmap.value_counts().min())
        xmax = args.protein_xmax if m == "protein" else args.gene_xmax
        curves_panel(ax6[r][0], m, dat, gmap, eval_n, args.qiime_env, xmax)
    fig6.suptitle("Figure 6 — Power curves under tree perturbation", y=1.005,
                  fontweight="bold", fontsize=13)
    fig6.tight_layout()
    fig6.savefig(args.out / "Figure6_curves.png", bbox_inches="tight", dpi=200)
    plt.close(fig6)
    print(f"[fig5] done -> {args.out}/Figure5_heatmaps.png + Figure6_curves.png", flush=True)


if __name__ == "__main__":
    main()
