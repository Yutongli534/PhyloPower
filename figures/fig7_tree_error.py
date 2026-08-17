#!/usr/bin/env python3
"""Redraw manuscript Figure 7 as a 3 x 2 panel figure.

Rows are baseline omega-squared, baseline power, and refitted power curves.
Columns are metagenomics/Gemelli and metaproteomics/PhyloFunc.

Default mode only plots from the archived run
(data/archived_runs/fig5_rerun_20260701/: the _hm/*.npz heatmap shards and
fig7_tree_perturbation_curves_rerun.csv); nothing is recomputed. If any of
those inputs is missing (e.g. a release checkout without the archived data),
default mode automatically falls back to the --compute path below and prints
a notice.

With --compute, the archived run data is regenerated first (computation ported
verbatim from the retired producer analysis/produce_tree_error_heatmaps.py):
the sigma x p_NNI heatmap grid (_hm/<modality>_full.npz, or a shard with
--shard k/N) and the representative-combo refitted power curves
(fig7_tree_perturbation_curves_rerun.csv); the same 3x2 figure is then
plotted from the fresh data. Compute mode needs the QIIME 2 env for gene
(Gemelli):
    /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig7_tree_error.py --compute
"""
from __future__ import annotations

import argparse
import os
import sys  # noqa: E402
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import figstyle  # noqa: E402
from logistic_fit import fit_logistic, logistic_curve  # noqa: E402

INDIR = ROOT / "data" / "archived_runs" / "fig5_rerun_20260701"
OUTDIR = ROOT / "figures" / "output"
OUT_PNG = OUTDIR / "fig7_tree_error_3x2_abcdef.png"
OUT_PDF = OUTDIR / "fig7_tree_error_3x2_abcdef.pdf"

CMAP_O2 = LinearSegmentedColormap.from_list("o2_light", ["#ffffff", figstyle.SYN])
CMAP_PW = LinearSegmentedColormap.from_list("pw_light", ["#ffffff", figstyle.GROUP[0]])


# ---------------------------------------------------------------------------
# Compute path (--compute): tree-perturbation heatmaps + refitted power
# curves, ported verbatim from the retired producer
# analysis/produce_tree_error_heatmaps.py. The old producer's own Figure5/
# Figure6 rendering was discarded (this script's 3x2 plot is the renderer);
# the computation below is unchanged, except that the refitted power curves —
# which the old producer only plotted — are now also persisted to
# fig7_tree_perturbation_curves_rerun.csv (same columns as the archived CSV),
# since that table is this figure's plotting input.
# Heavy deps (phylopower core finder, semisynthetic_power, the gene/protein
# power workflows, joblib) are imported lazily so the default base-env
# plotting path never touches QIIME.
# ---------------------------------------------------------------------------

core = None  # phylopower.core, bound by _load_compute_deps()
gpw = None  # gene_power_workflow
ppw = None  # protein_power_workflow
Parallel = None  # joblib
delayed = None
ID_COLS = _read_group_map = _read_protein_long_table = _read_taxon_feature_table = None
_build_ordination_model = _effective_pool_size = _ordination_pool_dm = None
_ordination_pool_group_map = summarize_distance_metrics_with_replacement = _subsample_group_map = None

PROTEIN_TABLE = PROTEIN_GROUP = PROTEIN_TREE = None

REPRESENTATIVE = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.25), (0.75, 0.5), (1.0, 1.0)]


def _load_compute_deps() -> None:
    """Bind the heavy compute-only modules (same idiom as the retired producer)."""
    global core, gpw, ppw, Parallel, delayed
    global ID_COLS, _read_group_map, _read_protein_long_table, _read_taxon_feature_table
    global _build_ordination_model, _effective_pool_size, _ordination_pool_dm
    global _ordination_pool_group_map, summarize_distance_metrics_with_replacement, _subsample_group_map
    global PROTEIN_TABLE, PROTEIN_GROUP, PROTEIN_TREE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT))
    from joblib import Parallel as _Parallel, delayed as _delayed
    from phylopower import core as _core  # import first: installs the embedded-module finder
    _core.load_core_runtime()
    import gene_power_workflow as _gpw
    import protein_power_workflow as _ppw
    import semisynthetic_power as _ssp
    for _name in (
        "ID_COLS", "_read_group_map", "_read_protein_long_table", "_read_taxon_feature_table",
        "_build_ordination_model", "_effective_pool_size", "_ordination_pool_dm",
        "_ordination_pool_group_map", "summarize_distance_metrics_with_replacement",
        "_subsample_group_map",
    ):
        globals()[_name] = getattr(_ssp, _name)
    core = _core
    gpw = _gpw
    ppw = _ppw
    Parallel = _Parallel
    delayed = _delayed
    PROTEIN_TABLE = core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv"
    PROTEIN_GROUP = core.DATAPRO_DIR / "group.csv"
    PROTEIN_TREE = core.DATAPRO_DIR / "rooted-tree.nwk"


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
    print(f"[fig7] {modality} heatmap shard {shard} done ({len(idxs)} pts)", flush=True)
    return O, P


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


def refit_curve_rows(modality, data, group_map, eval_n, qiime_env):
    """The compute half of the retired producer's curves_panel: per-combo
    pilot_curve refits (its matplotlib calls dropped; rows returned instead)."""
    frames = []
    for li, (s, n) in enumerate(REPRESENTATIVE):
        a = _args(PROTEIN_TREE if modality == "protein" else core.DATAGENE_DIR / "rooted-tree.nwk",
                  s, n, qiime_env=qiime_env)
        try:
            dm = _pilot_distance(modality, data, a, f"cv_{s:.2f}_{n:.2f}")
            model = _build_ordination_model(dm, group_map, center_mode="omega-calibrated",
                cov_estimator="ledoit-wolf", pool_dist="student-t", pool_df="auto")
            df = (ppw if modality == "protein" else gpw).pilot_curve(model, a, 700 + li, eval_n)
        except Exception as e:
            print(f"[fig7] {modality} curve ({s},{n}) failed: {e}", flush=True)
            continue
        df = df.copy()
        df["modality"] = modality
        df["combo_index"] = li
        df["sigma"] = float(s)
        df["nni"] = float(n)
        df["eval_n"] = int(eval_n)
        frames.append(df)
        print(f"[fig7] {modality} curve sigma={s:.2f} p={n:.2f} done", flush=True)
    return pd.concat(frames, ignore_index=True)


def compute_tree_error(args) -> None:
    """Regenerate the archived run data (the retired producer's main, render task aside)."""
    _load_compute_deps()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_hm").mkdir(exist_ok=True)
    sigmas = np.linspace(0.0, 1.0, args.grid).tolist()
    nnis = np.linspace(0.0, 1.0, args.grid).tolist()
    mods = [args.modality] if args.modality else ["gene", "protein"]

    if args.task in ("heatmap", "all"):
        shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
        for m in mods:
            dat, gmap = load_modality(m, args.protein_n, args.gene_n, args.seed)
            O, P = heatmap_grid(m, dat, gmap, sigmas, nnis, args.qiime_env, args.n_jobs, shard=shard)
            tag = f"{shard[0]}_{shard[1]}" if shard else "full"
            np.savez(args.out / "_hm" / f"{m}_{tag}.npz", O=O, P=P)
            print(f"[fig7] saved shard {m} {tag}", flush=True)

    if args.task in ("curves", "all"):
        frames = []
        for m in mods:
            dat, gmap = load_modality(m, args.protein_n, args.gene_n, args.seed)
            eval_n = int(gmap.value_counts().min())
            frames.append(refit_curve_rows(m, dat, gmap, eval_n, args.qiime_env))
        curves_out = args.out / "fig7_tree_perturbation_curves_rerun.csv"
        pd.concat(frames, ignore_index=True).to_csv(curves_out, index=False)
        print(f"[fig7] curves -> {curves_out}", flush=True)


def apply_style() -> None:
    figstyle.apply_style()
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 15.0,
            "axes.titlesize": 17.5,
            "axes.labelsize": 16.0,
            "xtick.labelsize": 13.8,
            "ytick.labelsize": 13.8,
            "legend.fontsize": 11.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
        }
    )


def merged_heatmap(modality: str, key: str) -> np.ndarray:
    sigmas = np.linspace(0.0, 1.0, 20)
    nnis = np.linspace(0.0, 1.0, 20)
    out = np.full((len(nnis), len(sigmas)), np.nan)
    for path in sorted((INDIR / "_hm").glob(f"{modality}_*.npz")):
        arr = np.load(path)[key]
        out = np.where(np.isfinite(arr), arr, out)
    return out


def draw_heatmap(ax, data: np.ndarray, title: str, cmap, fig) -> None:
    sigmas = np.linspace(0.0, 1.0, data.shape[1])
    nnis = np.linspace(0.0, 1.0, data.shape[0])
    im = ax.imshow(
        np.ma.masked_invalid(data),
        cmap=cmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    ticks = np.linspace(0, len(sigmas) - 1, 6, dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{sigmas[i]:.2f}" for i in ticks])
    ax.set_yticklabels([f"{nnis[i]:.2f}" for i in ticks])
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$p_{\mathrm{NNI}}$")
    ax.set_title(title, fontweight="bold", color="black", pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.76, pad=0.025)
    cb.ax.tick_params(labelsize=12.2)


def draw_curves(ax, curves: pd.DataFrame, modality: str, title: str, xmax: float) -> None:
    colors, _, _ = figstyle.seq_colors(sorted(curves["combo_index"].unique()))
    xg = np.linspace(0, xmax, 300)
    for i, combo in enumerate(sorted(curves["combo_index"].unique())):
        sub = curves[(curves["modality"].eq(modality)) & (curves["combo_index"].eq(combo))].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("true_omega2")
        sigma = float(sub["sigma"].iloc[0])
        nni = float(sub["nni"].iloc[0])
        color = colors[i % len(colors)]
        ax.scatter(sub["true_omega2"], sub["power"], s=13, color=color, alpha=0.34, zorder=2)
        fit = fit_logistic(sub[["true_omega2", "power"]].copy(), alpha=0.05)
        params = fit.get("params")
        if params:
            y = logistic_curve(xg, params["k"], params["x0"], 0.05)
            lw = 2.65 if sigma == 0.0 and nni == 0.0 else 1.75
            ax.plot(xg, y, color=color, lw=lw, label=rf"$\sigma$={sigma:.2f}, $p_{{NNI}}$={nni:.2f}")
        else:
            ax.plot(sub["true_omega2"], sub["power"], color=color, lw=1.6, label=rf"$\sigma$={sigma:.2f}, $p_{{NNI}}$={nni:.2f}")
    ax.axhline(0.8, color="#cc3333", ls="--", lw=1.2, alpha=0.72)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("ω²")
    ax.set_ylabel("Power")
    ax.set_title(title, fontweight="bold", color="black", pad=8)
    ax.grid(True, color="#e5e7eb", lw=0.65, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
        fontweight="bold",
        color="black",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 7: tree-error sensitivity, both modalities.")
    ap.add_argument("--compute", action="store_true",
                    help="Recompute the archived run data (heatmap grid + refitted curves) "
                         "before plotting (needs the QIIME 2 env for gene).")
    # Compute knobs, ported verbatim (identical defaults) from the retired
    # producer analysis/produce_tree_error_heatmaps.py; only used with
    # --compute (except the xmax pair, which also set the curve-panel x
    # limits). The producer's --out was required; here it defaults to the
    # archived run dir so default-knob compute refreshes it in place. Its
    # --task choice "render" is renamed "curves" (rendering is this script's
    # plotting path, which always runs after compute unless --task heatmap).
    ap.add_argument("--qiime-env", default="qiime2-metagenome-2024.10")
    ap.add_argument("--grid", type=int, default=20, help="sigma/pNNI grid size (GxG).")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--protein-n", type=int, default=7)
    ap.add_argument("--gene-n", type=int, default=4)
    ap.add_argument("--protein-xmax", type=float, default=0.18)
    ap.add_argument("--gene-xmax", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--task", choices=["heatmap", "curves", "all"], default="all")
    ap.add_argument("--modality", choices=["protein", "gene"])
    ap.add_argument("--shard", default=None, help="k/N: compute only grid indices i%%N==k.")
    ap.add_argument("--out", type=Path, default=INDIR,
                    help="compute output dir (default: the archived run dir).")
    args = ap.parse_args()

    if not args.compute:
        # Fallback: the release ships no archived run data, so when a
        # required plotting input (the refitted-curves CSV, or the per-modality
        # heatmap shards) is missing, recompute first instead of dying with
        # FileNotFoundError.
        hm_dir = INDIR / "_hm"
        missing = []
        if not (INDIR / "fig7_tree_perturbation_curves_rerun.csv").exists():
            missing.append(INDIR / "fig7_tree_perturbation_curves_rerun.csv")
        for modality in ("gene", "protein"):
            if not any(hm_dir.glob(f"{modality}_*.npz")):
                missing.append(hm_dir / f"{modality}_*.npz")
        if missing:
            print(
                f"[fig7] archived data not found ({', '.join(str(m) for m in missing)}); "
                "computing from scratch (--compute with default knobs: a 20x20 grid for both "
                "modalities plus the refitted curves; this can take a long time and needs the "
                "QIIME 2 env for gene) ...",
                flush=True,
            )
            args.compute = True
            args.out = INDIR  # compute must refresh the directory the plot reads

    if args.compute:
        compute_tree_error(args)
        if args.task == "heatmap":
            return

    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(INDIR / "fig7_tree_perturbation_curves_rerun.csv")

    modalities = [
        ("gene", "Metagenomics", "Gemelli", args.gene_xmax),
        ("protein", "Metaproteomics", "PhyloFunc", args.protein_xmax),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(16.7, 9.6), squeeze=False)
    for c, (modality, omics_label, metric_label, xmax) in enumerate(modalities):
        draw_heatmap(
            axes[0, c],
            merged_heatmap(modality, "O"),
            f"{omics_label} baseline ω² ({metric_label})",
            CMAP_O2,
            fig,
        )
        draw_heatmap(
            axes[1, c],
            merged_heatmap(modality, "P"),
            f"{omics_label} baseline power ({metric_label})",
            CMAP_PW,
            fig,
        )
        draw_curves(
            axes[2, c],
            curves,
            modality,
            f"{omics_label} refitted power curves ({metric_label})",
            xmax,
        )

    for letter, ax in zip("abcdef", axes.ravel()):
        add_panel_label(ax, letter)

    fig.tight_layout(rect=(0.022, 0.0, 0.998, 1.0), w_pad=2.35, h_pad=0.9)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
