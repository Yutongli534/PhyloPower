#!/usr/bin/env python3
"""Redraw the tree-error sample-size supplement in the Figure 5 layout.

Default mode only plots from the archived runs
(data/archived_runs/tree_error_fig5_like_quick/ and
data/archived_runs/tree_error_fig5_like_full25/); nothing is recomputed.

--compute {curves,midpoint,all} recomputes producer data first, then plots:

  curves    tree-error sample-size curve family, computation ported verbatim
            from the retired producer analysis/run_tree_error_sample_size_curves.py
            (writes tree_error_sample_size_curves.csv/.png/.pdf into --out,
            default ./tree_error_sample_size_curves as in the producer).

  midpoint  fixed-scale midpoint supplement rows for the bottom-row panels,
            computation ported verbatim from the retired producer
            analysis/run_fixed_scale_midpoint_supplement.py (writes
            selected_scales.csv + fixed_midpoint_rows.csv under
            --out-dir/<modality>/p<pilot>/, defaulting to the archived
            tree_error_fig5_like_quick/fixed_midpoint_supplement/).

  all       both.

Note: the study-size / pilot-consistency / pilot-extrapolation CSVs plotted
here were produced by the gene/protein power-workflow CLIs, not by the two
retired producers above; --compute refreshes only what those producers wrote.
Compute mode needs the QIIME 2 env for gene (Gemelli):
    /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig8_tree_error_power.py --compute all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _power_panels_gene as gene_fig
import _power_panels_protein as protein_fig

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "archived_runs" / "tree_error_fig5_like_quick"
FULL25_DIR = ROOT / "data" / "archived_runs" / "tree_error_fig5_like_full25"
OUTDIR = ROOT / "figures" / "output"
OUT_PNG = OUTDIR / "tree_error_fig5_like_3x2.png"
OUT_PDF = OUTDIR / "tree_error_fig5_like_3x2.pdf"


# ---------------------------------------------------------------------------
# Compute path (--compute): ported verbatim from the retired producers
# analysis/run_tree_error_sample_size_curves.py ("curves") and
# analysis/run_fixed_scale_midpoint_supplement.py ("midpoint").
# The curves producer imported PROTEIN_TABLE/PROTEIN_GROUP/PROTEIN_TREE and
# _pilot_distance from analysis/fig5.py, which no longer exists; those four
# symbols are ported here verbatim from analysis/produce_tree_error_heatmaps.py
# (its own source for them). Heavy deps (phylopower core finder,
# semisynthetic_power, the gene/protein power workflows) are imported lazily
# so the default base-env plotting path never touches QIIME.
# ---------------------------------------------------------------------------

core = None  # phylopower.core, bound by _load_compute_deps()
gpw = None  # gene_power_workflow (curves producer alias)
ppw = None  # protein_power_workflow (curves producer alias)
gene_wf = None  # gene_power_workflow (midpoint producer alias)
protein_wf = None  # protein_power_workflow (midpoint producer alias)
figstyle = None  # shared publication style (curves producer's plot_curves)
ID_COLS = _build_ordination_model = _effective_pool_size = None
_ordination_pool_dm = _ordination_pool_group_map = _ordination_scales_by_power_uniform = None
_read_group_map = _read_protein_long_table = _read_taxon_feature_table = _subsample_group_map = None
generate_taxon_function_pool = generate_taxon_pool = summarize_distance_metrics_with_replacement = None

PROTEIN_TABLE = PROTEIN_GROUP = PROTEIN_TREE = None
MODALITIES = None  # bound by _load_compute_deps() (needs core.DATAGENE_DIR)

TREE_SETTINGS = [
    ("original", 0.0, 0.0),
    ("tree_error_0.50_0.50", 0.5, 0.5),
]

BASE = DATA_DIR  # same path as the retired midpoint producer's BASE
PILOTS = {
    "gene": [4, 7, 10, 30, 50, 80],
    "protein": [7, 10, 17, 30, 50, 80],
}


def _load_compute_deps() -> None:
    """Bind the heavy compute-only modules (same idiom as the retired producers)."""
    global core, gpw, ppw, gene_wf, protein_wf, figstyle
    global ID_COLS, _build_ordination_model, _effective_pool_size
    global _ordination_pool_dm, _ordination_pool_group_map, _ordination_scales_by_power_uniform
    global _read_group_map, _read_protein_long_table, _read_taxon_feature_table, _subsample_group_map
    global generate_taxon_function_pool, generate_taxon_pool, summarize_distance_metrics_with_replacement
    global PROTEIN_TABLE, PROTEIN_GROUP, PROTEIN_TREE, MODALITIES
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp") / "numba_cache_phylopower"))
    os.environ["PATH"] = "/opt/miniconda3/bin:" + os.environ.get("PATH", "")
    try:
        import psutil

        if psutil.cpu_count() is None:
            psutil.cpu_count = lambda logical=True: 8
    except Exception:
        pass
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "analysis"))
    from phylopower import core as _core  # import first: installs the embedded-module finder
    _core.load_core_runtime()
    import figstyle as _figstyle
    import gene_power_workflow as _gpw
    import protein_power_workflow as _ppw
    import semisynthetic_power as _ssp
    for _name in (
        "ID_COLS", "_build_ordination_model", "_effective_pool_size",
        "_ordination_pool_dm", "_ordination_pool_group_map", "_ordination_scales_by_power_uniform",
        "_read_group_map", "_read_protein_long_table", "_read_taxon_feature_table",
        "_subsample_group_map", "generate_taxon_function_pool", "generate_taxon_pool",
        "summarize_distance_metrics_with_replacement",
    ):
        globals()[_name] = getattr(_ssp, _name)
    core = _core
    gpw = gene_wf = _gpw
    ppw = protein_wf = _ppw
    figstyle = _figstyle
    PROTEIN_TABLE = core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv"
    PROTEIN_GROUP = core.DATAPRO_DIR / "group.csv"
    PROTEIN_TREE = core.DATAPRO_DIR / "rooted-tree.nwk"
    MODALITIES = {
        "gene": {
            "label": "Metagenomics (Gemelli)",
            "pilot_n": 10,
            "eval_ns": [4, 7, 10, 30, 50, 80],
            "tree": core.DATAGENE_DIR / "rooted-tree.nwk",
            "xmax": 0.60,
        },
        "protein": {
            "label": "Metaproteomics (PhyloFunc)",
            "pilot_n": 17,
            "eval_ns": [7, 10, 17, 30, 50, 80],
            "tree": PROTEIN_TREE,
            "xmax": 0.18,
        },
    }


# --- curves (analysis/run_tree_error_sample_size_curves.py) -----------------


def _pilot_distance(modality, data, args, stem):
    if modality == "protein":
        return ppw.protein_pilot_distance(data, args)
    return gpw.gene_pilot_distance(data, args, stem)


def make_args(tree: Path, sigma: float, nni: float, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        alpha=0.05,
        target_power=0.80,
        pool_size_per_group=args.pool_m,
        boot_number=args.boot,
        permutations=args.permutations,
        n_jobs=1,
        omega2_floor=0.0,
        effect_grid="power-uniform",
        engine="ordination",
        ordination_enhance_max=3.0,
        omega_calibrated_max_scale=20.0,
        adaptive_reference_n=17.0,
        power_preview_boot_number=args.preview_boot,
        power_preview_permutations=args.preview_permutations,
        omega_grid_candidates=args.omega_grid_candidates,
        fit_power_min=0.15,
        increase_num=args.increase_num,
        decrease_num=args.decrease_num,
        center_mode="omega-calibrated",
        cov_estimator="ledoit-wolf",
        embed_dim=None,
        pool_cov=False,
        cov_eb_pool=False,
        pool_dist="student-t",
        pool_df="auto",
        protein_transform="vst",
        lowrank_rank=5,
        agg_clades=300,
        tree=tree,
        tree_jitter_sigma=float(sigma),
        tree_nni_prob=float(nni),
        tree_support_threshold=None,
        random_seed=args.seed,
        use_phylogeny=True,
        qiime_env=args.qiime_env,
        taxonomy=core.DATAGENE_DIR / "taxonomy.csv",
    )


def load_modality(modality: str, pilot_n: int, seed: int):
    if modality == "protein":
        group_map = _read_group_map(PROTEIN_GROUP)
        long_df, aligned_group_map = _read_protein_long_table(PROTEIN_TABLE, group_map)
        pilot_group = _subsample_group_map(aligned_group_map, pilot_n, seed)
        return long_df[ID_COLS + list(pilot_group.index)].copy(), pilot_group

    group_map = _read_group_map(core.DATAGENE_DIR / "group.csv")
    table, aligned_group_map = _read_taxon_feature_table(core.DATAGENE_DIR / "table.csv", group_map)
    pilot_group = _subsample_group_map(aligned_group_map, pilot_n, seed)
    return table[list(pilot_group.index)].copy(), pilot_group


def curve_fixed_pool(
    modality: str,
    model: dict,
    run_args: SimpleNamespace,
    eval_n: int,
    seed: int,
) -> pd.DataFrame:
    pool_group_map = _ordination_pool_group_map(model, run_args.pool_size_per_group)
    scales = _ordination_scales_by_power_uniform(
        run_args,
        eval_n,
        model,
        run_args.pool_size_per_group,
        pool_group_map,
        seed=seed,
        workflow="taxon" if modality == "gene" else "taxon-function",
    )
    rows = []
    for i, scale in enumerate(scales):
        dm = _ordination_pool_dm(
            model,
            float(scale),
            run_args.pool_size_per_group,
            seed + 9000 + i * 89,
        )
        metrics = summarize_distance_metrics_with_replacement(
            dm=dm,
            group_map=pool_group_map,
            boot_number=run_args.boot_number,
            alpha=run_args.alpha,
            n_jobs=1,
            random_seed=seed + i,
            n_per_group=int(eval_n),
            permutations=run_args.permutations,
            omega2_floor=run_args.omega2_floor,
        )
        rows.append(
            {
                "scale": float(scale),
                "true_omega2": float(metrics["true_omega2"]),
                "power": float(metrics["power"]),
                "mean_boot_omega2": float(metrics["mean_boot_omega2"]),
                "failed_bootstraps": int(metrics["failed_bootstraps"]),
                "mode": "dilution" if scale < 1.0 else "enhancement",
            }
        )
    return pd.DataFrame(rows)


def run_one(modality: str, setting: tuple[str, float, float], args: argparse.Namespace) -> pd.DataFrame:
    setting_label, sigma, nni = setting
    meta = MODALITIES[modality]
    pilot_seed = args.seed + int(meta["pilot_n"]) * 1009
    data, pilot_group = load_modality(modality, int(meta["pilot_n"]), pilot_seed)
    run_args = make_args(meta["tree"], sigma, nni, args)
    print(
        f"[tree-size] {modality} {setting_label}: pilot_n={meta['pilot_n']} "
        f"sigma={sigma:.2f} p_NNI={nni:.2f}",
        flush=True,
    )
    dm = _pilot_distance(modality, data, run_args, f"supp_{modality}_{setting_label}")
    common = dm.index.intersection(pilot_group.index)
    pilot_group = pilot_group.loc[common]
    dm = dm.loc[common, common]
    model = _build_ordination_model(
        dm,
        pilot_group,
        center_mode="omega-calibrated",
        cov_estimator="ledoit-wolf",
        pool_dist="student-t",
        pool_df="auto",
    )
    frames = []
    for eval_n in meta["eval_ns"]:
        df = curve_fixed_pool(
            modality,
            model,
            run_args,
            int(eval_n),
            seed=args.seed + int(eval_n) * 17 + int(1000 * sigma) + int(1000 * nni),
        )
        df.insert(0, "eval_n", int(eval_n))
        frames.append(df)
        n_transition = int(((df["power"] >= 0.15) & (df["power"] <= 0.95)).sum())
        print(
            f"[tree-size] {modality} {setting_label} eval_n={eval_n}: "
            f"{len(df)} points, transition={n_transition}",
            flush=True,
        )
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "setting", setting_label)
    out.insert(1, "sigma", float(sigma))
    out.insert(2, "p_nni", float(nni))
    out.insert(3, "modality", modality)
    out["pilot_n"] = int(meta["pilot_n"])
    out["pool_m_per_group"] = int(args.pool_m)
    out["boot"] = int(args.boot)
    out["permutations"] = int(args.permutations)
    return out


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    figstyle.apply_style()
    eval_order = {
        "gene": MODALITIES["gene"]["eval_ns"],
        "protein": MODALITIES["protein"]["eval_ns"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), squeeze=False)
    for row, modality in enumerate(["gene", "protein"]):
        meta = MODALITIES[modality]
        color_map, _, _ = figstyle.seq_colors(eval_order[modality])
        for col, (setting_label, sigma, nni) in enumerate(TREE_SETTINGS):
            ax = axes[row, col]
            sub_all = df[(df["modality"].eq(modality)) & (df["setting"].eq(setting_label))]
            for eval_n in eval_order[modality]:
                sub = sub_all[sub_all["eval_n"].eq(eval_n)].sort_values("true_omega2")
                if sub.empty:
                    continue
                ax.scatter(
                    sub["true_omega2"],
                    sub["power"],
                    s=16,
                    color=color_map[eval_n],
                    alpha=0.35,
                    linewidths=0,
                )
                sm = sub[["true_omega2", "power"]].dropna().sort_values("true_omega2")
                if len(sm) >= 2:
                    sm["power_mono"] = np.maximum.accumulate(sm["power"].to_numpy(float))
                    ax.plot(
                        sm["true_omega2"],
                        sm["power_mono"],
                        color=color_map[eval_n],
                        lw=2.1,
                        label=f"n={eval_n}",
                    )
            ax.axhline(0.8, color="#cc3333", ls="--", lw=1.1, alpha=0.7)
            ax.axhline(0.05, color="#666666", ls=":", lw=0.9, alpha=0.6)
            data_xmax = float(sub_all["true_omega2"].dropna().max()) if sub_all["true_omega2"].notna().any() else meta["xmax"]
            ax.set_xlim(0, max(meta["xmax"], data_xmax * 1.05))
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, color="#e5e7eb", lw=0.65, alpha=0.72)
            ax.set_xlabel(r"realized $\omega^2$")
            ax.set_ylabel("Power")
            ax.set_title(
                f"{meta['label']}\n"
                + (r"original tree" if setting_label == "original" else r"$\sigma=0.50$, $p_{\mathrm{NNI}}=0.50$"),
                fontweight="bold",
            )
            ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(w_pad=2.0, h_pad=1.4)
    fig.savefig(out_dir / "tree_error_sample_size_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "tree_error_sample_size_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def compute_curves(args: argparse.Namespace) -> None:
    """The retired curves producer's main()."""
    _load_compute_deps()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = []
    for modality in ["gene", "protein"]:
        for setting in TREE_SETTINGS:
            frames.append(run_one(modality, setting, args))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out / "tree_error_sample_size_curves.csv", index=False)
    plot_curves(df, args.out)
    print(f"[tree-size] wrote {args.out}", flush=True)


# --- midpoint (analysis/run_fixed_scale_midpoint_supplement.py) -------------


def default_args(modality: str, out: Path) -> SimpleNamespace:
    if modality == "gene":
        return SimpleNamespace(
            table=core.DATAGENE_DIR / "table.csv",
            tree=core.DATAGENE_DIR / "rooted-tree.nwk",
            taxonomy=core.DATAGENE_DIR / "taxonomy.csv",
            group=core.DATAGENE_DIR / "group.csv",
            use_phylogeny=True,
            qiime_env="qiime2-metagenome-2024.10",
            tree_jitter_sigma=0.5,
            tree_nni_prob=0.5,
            tree_support_threshold=None,
            random_seed=20260614,
            sim_seed=20260614,
            pool_size_per_group=300,
            boot_number=12,
            permutations=49,
            alpha=0.05,
            eval_n=80,
            center_mode="omega-calibrated",
            cov_estimator="ledoit-wolf",
            embed_dim=None,
            cov_eb_pool=False,
            pool_cov=False,
            pool_dist="student-t",
            pool_df="auto",
            ordination_enhance_max=3.0,
            omega_calibrated_max_scale=20.0,
            adaptive_reference_n=17.0,
            omega2_floor=0.0,
            n_jobs=1,
            pcam_gene_blocks=8,
            pcam_ndon=1,
            out=out,
        )
    return SimpleNamespace(
        table=core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv",
        tree=core.DATAPRO_DIR / "rooted-tree.nwk",
        group=core.DATAPRO_DIR / "group.csv",
        tree_jitter_sigma=0.5,
        tree_nni_prob=0.5,
        tree_support_threshold=None,
        random_seed=20260614,
        sim_seed=20260614,
        pool_size_per_group=160,
        boot_number=12,
        permutations=49,
        alpha=0.05,
        eval_n=80,
        center_mode="omega-calibrated",
        cov_estimator="ledoit-wolf",
        embed_dim=None,
        pool_cov=False,
        cov_eb_pool=False,
        protein_transform="vst",
        lowrank_rank=5,
        agg_clades=300,
        pool_dist="student-t",
        pool_df="auto",
        protein_generator="template-mask",
        omega2_floor=0.0,
        n_jobs=1,
        out=out,
    )


def existing_curve(modality: str, pilot: int) -> pd.DataFrame:
    if modality == "gene":
        path = BASE / "gene_pilot_extrapolation" / "gene_power_curves_raw.csv"
    else:
        path = BASE / "protein_pilot_extrapolation_refit" / "protein_power_curves_raw_combined.csv"
    df = pd.read_csv(path)
    return df[df["pilot_n"].eq(pilot)].copy().sort_values("scale")


def choose_midpoint_scales(df: pd.DataFrame, max_new: int) -> List[float]:
    rows = df.sort_values("scale")[["scale", "power"]].dropna().drop_duplicates("scale")
    vals = rows.to_numpy(float)
    candidates: List[tuple[float, float]] = []
    for left, right in zip(vals[:-1], vals[1:]):
        s0, p0 = left
        s1, p1 = right
        if s1 <= s0:
            continue
        in_window = (
            0.10 <= p0 <= 0.97
            or 0.10 <= p1 <= 0.97
            or (min(p0, p1) < 0.80 < max(p0, p1))
            or (min(p0, p1) < 0.50 < max(p0, p1))
        )
        if in_window:
            candidates.append((abs(p1 - p0), (s0 + s1) / 2.0))
            if abs(p1 - p0) >= 0.18:
                candidates.append((abs(p1 - p0) * 0.8, s0 + (s1 - s0) * 0.33))
                candidates.append((abs(p1 - p0) * 0.8, s0 + (s1 - s0) * 0.67))
    candidates.sort(reverse=True)
    existing = set(np.round(rows["scale"].to_numpy(float), 10))
    selected: List[float] = []
    for _score, scale in candidates:
        rounded = round(float(scale), 10)
        if rounded in existing or any(abs(float(scale) - s) < 1e-8 for s in selected):
            continue
        selected.append(float(scale))
        if len(selected) >= max_new:
            break
    return sorted(selected)


def build_gene_model_for_pilot(args: SimpleNamespace, pilot: int) -> Dict:
    group_map = _read_group_map(args.group)
    table, agm = _read_taxon_feature_table(args.table, group_map)
    observed_n = int(agm.value_counts().min())
    seed = int(args.random_seed) + pilot * 1009
    if pilot <= observed_n:
        rng = np.random.default_rng(seed)
        picked: List[str] = []
        for g in sorted(agm.unique()):
            members = agm[agm == g].index.to_numpy()
            picked.extend(rng.choice(members, size=pilot, replace=False).tolist())
        ptable = table[picked].copy()
        pgm = agm.loc[picked]
    else:
        ptable, pgm, _ = generate_taxon_pool(
            table,
            agm,
            pool_size_per_group=pilot,
            random_seed=int(args.sim_seed),
            between_scale=1.0,
            residual_scale=1.0,
            noise_multiplier=0.10,
        )
    return gene_wf.build_gene_model(ptable, pgm, args, stem=f"fixed_midpoint_pilot{pilot}")


def build_protein_model_for_pilot(args: SimpleNamespace, pilot: int) -> Dict:
    group_map = _read_group_map(args.group)
    long_df, agm = _read_protein_long_table(args.table, group_map)
    observed_n = int(agm.value_counts().min())
    seed = int(args.random_seed) + pilot * 1009
    if pilot <= observed_n:
        pgm = protein_wf._subsample_group_map(agm, pilot, seed)
        pdf = long_df[ID_COLS + list(pgm.index)].copy()
    else:
        pdf, pgm, _ = generate_taxon_function_pool(
            long_df,
            agm,
            pool_size_per_group=pilot,
            random_seed=int(args.sim_seed),
            between_scale=1.0,
            residual_scale=1.0,
            noise_multiplier=0.10,
            detection_slope=1.0,
            protein_generator=args.protein_generator,
        )
    return protein_wf.build_protein_model(pdf, pgm, args)


def compute_fixed_scales(model: Dict, args: SimpleNamespace, scales: Iterable[float], seed: int) -> pd.DataFrame:
    eval_n = int(args.eval_n)
    psize = _effective_pool_size(args.pool_size_per_group, eval_n)
    pool_gm = _ordination_pool_group_map(model, psize)
    rows = []
    for i, sc in enumerate(scales):
        dm = _ordination_pool_dm(model, float(sc), psize, seed + 19000 + i * 997)
        metrics = summarize_distance_metrics_with_replacement(
            dm=dm,
            group_map=pool_gm,
            boot_number=int(args.boot_number),
            alpha=float(args.alpha),
            n_jobs=int(args.n_jobs),
            random_seed=seed + 7000 + i,
            n_per_group=eval_n,
            permutations=int(args.permutations),
            omega2_floor=getattr(args, "omega2_floor", None),
        )
        rows.append(
            {
                "scale": float(sc),
                "true_omega2": float(metrics["true_omega2"]),
                "power": float(metrics["power"]),
                "mode": "fixed_midpoint",
            }
        )
    return pd.DataFrame(rows)


def _run_fixed_midpoint(modality: str, pilot: int, max_new: int | None, out_dir: Path) -> None:
    """One (modality, pilot) invocation of the retired midpoint producer's main()."""
    out_dir = out_dir / modality / f"p{pilot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = default_args(modality, out_dir)
    base = existing_curve(modality, pilot)
    max_new = int(max_new if max_new is not None else (10 if modality == "gene" else 12))
    scales = choose_midpoint_scales(base, max_new=max_new)
    if not scales:
        raise SystemExit(f"no midpoint scales selected for {modality} p={pilot}")
    pd.DataFrame({"scale": scales}).to_csv(out_dir / "selected_scales.csv", index=False)
    seed = int(args.random_seed) + pilot * 1009
    if modality == "gene":
        model = build_gene_model_for_pilot(args, pilot)
    else:
        model = build_protein_model_for_pilot(args, pilot)
    rows = compute_fixed_scales(model, args, scales, seed=seed)
    rows.insert(0, "pilot_n", int(pilot))
    rows.insert(1, "eval_n", int(args.eval_n))
    rows.to_csv(out_dir / "fixed_midpoint_rows.csv", index=False)
    print(f"[fixed] {modality} p={pilot}: {len(rows)} rows -> {out_dir / 'fixed_midpoint_rows.csv'}", flush=True)


def compute_midpoint(args: argparse.Namespace) -> None:
    """The retired midpoint producer's main(), looped over modalities/pilots
    (its --modality/--pilot were required; here the default runs all PILOTS)."""
    _load_compute_deps()
    modalities = [args.modality] if args.modality else ["gene", "protein"]
    for modality in modalities:
        pilots = [args.pilot] if args.pilot is not None else PILOTS[modality]
        for pilot in pilots:
            _run_fixed_midpoint(modality, pilot, args.max_new, args.out_dir)


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.105,
        1.035,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=25,
        fontweight="bold",
        color="black",
    )


def load_protein_extrapolation() -> pd.DataFrame:
    combined = DATA_DIR / "protein_pilot_extrapolation_refit" / "protein_power_curves_raw_combined.csv"
    if combined.exists():
        return pd.read_csv(combined)
    frames = []
    for pn in protein_fig.PILOT_KEYS:
        path = DATA_DIR / "protein_pilot_extrapolation_refit" / f"p{pn}" / "protein_power_curves_raw.csv"
        frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(combined, index=False)
    return out


def with_eval_n(df: pd.DataFrame, eval_n: int) -> pd.DataFrame:
    out = df.copy()
    if "eval_n" not in out.columns:
        out["eval_n"] = int(eval_n)
    return out


def append_fixed_midpoints(df: pd.DataFrame, modality: str, pilots: list[int]) -> pd.DataFrame:
    frames = [df.copy()]
    root = DATA_DIR / "fixed_midpoint_supplement" / modality
    for pn in pilots:
        path = root / f"p{pn}" / "fixed_midpoint_rows.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    out = pd.concat(frames, ignore_index=True)
    if "pilot" in out.columns and "pilot_n" not in out.columns:
        out["pilot_n"] = out["pilot"]
    out = out.drop_duplicates(["pilot_n", "scale"], keep="last").sort_values(["pilot_n", "scale"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure 8: tree-error sample-size supplement (Figure 5 layout).")
    ap.add_argument("--compute", choices=["curves", "midpoint", "all"], default=None,
                    help="Recompute producer data before plotting: 'curves' = tree-error "
                         "sample-size curves, 'midpoint' = fixed-scale midpoint supplement, "
                         "'all' = both (needs the QIIME 2 env for gene).")
    # Curves knobs, ported verbatim (identical defaults) from the retired
    # producer analysis/run_tree_error_sample_size_curves.py; only used with
    # --compute curves/all.
    ap.add_argument("--out", type=Path, default=Path("tree_error_sample_size_curves"))
    ap.add_argument("--qiime-env", default="qiime2-metagenome-2024.10")
    ap.add_argument("--pool-m", type=int, default=300)
    ap.add_argument("--boot", type=int, default=80)
    ap.add_argument("--permutations", type=int, default=99)
    ap.add_argument("--preview-boot", type=int, default=20)
    ap.add_argument("--preview-permutations", type=int, default=49)
    ap.add_argument("--omega-grid-candidates", type=int, default=30)
    ap.add_argument("--increase-num", type=int, default=8)
    ap.add_argument("--decrease-num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260616)
    # Midpoint knobs, ported verbatim (identical defaults) from the retired
    # producer analysis/run_fixed_scale_midpoint_supplement.py; only used with
    # --compute midpoint/all. Its --modality/--pilot were required; here they
    # default to looping over both modalities and all PILOTS.
    ap.add_argument("--modality", choices=["gene", "protein"], default=None)
    ap.add_argument("--pilot", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=BASE / "fixed_midpoint_supplement")
    args = ap.parse_args()

    if args.compute in ("curves", "all"):
        compute_curves(args)
    if args.compute in ("midpoint", "all"):
        compute_midpoint(args)

    gene_fig.apply_local_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 12.2,
            "legend.fontsize": 10.4,
        }
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)

    gene_study = pd.read_csv(DATA_DIR / "gene_study_size" / "gene_evalsweep_raw.csv")
    protein_study = pd.read_csv(DATA_DIR / "protein_study_size" / "protein_evalsweep_raw.csv")
    gene_consistency = with_eval_n(pd.read_csv(DATA_DIR / "gene_pilot_consistency" / "gene_power_curves_raw.csv"), 10)
    protein_consistency = with_eval_n(
        pd.read_csv(DATA_DIR / "protein_pilot_consistency" / "protein_power_curves_raw.csv"), 17
    )
    gene_full25 = FULL25_DIR / "gene_pilot_extrapolation" / "gene_power_curves_raw.csv"
    protein_full25 = FULL25_DIR / "protein_pilot_extrapolation" / "protein_power_curves_raw.csv"
    if gene_full25.exists():
        gene_extrapolation = with_eval_n(pd.read_csv(gene_full25), 80)
    else:
        gene_extrapolation = with_eval_n(pd.read_csv(DATA_DIR / "gene_pilot_extrapolation" / "gene_power_curves_raw.csv"), 80)
        gene_extrapolation = append_fixed_midpoints(gene_extrapolation, "gene", gene_fig.PILOTS_EXTENDED)
    if protein_full25.exists():
        protein_extrapolation = with_eval_n(pd.read_csv(protein_full25), 80)
    else:
        protein_extrapolation = with_eval_n(load_protein_extrapolation(), 80)
        protein_extrapolation = append_fixed_midpoints(protein_extrapolation, "protein", protein_fig.PILOT_KEYS)

    gene_consistency = gene_consistency.rename(columns={"pilot_n": "pilot"})
    gene_extrapolation = gene_extrapolation.rename(columns={"pilot_n": "pilot"})

    fig, axes = plt.subplots(3, 2, figsize=(16.7, 9.35), squeeze=False)

    gene_fig.draw_points_panel(
        axes[0, 0],
        gene_study[gene_study["eval_n"].isin(gene_fig.STUDY_SIZES)],
        by="eval_n",
        keys=gene_fig.STUDY_SIZES,
        colors=gene_fig.SIZE_COLORS,
        title="Metagenomics study-size family (pilot n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_points_panel(
        axes[0, 1],
        protein_study[protein_study["eval_n"].isin(protein_fig.PILOT_KEYS)],
        by="eval_n",
        keys=protein_fig.PILOT_KEYS,
        title="Metaproteomics study-size family (pilot n=17)",
        bin_width=0.004,
    )

    gene_fig.draw_band_panel(
        axes[1, 0],
        gene_consistency[(gene_consistency["pilot"].isin(gene_fig.PILOTS)) & (gene_consistency["eval_n"].eq(10))],
        by="pilot",
        keys=gene_fig.PILOTS,
        colors=gene_fig.PILOT_COLORS,
        title="Metagenomics pilot consistency (eval n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[1, 1],
        protein_consistency[
            (protein_consistency["pilot_n"].isin([7, 10, 17])) & (protein_consistency["eval_n"].eq(17))
        ],
        by="pilot_n",
        keys=[7, 10, 17],
        title="Metaproteomics pilot consistency (eval n=17)",
        bin_width=0.004,
        seed_base=3500,
    )

    gene_fig.draw_band_panel(
        axes[2, 0],
        gene_extrapolation[
            (gene_extrapolation["pilot"].isin(gene_fig.PILOTS_EXTENDED)) & (gene_extrapolation["eval_n"].eq(80))
        ],
        by="pilot",
        keys=gene_fig.PILOTS_EXTENDED,
        colors=gene_fig.PILOT_EXTENDED_COLORS,
        title="Metagenomics pilot extrapolation (eval n=80)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[2, 1],
        protein_extrapolation[
            (protein_extrapolation["pilot_n"].isin(protein_fig.PILOT_KEYS)) & (protein_extrapolation["eval_n"].eq(80))
        ],
        by="pilot_n",
        keys=protein_fig.PILOT_KEYS,
        title="Metaproteomics pilot extrapolation (eval n=80)",
        bin_width=0.003,
        seed_base=7600,
    )

    for letter, ax in zip("abcdef", axes.ravel()):
        add_panel_label(ax, letter)

    fig.tight_layout(rect=(0.018, 0.0, 0.998, 1.0), w_pad=2.4, h_pad=0.85)
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
