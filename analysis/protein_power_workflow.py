#!/usr/bin/env python3
"""Complete PROTEIN (taxon-function / metaproteomic) power-analysis workflow.

Default pipeline:

    real pilot taxon-function table
      -> MDC-TF Taxon-Function structure-preserving raw synthetic sample pool
      -> recompute PhyloFunc distance for every effect level
      -> with-replacement bootstrap at the target sample size -> PERMANOVA -> power
      -> binned monotone curve fit starting from the empirical null point

Legacy ordination mode is still available with ``--engine ordination``:

    real pilot taxon-function table
      -> PhyloFunc distance                                 [modality-specific, pure Python]
      -> PCoA embedding (Euclidean coords reproduce the distance)
      -> per-group centroid + Ledoit-Wolf shrunk covariance, omega-calibrated centroid
      -> POWER-UNIFORM effect grid (two-stage preview + invert power(scale))
      -> large MVN synthetic pool (fresh points; no duplicate-zero inflation)
      -> with-replacement bootstrap at the target sample size -> PERMANOVA -> power
      -> free logistic fit of (omega^2, power)  [not anchored to (0, alpha)]

In raw-pool mode, pilot sizes larger than the observed per-group count are SIMULATED first with
MDC-TF, then run through the same pipeline, so any target pilot size can be evaluated.

Example:
    python protein_power_workflow.py --pilot-ns 5,10,17,25,40 --eval-n 50 --target-omega2 0.05 \
        --out protein_workflow_out
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pcam_gen as P  # noqa: E402
from _fig4_curve_plotting import draw_binned_null_hill_group, null_started_hill_sigmoid  # noqa: E402
from _protein_mdctf_curve import mdctf_pool  # noqa: E402
from _protein_mdctf_optimized_curve import _exchangeable_null_power  # noqa: E402
from semisynthetic_power import (  # noqa: E402
    ID_COLS,
    _as_distance_frame,
    _build_ordination_model,
    _effective_pool_size,
    _ordination_pool_dm,
    _ordination_pool_group_map,
    _ordination_scales_by_power_uniform,
    _read_group_map,
    _read_protein_long_table,
    _subsample_group_map,
    generate_taxon_function_pool,
    summarize_distance_metrics_with_replacement,
)
from phylopower import core
from logistic_fit import fit_logistic, logistic_curve  # noqa: E402

core.load_core_runtime()


def _resolve_auto_edge_fraction(value: float | str) -> float:
    if str(value).strip().lower() == "auto":
        return 1.0
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("edge_fraction must be in [0, 1] or 'auto'.")
    return resolved


def _effective_tree_path(args: argparse.Namespace, work: Path, stem: str) -> str:
    """Return the tree path to use, optionally branch-length-jittered (sigma) and/or NNI-perturbed
    (nni_prob) for phylogenetic-uncertainty sensitivity analysis. sigma=nni_prob=0 -> original tree."""
    sigma = float(getattr(args, "tree_jitter_sigma", 0.0))
    nni = float(getattr(args, "tree_nni_prob", 0.0))
    if sigma <= 0.0 and nni <= 0.0:
        return str(args.tree)
    from skbio import TreeNode
    base_tree = TreeNode.read(str(args.tree))
    pth = core.materialize_perturbed_tree(
        base_tree, sigma=sigma, nni_prob=nni, random_seed=int(getattr(args, "random_seed", 0)),
        out_dir=work, output_stem=f"{stem}_treepert", output_format="newick",
        support_threshold=getattr(args, "tree_support_threshold", None))
    return str(pth) if pth is not None else str(args.tree)


def protein_pilot_distance(long_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """The pilot PhyloFunc distance exactly as the pipeline computes it (transform + tree)."""
    import tempfile as _tf
    with _tf.TemporaryDirectory(prefix="protein_tree_") as _tmp:
        tree_path = _effective_tree_path(args, Path(_tmp), "protein")   # perturbed if sigma/nni > 0
        if getattr(args, "protein_transform", "none") != "none":
            from protein_transforms import transform_table
            long_df = transform_table(long_df, args.protein_transform, tree_path,
                                      rank=args.lowrank_rank, n_clades=args.agg_clades)
        return _as_distance_frame(core.compute_phylofunc_distance_matrix(long_df, tree_path))


def build_protein_model(long_df: pd.DataFrame, group_map: pd.Series, args: argparse.Namespace) -> Dict:
    """PhyloFunc distance on the pilot table -> PCoA -> coordinate model."""
    dm = protein_pilot_distance(long_df, args)
    pool_df = args.pool_df if (isinstance(args.pool_df, str) and args.pool_df == "auto") else float(args.pool_df)
    return _build_ordination_model(dm, group_map,
                                   center_mode=args.center_mode, cov_estimator=args.cov_estimator,
                                   n_axes=args.embed_dim, pool_cov=args.pool_cov,
                                   cov_eb_pool=args.cov_eb_pool,
                                   pool_dist=args.pool_dist, pool_df=pool_df)


def _as_protein_raw_dict(base: dict, tab: pd.DataFrame, sgm: pd.Series) -> dict:
    groups = list(pd.unique(sgm))
    e = dict(base)
    e["abund"] = tab.to_numpy(dtype=float)
    e["L"] = np.log1p(e["abund"])
    e["groups"] = groups
    e["gs"] = {g: np.where(sgm.to_numpy() == g)[0] for g in groups}
    pall = np.concatenate([e["gs"][g] for g in groups])
    e["libs"] = e["abund"][:, pall].sum(axis=0)
    grand = e["L"][:, pall].mean(axis=1)
    e["dev"] = {g: e["L"][:, e["gs"][g]].mean(axis=1) - grand for g in groups}
    return e


def _build_protein_raw_dict(long_df: pd.DataFrame, group_map: pd.Series, args: argparse.Namespace) -> dict:
    samples = [c for c in long_df.columns if c in group_map.index]
    meta = long_df[ID_COLS].reset_index(drop=True).copy()
    abund = long_df[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    unit = meta["Taxon"].astype(str).to_numpy()
    taxa = pd.unique(unit)
    uid = pd.Series(np.arange(len(taxa)), index=taxa).loc[unit].to_numpy()
    grp = group_map.loc[samples].to_numpy()
    groups = list(pd.unique(grp))
    if len(groups) != 2:
        raise ValueError("raw-pool workflow currently expects exactly two groups.")
    gs = {g: np.where(grp == g)[0] for g in groups}
    other = {groups[0]: groups[1], groups[1]: groups[0]}
    L = np.log1p(abund)
    pall = np.concatenate([gs[g] for g in groups])
    grand = L[:, pall].mean(axis=1)
    dev = {g: L[:, gs[g]].mean(axis=1) - grand for g in groups}
    return {
        "modality": "protein",
        "abund": abund,
        "L": L,
        "dev": dev,
        "unit": unit,
        "uid": uid,
        "rows": [],
        "gs": gs,
        "groups": groups,
        "other": other,
        "libs": abund[:, pall].sum(axis=0),
        "meta": meta,
        "tree_path": str(args.tree),
        "post": None,
    }


def _protein_pilot_raw_view(base: dict, pilot_n: int, seed: int, args: argparse.Namespace) -> tuple[dict, str]:
    observed_n = min(len(base["gs"][g]) for g in base["groups"])
    if pilot_n <= observed_n:
        return P.pilot_view(base, pilot_n, seed), "real"
    tab, sgm = mdctf_pool(base, pilot_n, seed, 1.0, edge_fraction=args.edge_fraction)
    return _as_protein_raw_dict(base, tab, sgm), "simulated"


def _mdctf_power_uniform_strengths(pilot: dict, args: argparse.Namespace, eval_n: int, seed: int) -> List[float]:
    coarse = np.linspace(0.0, 1.0, int(args.mdctf_strength_candidates))
    rows: List[Dict] = []
    preview_boot = int(args.power_preview_boot_number)
    preview_perms = int(args.power_preview_permutations)
    preview_M = int(args.mdctf_preview_pool_size)
    for i, s in enumerate(coarse):
        point_seed = seed + i * 7919
        tab, sgm = mdctf_pool(pilot, preview_M, point_seed, float(s), edge_fraction=args.edge_fraction)
        dm = P.recompute_distance(pilot, tab)
        omega = 0.0 if np.isclose(s, 0.0) else max(0.0, float(core.compute_omega2(dm, sgm)))
        if np.isclose(s, 0.0):
            power = _exchangeable_null_power(
                dm, eval_n=eval_n, boot=preview_boot, perms=preview_perms, seed=point_seed + 41
            )
        else:
            metrics = summarize_distance_metrics_with_replacement(
                dm=dm,
                group_map=sgm,
                boot_number=preview_boot,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=point_seed + 31,
                n_per_group=eval_n,
                permutations=preview_perms,
                omega2_floor=args.omega2_floor,
            )
            power = float(metrics["power"])
        rows.append({"strength": float(s), "omega2": float(omega), "power": float(power)})
        print(f"[protein raw preview] s={s:.3f} omega2={omega:.4f} power={power:.3f}", flush=True)

    preview = pd.DataFrame(rows).sort_values("strength")
    power = np.maximum.accumulate(preview["power"].to_numpy(dtype=float))
    omega = np.maximum.accumulate(preview["omega2"].to_numpy(dtype=float))
    strength = preview["strength"].to_numpy(dtype=float)
    if np.nanmax(omega) <= 1e-10:
        return [0.0, 1.0]
    plateau_idx = np.flatnonzero(power >= 1.0 - 1e-12)
    transition_hi = float(omega[plateau_idx[0]]) if len(plateau_idx) and plateau_idx[0] > 0 else float(omega[-1])
    omega_targets = np.linspace(0.0, transition_hi, int(args.mdctf_power_points))
    unique_omega, idx = np.unique(omega, return_index=True)
    unique_strength = strength[idx]
    selected = np.interp(np.clip(omega_targets, unique_omega[0], unique_omega[-1]), unique_omega, unique_strength).tolist()
    selected.extend([0.0, 1.0])
    if int(args.mdctf_plateau_points) > 0 and len(plateau_idx):
        selected.extend(np.linspace(float(strength[plateau_idx[0]]), 1.0, int(args.mdctf_plateau_points)).tolist())
    out: List[float] = []
    for s in sorted(selected):
        val = float(np.clip(np.round(s, 4), 0.0, 1.0))
        if not out or abs(val - out[-1]) > 1e-4:
            out.append(val)
    return out


def raw_mdctf_curve(d: dict, args: argparse.Namespace, pilot_n: int, seed: int,
                    eval_n: Optional[int] = None) -> pd.DataFrame:
    eval_n = int(args.eval_n if eval_n is None else eval_n)
    pilot, source = _protein_pilot_raw_view(d, pilot_n, seed, args)
    if str(args.mdctf_strengths).strip().lower() in {"auto", "power-uniform"}:
        strengths = _mdctf_power_uniform_strengths(pilot, args, eval_n, seed + 70000)
    else:
        strengths = [float(x) for x in str(args.mdctf_strengths).split(",") if x.strip()]
    print(f"[protein raw] pilot={pilot_n} selected strengths={strengths}", flush=True)
    rows: List[Dict] = []
    for i, s in enumerate(strengths):
        point_seed = seed + 9000 + i * 1291
        tab, sgm = mdctf_pool(pilot, args.pool_size_per_group, point_seed, s, edge_fraction=args.edge_fraction)
        dm = P.recompute_distance(pilot, tab)
        if np.isclose(s, 0.0):
            omega = 0.0
            power = _exchangeable_null_power(
                dm, eval_n=eval_n, boot=args.boot_number, perms=args.permutations, seed=point_seed + 41
            )
            mode = "exchangeable_null"
        else:
            omega = max(0.0, float(core.compute_omega2(dm, sgm)))
            metrics = summarize_distance_metrics_with_replacement(
                dm=dm,
                group_map=sgm,
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=seed + i,
                n_per_group=eval_n,
                permutations=args.permutations,
                omega2_floor=args.omega2_floor,
            )
            power = float(metrics["power"])
            mode = "labeled_bootstrap"
        rows.append({
            "pilot_n": int(pilot_n),
            "pilot_source": source,
            "strength": float(s),
            "eval_n": int(eval_n),
            "true_omega2": float(omega),
            "power": float(power),
            "mode": mode,
        })
        print(f"[protein raw] pilot={pilot_n} s={s:.3f} omega2={omega:.4f} power={power:.3f}", flush=True)
    return pd.DataFrame(rows)


def _plot_raw_power(raw: pd.DataFrame, args: argparse.Namespace, modality: str, stem: str) -> None:
    pilots = sorted(raw["pilot_n"].unique())
    norm = plt.Normalize(min(pilots), max(pilots))
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(10, 6.6))
    data_xmax = float(raw["true_omega2"].dropna().max()) if raw["true_omega2"].notna().any() else 0.0
    xmax = max(0.06, float(args.target_omega2) * 1.2, data_xmax * 1.05)
    xg = np.linspace(0, xmax, 400)
    fit_rows: List[Dict] = []
    for pn in pilots:
        df = raw[raw["pilot_n"] == pn][["true_omega2", "power"]].dropna()
        c = cmap(norm(pn))
        params = draw_binned_null_hill_group(
            ax, df, color=c, label=f"pilot {pn}", x=xg, bin_width=args.fit_bin_width, raw_alpha=0.28
        )
        pred = np.nan
        if params:
            pred = float(null_started_hill_sigmoid(
                np.array([args.target_omega2]), params["h"], params["x0"], params["floor"]
            )[0])
        fit_rows.append({"pilot_n": pn, **({} if params is None else params), "power_at_target_omega2": pred})
    ax.axhline(args.target_power, color="gray", ls=":", lw=1.2)
    ax.axhline(args.alpha, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.axvline(args.target_omega2, color="red", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("true omega^2")
    ax.set_ylabel("power")
    ax.set_title(f"{modality}: raw MDC-TF synthetic pool (eval_n={args.eval_n})")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / f"{stem}_power_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(args.out / f"{stem}_sigmoid_summary.csv", index=False)


def pilot_curve(model: Dict, args: argparse.Namespace, seed: int, eval_n: Optional[int] = None) -> pd.DataFrame:
    """Run the power-uniform effect sweep on a fitted model; return (scale, omega^2, power) rows."""
    eval_n = int(args.eval_n if eval_n is None else eval_n)
    psize = _effective_pool_size(args.pool_size_per_group, eval_n)
    pool_gm = _ordination_pool_group_map(model, psize)
    scales = _ordination_scales_by_power_uniform(
        args, eval_n, model, psize, pool_gm, seed=seed, workflow="taxon-function"
    )
    rows: List[Dict] = []
    for i, sc in enumerate(scales):
        dm = _ordination_pool_dm(model, float(sc), psize, seed + 9000 + i * 89)
        m = summarize_distance_metrics_with_replacement(
            dm=dm,
            group_map=pool_gm,
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            random_seed=seed + i,
            n_per_group=eval_n,
            permutations=args.permutations,
            omega2_floor=args.omega2_floor,
        )
        rows.append(
            {
                "scale": float(sc),
                "true_omega2": float(m["true_omega2"]),
                "power": float(m["power"]),
                "mode": "dilution" if sc < 1.0 else "enhancement",
            }
        )
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Protein (PhyloFunc) raw MDC-TF power workflow.")
    p.add_argument("--table", type=Path, default=core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv")
    p.add_argument("--tree", type=Path, default=core.DATAPRO_DIR / "rooted-tree.nwk")
    p.add_argument("--group", type=Path, default=core.DATAPRO_DIR / "group.csv")
    p.add_argument("--pilot-ns", type=str, default="5,10,17")
    p.add_argument("--eval-n", type=int, default=None, help="Target per-group sample size (default: observed).")
    p.add_argument("--eval-ns", type=str, default=None,
                   help="Comma list of target sample sizes; enables EVAL-SWEEP mode (fix one pilot, "
                        "vary target n) -> layered curves like adaptive_evaln. e.g. 17,50,100")
    p.add_argument("--sweep-pilot", type=int, default=None,
                   help="Pilot n to fix in eval-sweep mode (default: observed n).")
    p.add_argument("--target-power", type=float, default=0.8)
    p.add_argument("--target-omega2", type=float, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--engine", type=str, default="raw-pool", choices=["raw-pool", "ordination"],
                   help="Default raw-pool uses MDC-TF synthetic Taxon-Function tables and recomputes "
                        "PhyloFunc. Use ordination to reproduce the legacy MVN/PCoA workflow.")
    p.add_argument("--pool-size-per-group", type=int, default=1000)
    p.add_argument("--boot-number", type=int, default=100)
    p.add_argument("--permutations", type=int, default=99)
    p.add_argument("--increase-num", type=int, default=18, help="Half of the power-uniform target count.")
    p.add_argument("--decrease-num", type=int, default=18)
    p.add_argument("--center-mode", type=str, default="omega-calibrated",
                   choices=["observed", "debiased", "empirical-bayes", "omega-calibrated"])
    p.add_argument("--cov-estimator", type=str, default="ledoit-wolf",
                   choices=["ledoit-wolf", "ledoit-wolf-ddof", "nonlinear-shrinkage", "isotropic"],
                   help="Per-group covariance shrinkage: linear Ledoit-Wolf (default); "
                        "'ledoit-wolf-ddof' adds an n/(n-1) trace de-bias that removes small-pilot "
                        "optimism; 'nonlinear-shrinkage' (LW 2020) not recommended for small n.")
    p.add_argument("--embed-dim", type=int, default=None,
                   help="Standardise the PCoA embedding to this many top axes across all pilots. "
                        "Removes the pilot-size dimensionality artefact (small pilot -> few axes -> "
                        "inflated power). Must be <= the axes a small pilot yields (~2*pilot_n-1).")
    p.add_argument("--pool-cov", action="store_true",
                   help="Pool both groups' within-group covariance into one shared Sigma (enforces "
                        "PERMANOVA equal-dispersion; drops noisy per-group dispersion difference).")
    p.add_argument("--cov-eb-pool", action="store_true",
                   help="Empirical-Bayes PARTIAL pooling: shrink each group's covariance toward the "
                        "pooled one by lambda=k/(k+n) (data-driven; not full pooling).")
    p.add_argument("--protein-transform", type=str, default="vst",
                   choices=["none", "vst", "lowrank", "aggregate"],
                   help="Pilot-table preprocessing before PhyloFunc (default vst): vst=log1p (D, "
                        "variance-stabilising, makes power converge across pilot sizes); "
                        "lowrank=SVD denoise (A); aggregate=phylo-clade taxon collapse (C); none.")
    p.add_argument("--lowrank-rank", type=int, default=5, help="Rank for --protein-transform lowrank.")
    p.add_argument("--agg-clades", type=int, default=300, help="Clades for --protein-transform aggregate.")
    p.add_argument("--tree-jitter-sigma", type=float, default=0.0,
                   help="Phylogenetic-uncertainty sensitivity: lognormal branch-length jitter sigma "
                        "applied to the tree before the distance (0.0 = original tree).")
    p.add_argument("--tree-nni-prob", type=float, default=0.0,
                   help="Phylogenetic-uncertainty sensitivity: NNI topology-swap probability on "
                        "low-support nodes before the distance (0.0 = original topology).")
    p.add_argument("--tree-support-threshold", type=float, default=None,
                   help="Only NNI-perturb internal nodes with support below this (None = all).")
    p.add_argument("--pool-dist", type=str, default="student-t", choices=["gaussian", "student-t"],
                   help="Synthetic pool distribution (default student-t: matches the heavy-tailed "
                        "spread of real omics distances; df data-driven via --pool-df auto).")
    p.add_argument("--pool-df", type=str, default="auto",
                   help="Student-t degrees of freedom: 'auto' (estimate from data) or a number.")
    p.add_argument("--protein-generator", type=str, default="template-mask", choices=["template-mask", "bernoulli"],
                   help="Generator used only to simulate pilots larger than observed n.")
    p.add_argument("--edge-fraction", type=str, default="auto",
                   help="MDC-TF edge-level residual fraction in [0,1], or 'auto' (=1.0).")
    p.add_argument("--mdctf-strengths", type=str,
                   default="auto",
                   help="MDC-TF strength grid. Use 'auto'/'power-uniform' for pilot-only "
                        "preview power-uniform points, or pass a comma-separated fixed grid.")
    p.add_argument("--mdctf-strength-candidates", type=int, default=15)
    p.add_argument("--mdctf-power-points", type=int, default=12)
    p.add_argument("--mdctf-plateau-points", type=int, default=5)
    p.add_argument("--mdctf-preview-pool-size", type=int, default=180)
    p.add_argument("--ordination-enhance-max", type=float, default=3.0)
    p.add_argument("--omega-calibrated-max-scale", type=float, default=20.0)
    p.add_argument("--adaptive-reference-n", type=float, default=17.0)
    p.add_argument("--power-preview-boot-number", type=int, default=20)
    p.add_argument("--power-preview-permutations", type=int, default=49)
    p.add_argument("--omega-grid-candidates", type=int, default=60)
    p.add_argument("--fit-bin-width", type=float, default=0.0015)
    p.add_argument("--fit-power-min", type=float, default=0.15)
    p.add_argument("--omega2-floor", type=float, default=0.0)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--random-seed", type=int, default=20260614)
    p.add_argument("--sim-seed", type=int, default=20260614, help="Seed for simulating pilots > observed n.")
    p.add_argument(
        "--curve-seed-offset",
        type=int,
        default=0,
        help="Optional offset applied only to power-curve sampling seeds; pilot construction is unchanged.",
    )
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(effect_grid="power-uniform")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.edge_fraction = _resolve_auto_edge_fraction(args.edge_fraction)

    group_map = _read_group_map(args.group)
    long_df, agm = _read_protein_long_table(args.table, group_map)
    observed_n = int(agm.value_counts().min())
    if args.eval_n is None:
        args.eval_n = observed_n

    if args.engine == "raw-pool":
        if args.eval_ns:
            raise ValueError("protein raw-pool currently supports --eval-n; use --engine ordination for --eval-ns.")
        pilot_ns = [int(x) for x in str(args.pilot_ns).split(",") if x.strip()]
        print(f"[protein raw] observed n/group={observed_n}, eval_n={args.eval_n}, pilots={pilot_ns}", flush=True)
        d = _build_protein_raw_dict(long_df, agm, args)
        frames = [raw_mdctf_curve(d, args, pn, args.random_seed + pn * 1009) for pn in pilot_ns]
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(args.out / "protein_power_curves_raw.csv", index=False)
        _plot_raw_power(raw, args, modality="Protein (PhyloFunc)", stem="protein")
        (args.out / "summary.json").write_text(json.dumps({
            "modality": "Protein (PhyloFunc)",
            "engine": "raw-pool",
            "generator": "MDC-TF",
            "eval_n": int(args.eval_n),
            "pilot_ns": pilot_ns,
            "target_power": args.target_power,
            "target_omega2": args.target_omega2,
            "pool_size_per_group": args.pool_size_per_group,
            "boot_number": args.boot_number,
            "permutations": args.permutations,
            "edge_fraction": args.edge_fraction,
            "mdctf_strengths": args.mdctf_strengths,
        }, indent=2), encoding="utf-8")
        print(f"[protein raw] done -> {args.out}", flush=True)
        return

    def _build_for_pilot(pn: int):
        seed = args.random_seed + pn * 1009
        if pn <= observed_n:
            pgm = _subsample_group_map(agm, pn, seed)
            pdf = long_df[ID_COLS + list(pgm.index)].copy()
            return build_protein_model(pdf, pgm, args), "real", seed
        pdf, pgm, _ = generate_taxon_function_pool(
            long_df, agm, pool_size_per_group=pn, random_seed=args.sim_seed,
            between_scale=1.0, residual_scale=1.0, noise_multiplier=0.10,
            detection_slope=1.0, protein_generator=args.protein_generator,
        )
        return build_protein_model(pdf, pgm, args), "simulated", seed

    if args.eval_ns:
        eval_ns = [int(x) for x in str(args.eval_ns).split(",") if x.strip()]
        pn = int(args.sweep_pilot if args.sweep_pilot is not None else observed_n)
        print(f"[protein] EVAL-SWEEP: pilot={pn}, eval_ns={eval_ns}", flush=True)
        model, source, seed = _build_for_pilot(pn)
        sweep_rows: List[pd.DataFrame] = []
        for en in eval_ns:
            df = pilot_curve(model, args, seed + int(args.curve_seed_offset) + en * 17, eval_n=en)
            df.insert(0, "eval_n", en)
            sweep_rows.append(df)
            ntrans = int(((df["power"] >= 0.15) & (df["power"] <= 0.95)).sum())
            print(f"[protein] eval_n={en}: {len(df)} points, {ntrans} in transition", flush=True)
        raw = pd.concat(sweep_rows, ignore_index=True)
        raw["pilot_n"] = pn
        raw["pilot_source"] = source
        raw.to_csv(args.out / "protein_evalsweep_raw.csv", index=False)
        _plot_eval_sweep(raw, args, pn, float(model.get("pilot_target_omega2") or np.nan), modality="Protein (PhyloFunc)")
        print(f"[protein] done (eval-sweep) -> {args.out}", flush=True)
        return

    pilot_ns = [int(x) for x in str(args.pilot_ns).split(",") if x.strip()]
    print(f"[protein] observed n/group={observed_n}, eval_n={args.eval_n}, pilots={pilot_ns}", flush=True)

    all_rows: List[pd.DataFrame] = []
    for pn in pilot_ns:
        seed = args.random_seed + pn * 1009
        if pn <= observed_n:
            pgm = _subsample_group_map(agm, pn, seed)
            pdf = long_df[ID_COLS + list(pgm.index)].copy()
            source = "real"
        else:
            pdf, pgm, _ = generate_taxon_function_pool(
                long_df, agm, pool_size_per_group=pn, random_seed=args.sim_seed,
                between_scale=1.0, residual_scale=1.0, noise_multiplier=0.10,
                detection_slope=1.0, protein_generator=args.protein_generator,
            )
            source = "simulated"
        model = build_protein_model(pdf, pgm, args)
        df = pilot_curve(model, args, seed + int(args.curve_seed_offset))
        df.insert(0, "pilot_n", pn)
        df.insert(1, "pilot_source", source)
        df["k_axes"] = model["k"]
        df["center_shrinkage"] = model.get("center_shrinkage")
        df["pilot_target_omega2"] = model.get("pilot_target_omega2")
        all_rows.append(df)
        ntrans = int(((df["power"] >= 0.15) & (df["power"] <= 0.95)).sum())
        print(f"[protein] pilot_n={pn} ({source}): {len(df)} points, {ntrans} in transition", flush=True)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(args.out / "protein_power_curves_raw.csv", index=False)
    _summarize_and_plot(raw, args, modality="Protein (PhyloFunc)")
    print(f"[protein] done -> {args.out}", flush=True)


def _empirical_alpha(raw: pd.DataFrame, nominal: float) -> float:
    """Empirical type-I rate = rejection rate of the simulated null (scale=0, identical centroids),
    pooled across pilots. Anchoring the sigmoid here (instead of the nominal alpha) reports the
    floor the simulation actually delivers -- honest, and = the E1 type-I calibration check."""
    null_pts = raw[raw["scale"] == 0.0] if "scale" in raw else raw.iloc[0:0]
    if len(null_pts) == 0:
        null_pts = raw[raw["true_omega2"] < 0.003]
    if len(null_pts) == 0:
        return float(nominal)
    return float(min(max(null_pts["power"].mean(), 1e-3), 0.5))


def _summarize_and_plot(raw: pd.DataFrame, args: argparse.Namespace, modality: str) -> None:
    pilots = sorted(raw["pilot_n"].unique())
    norm = plt.Normalize(min(pilots), max(pilots))
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(10, 6.6))
    data_xmax = float(raw["true_omega2"].dropna().max()) if raw["true_omega2"].notna().any() else 0.0
    xmax = max(0.06, float(args.target_omega2) * 1.2, data_xmax * 1.05)
    xg = np.linspace(0, xmax, 400)
    emp_alpha = _empirical_alpha(raw, args.alpha)   # measured null rejection rate -> sigmoid floor
    fit_rows: List[Dict] = []
    for pn in pilots:
        df = raw[raw["pilot_n"] == pn][["true_omega2", "power", "mode"]].dropna()
        c = cmap(norm(pn))
        ax.scatter(df["true_omega2"], df["power"], s=20, color=c, alpha=0.5, edgecolors="none", zorder=3)
        fit = fit_logistic(df.copy(), alpha=emp_alpha)
        params = fit.get("params")
        pred = np.nan
        if params:
            ax.plot(xg, logistic_curve(xg, params["k"], params["x0"], emp_alpha), color=c, lw=2.0, zorder=2)
            pred = float(logistic_curve(np.array([args.target_omega2]), params["k"], params["x0"], emp_alpha)[0])
        fit_rows.append({"pilot_n": pn, "fit_status": fit.get("status"),
                         "k": None if not params else params["k"], "x0": None if not params else params["x0"],
                         "power_at_target_omega2": pred, "empirical_alpha": emp_alpha})
    ax.axhline(args.target_power, color="gray", ls=":", lw=1.2)
    ax.axhline(emp_alpha, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.text(xmax * 0.99, emp_alpha + 0.01, f"empirical type-I = {emp_alpha:.3f} (nominal {args.alpha})",
            ha="right", va="bottom", fontsize=8, color="black")
    ax.axvline(args.target_omega2, color="red", ls="--", lw=1.0, alpha=0.6)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, label="pilot n per group"); cb.set_ticks(pilots)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("true omega^2"); ax.set_ylabel("power")
    ax.set_title(f"{modality}: ordination + power-uniform (eval_n={args.eval_n})")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(args.out / "protein_power_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(args.out / "protein_sigmoid_summary.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps({
        "modality": modality, "engine": "ordination", "effect_grid": "power-uniform",
        "eval_n": int(args.eval_n), "pilot_ns": [int(x) for x in pilots], "target_power": args.target_power,
        "target_omega2": args.target_omega2, "pool_size_per_group": args.pool_size_per_group,
        "boot_number": args.boot_number, "permutations": args.permutations,
        "center_mode": args.center_mode, "protein_generator": args.protein_generator,
        "empirical_alpha": _empirical_alpha(raw, args.alpha), "nominal_alpha": args.alpha,
    }, indent=2), encoding="utf-8")


def _plot_eval_sweep(raw: pd.DataFrame, args: argparse.Namespace, pilot_n: int,
                     full_omega2: float, modality: str) -> None:
    """Layered power curves: one per target sample size (eval_n), fixed pilot."""
    eval_ns = sorted(raw["eval_n"].unique())
    cmap = cm.viridis
    norm = plt.Normalize(min(eval_ns), max(eval_ns))
    fig, ax = plt.subplots(figsize=(10, 6.6))
    data_xmax = float(raw["true_omega2"].dropna().max()) if raw["true_omega2"].notna().any() else 0.06
    xmax = max(0.06, data_xmax * 1.05)
    xg = np.linspace(0, xmax, 400)
    fit_rows: List[Dict] = []
    for en in eval_ns:
        df = raw[raw["eval_n"] == en][["true_omega2", "power", "mode"]].dropna()
        c = cmap(norm(en))
        ax.scatter(df["true_omega2"], df["power"], s=20, color=c, alpha=0.5, edgecolors="none", zorder=3)
        fit = fit_logistic(df.copy(), alpha=args.alpha)
        params = fit.get("params")
        if params:
            ax.plot(xg, logistic_curve(xg, params["k"], params["x0"], args.alpha),
                    color=c, lw=2.2, zorder=2, label=f"eval n={en}")
        fit_rows.append({"eval_n": en, "fit_status": fit.get("status"),
                         "k": None if not params else params["k"], "x0": None if not params else params["x0"]})
    if np.isfinite(full_omega2) and full_omega2 > 0:
        ax.axvline(full_omega2, color="gray", ls="--", lw=1.2, alpha=0.8,
                   label=f"pilot full omega^2={full_omega2:.3f}")
    ax.axhline(args.target_power, color="gray", ls=":", lw=1.0)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("true omega^2"); ax.set_ylabel("power")
    ax.set_title(f"{modality}: eval-sweep (pilot n={pilot_n}, power-uniform)")
    ax.grid(alpha=0.22); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "protein_evalsweep_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(args.out / "protein_evalsweep_sigmoid_summary.csv", index=False)


if __name__ == "__main__":
    main()
