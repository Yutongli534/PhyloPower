#!/usr/bin/env python3
"""Gene/taxonomic raw-pool power-analysis workflow.

The default workflow builds PCAM synthetic feature pools, recomputes Gemelli
distances for each effect level, bootstraps PERMANOVA at the requested sample
size, and fits a monotone power curve over realized omega squared.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from skbio import TreeNode

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pcam_gen as P  # noqa: E402
from _fig4_curve_plotting import draw_binned_null_hill_group  # noqa: E402
from semisynthetic_power import (  # noqa: E402
    _as_distance_frame,
    _build_ordination_model,
    _effective_pool_size,
    _ordination_pool_dm,
    _ordination_pool_group_map,
    _ordination_scales_by_power_uniform,
    _read_group_map,
    _read_taxon_feature_table,
    _write_taxon_csv,
    generate_taxon_pool,
    summarize_distance_metrics_with_replacement,
)
from phylopower import core
from logistic_fit import fit_logistic, logistic_curve  # noqa: E402

core.load_core_runtime()

GENE_GRID = [
    (0.5, 1.0), (0.6, 1.0), (0.68, 1.0), (0.75, 1.0), (0.82, 1.0),
    (0.88, 1.0), (0.93, 1.0), (0.97, 1.0), (1.0, 1.0), (1.0, 1.3),
    (1.0, 1.7),
]


def _build_pcam_scenario_worker(payload: Dict) -> Dict:
    pilot = payload["pilot"]
    pi = float(payload["pi"])
    scale = float(payload["scale"])
    point_seed = int(payload["point_seed"])
    pool_n = int(payload["pool_n"])
    ndon = payload["pcam_ndon"]
    tab, sgm = P.pcam_pool(pilot, pool_n, point_seed, pi, scale, ndon=ndon)
    dm = P.recompute_distance(pilot, tab)
    omega = (
        0.0
        if np.isclose(pi, 0.5) and np.isclose(scale, 1.0)
        else max(0.0, float(core.compute_omega2(dm, sgm)))
    )
    return {
        "scenario_index": int(payload["scenario_index"]),
        "pilot_n": int(payload["pilot_n"]),
        "pilot_source": payload["pilot_source"],
        "pi": pi,
        "scale": scale,
        "dm": dm,
        "group_map": sgm,
        "true_omega2": float(omega),
        "pool_size_per_group": int(pool_n),
        "point_seed": point_seed,
    }


def _build_pcam_scenarios_parallel(
    *,
    pilot: dict,
    source: str,
    pilot_n: int,
    pool_n: int,
    seed: int,
    effects: List[Tuple[float, float]],
    ndon: int,
    n_jobs: int,
    offset: int = 0,
) -> List[Dict]:
    payloads = []
    for j, (pi, scale) in enumerate(effects):
        i = int(offset + j)
        payloads.append(
            {
                "pilot": pilot,
                "pilot_source": source,
                "pilot_n": int(pilot_n),
                "pool_n": int(pool_n),
                "scenario_index": i,
                "point_seed": int(seed + 9000 + i * 131),
                "pi": float(pi),
                "scale": float(scale),
                "pcam_ndon": ndon,
            }
        )
    if max(1, int(n_jobs)) == 1 or len(payloads) <= 1:
        return [_build_pcam_scenario_worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=max(1, int(n_jobs))) as ex:
        return list(ex.map(_build_pcam_scenario_worker, payloads))


def _gene_candidate_grid(args: argparse.Namespace) -> List[Tuple[float, float]]:
    """Return PCAM candidate effects for the raw-pool workflow.

    ``fixed`` uses the predefined PCAM grid. ``auto`` evaluates a
    modest two-dimensional candidate set and lets realized omega^2, rather than
    the PCAM parameters themselves, decide the formal plotting/fitting points.
    """
    mode = str(getattr(args, "pcam_grid", "auto")).strip().lower()
    if mode == "fixed":
        return [(float(pi), float(scale)) for pi, scale in GENE_GRID]
    if mode not in {"auto", "omega-uniform", "power-uniform"}:
        raise ValueError("pcam_grid must be 'auto'/'omega-uniform' or 'fixed'.")

    pi_n = max(3, int(getattr(args, "pcam_pi_candidates", 17)))
    scale_n = max(2, int(getattr(args, "pcam_scale_candidates", 6)))
    scale_max = max(1.0, float(getattr(args, "pcam_scale_max", 1.7)))
    pi_values = np.linspace(0.5, 1.0, pi_n)
    scale_values = np.linspace(1.0, scale_max, scale_n)

    candidates: List[Tuple[float, float]] = []
    candidates.extend((float(pi), 1.0) for pi in pi_values)
    candidates.extend((1.0, float(scale)) for scale in scale_values)

    # A small interior grid helps when pi and scale interact nonlinearly, but
    # keeps the preview cheap enough for routine sensitivity analysis.
    interior_pi = np.linspace(0.7, 1.0, min(5, pi_n))
    interior_scale = np.linspace(1.0, scale_max, min(4, scale_n))
    candidates.extend((float(pi), float(scale)) for pi in interior_pi for scale in interior_scale)

    out: List[Tuple[float, float]] = []
    seen: set[Tuple[float, float]] = set()
    for pi, scale in candidates:
        key = (round(float(pi), 4), round(float(scale), 4))
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _select_omega_uniform_gene_scenarios(
    scenarios: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    mode = str(getattr(args, "pcam_grid", "auto")).strip().lower()
    if mode == "fixed":
        return scenarios
    if not scenarios:
        return scenarios

    n_points = max(3, int(getattr(args, "pcam_power_points", 20)))
    near_zero_points = max(0, int(getattr(args, "pcam_near_zero_points", 4)))
    target_omega2 = float(getattr(args, "target_omega2", np.nan))
    ordered = sorted(scenarios, key=lambda s: (float(s["true_omega2"]), float(s["pi"]), float(s["scale"])))
    omega = np.asarray([float(s["true_omega2"]) for s in ordered], dtype=float)
    max_omega = float(np.nanmax(omega)) if len(omega) else 0.0
    if max_omega <= 1e-12:
        selected = ordered[:1]
    else:
        global_points = max(3, n_points - near_zero_points)
        target_specs: List[Tuple[float, bool]] = [(float(x), False) for x in np.linspace(0.0, max_omega, global_points)]
        if near_zero_points > 0:
            near_zero_arg = str(getattr(args, "pcam_near_zero_omega_max", "auto")).strip().lower()
            if near_zero_arg == "auto":
                if np.isfinite(target_omega2) and target_omega2 > 0:
                    near_zero_hi = max(2.0 * target_omega2, target_omega2 + 0.02)
                else:
                    near_zero_hi = 0.1 * max_omega
                near_zero_hi = min(float(near_zero_hi), max(0.06, 0.15 * max_omega))
            else:
                near_zero_hi = float(near_zero_arg)
            near_zero_hi = float(np.clip(near_zero_hi, 0.0, max_omega))
            if near_zero_hi > 0:
                target_specs.extend((float(x), True) for x in np.linspace(0.0, near_zero_hi, near_zero_points))
        if np.isfinite(target_omega2):
            target_specs.append((float(np.clip(target_omega2, 0.0, max_omega)), True))
        compact: Dict[float, bool] = {}
        for value, force in target_specs:
            key = float(np.round(value, 8))
            compact[key] = bool(compact.get(key, False) or force)
        target_specs = sorted((k, v) for k, v in compact.items())
        chosen: List[int] = []
        min_spacing = max_omega / max(float(n_points) * 1.8, 1.0)
        for target, force_near_zero in target_specs:
            ranked = np.argsort(np.abs(omega - float(target)))
            fallback_idx: Optional[int] = None
            for idx in ranked:
                idx = int(idx)
                if idx in chosen:
                    continue
                if fallback_idx is None:
                    fallback_idx = idx
                if force_near_zero or all(abs(float(omega[idx]) - float(omega[j])) >= min_spacing for j in chosen):
                    chosen.append(idx)
                    break
            else:
                if fallback_idx is not None and fallback_idx not in chosen:
                    chosen.append(fallback_idx)
        selected = [ordered[i] for i in chosen]

    selected = sorted(selected, key=lambda s: (float(s["true_omega2"]), float(s["pi"]), float(s["scale"])))
    out: List[Dict] = []
    seen_effects: set[Tuple[float, float]] = set()
    for i, scenario in enumerate(selected):
        key = (round(float(scenario["pi"]), 4), round(float(scenario["scale"]), 4))
        if key in seen_effects:
            continue
        item = dict(scenario)
        item["scenario_index"] = len(out)
        out.append(item)
        seen_effects.add(key)
    if len(out) > n_points:
        keep = np.rint(np.linspace(0, len(out) - 1, n_points)).astype(int).tolist()
        keep_unique: List[int] = []
        for idx in keep:
            idx = int(np.clip(idx, 0, len(out) - 1))
            if idx not in keep_unique:
                keep_unique.append(idx)
        cursor = 0
        while len(keep_unique) < n_points and cursor < len(out):
            if cursor not in keep_unique:
                keep_unique.append(cursor)
            cursor += 1
        out = [out[i] for i in sorted(keep_unique[:n_points])]
        for i, item in enumerate(out):
            item["scenario_index"] = i
    return out


def precompute_pcam_scenarios(
    d: dict,
    args: argparse.Namespace,
    pilot_n: int,
    seed: int,
    pool_size_per_group: Optional[int] = None,
) -> List[Dict]:
    """Build reusable PCAM scenarios selected by realized omega^2 coverage."""
    pilot, source = _gene_pilot_raw_view(d, pilot_n, seed, args)
    pool_n = int(args.pool_size_per_group if pool_size_per_group is None else pool_size_per_group)
    candidates = _gene_candidate_grid(args)
    preview: List[Dict] = []
    scenario_n_jobs = max(1, int(getattr(args, "scenario_n_jobs", 1)))

    preview = _build_pcam_scenarios_parallel(
        pilot=pilot,
        source=source,
        pilot_n=pilot_n,
        pool_n=pool_n,
        seed=seed,
        effects=[(float(pi), float(scale)) for pi, scale in candidates],
        ndon=args.pcam_ndon,
        n_jobs=scenario_n_jobs,
    )
    for scenario in preview:
        print(
            f"[gene raw preview] pilot={pilot_n} pi={scenario['pi']:.3f} scale={scenario['scale']:.3f} "
            f"omega2={scenario['true_omega2']:.4f}",
            flush=True,
        )

    target_omega2 = float(getattr(args, "target_omega2", np.nan))
    extend_max = float(getattr(args, "pcam_scale_extend_max", getattr(args, "pcam_scale_max", 1.7)))
    current_scale_max = max(float(scale) for _, scale in candidates)
    current_omega_max = max(float(s["true_omega2"]) for s in preview) if preview else 0.0
    if (
        str(getattr(args, "pcam_grid", "auto")).strip().lower() != "fixed"
        and np.isfinite(target_omega2)
        and current_omega_max < max(target_omega2 * 1.25, target_omega2 + 0.01)
        and extend_max > current_scale_max
    ):
        extra_count = max(3, int(getattr(args, "pcam_scale_candidates", 6)))
        extra_scales = np.linspace(current_scale_max, extend_max, extra_count + 1)[1:]
        print(
            f"[gene raw preview] extending scale scan to {extend_max:.3g} "
            f"because max preview omega2={current_omega_max:.4f}",
            flush=True,
        )
        offset = len(preview)
        extra_scenarios = _build_pcam_scenarios_parallel(
            pilot=pilot,
            source=source,
            pilot_n=pilot_n,
            pool_n=pool_n,
            seed=seed,
            effects=[(1.0, float(scale)) for scale in extra_scales],
            ndon=args.pcam_ndon,
            n_jobs=scenario_n_jobs,
            offset=offset,
        )
        for scenario in extra_scenarios:
            preview.append(scenario)
            print(
                f"[gene raw preview] pilot={pilot_n} pi=1.000 scale={scenario['scale']:.3f} "
                f"omega2={scenario['true_omega2']:.4f}",
                flush=True,
            )

    selected = _select_omega_uniform_gene_scenarios(preview, args)
    print(
        f"[gene raw] selected {len(selected)} formal PCAM scenarios "
        f"from {len(preview)} preview candidates",
        flush=True,
    )
    return selected


def evaluate_precomputed_pcam_scenarios(
    scenarios: List[Dict],
    args: argparse.Namespace,
    eval_n: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for scenario in scenarios:
        metrics = summarize_distance_metrics_with_replacement(
            dm=scenario["dm"],
            group_map=scenario["group_map"],
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=1,
            random_seed=int(seed) + int(scenario["scenario_index"]),
            n_per_group=int(eval_n),
            permutations=args.permutations,
            omega2_floor=args.omega2_floor,
        )
        rows.append({
            "pilot_n": int(scenario["pilot_n"]),
            "pilot_source": scenario["pilot_source"],
            "pi": float(scenario["pi"]),
            "scale": float(scenario["scale"]),
            "eval_n": int(eval_n),
            "true_omega2": float(metrics["true_omega2"]),
            "power": float(metrics["power"]),
            "mode": "pcam_raw_pool",
            "scenario_index": int(scenario["scenario_index"]),
            "pool_size_per_group": int(scenario["pool_size_per_group"]),
            "failed_bootstraps": int(metrics.get("failed_bootstraps", 0)),
        })
        print(
            f"[gene raw] eval_n={eval_n} pilot={scenario['pilot_n']} "
            f"pi={scenario['pi']:.3f} scale={scenario['scale']:.3f} "
            f"omega2={metrics['true_omega2']:.4f} power={metrics['power']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


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


def gene_pilot_distance(table: pd.DataFrame, args: argparse.Namespace, stem: str) -> pd.DataFrame:
    """The pilot Gemelli (Phylo-)RPCA distance exactly as the pipeline computes it."""
    with tempfile.TemporaryDirectory(prefix=f"gene_wf_{stem}_") as tmp:
        work = Path(tmp)
        csv = work / "table.csv"
        _write_taxon_csv(table, csv)
        tree_path = _effective_tree_path(args, work, stem)
        table_qza, tree_qza, tax_qza = core.prepare_qza_inputs(csv, Path(tree_path), args.taxonomy, work)
        dm = core.compute_gemelli_rpca_distance(
            table_qza_path=table_qza,
            tree_qza_path=tree_qza,
            taxonomy_qza_path=tax_qza,
            out_dir=work,
            qiime_env_name=args.qiime_env,
            output_stem=stem,
            use_phylogeny=bool(getattr(args, "use_phylogeny", True)),
        )
    if dm is None:
        raise RuntimeError(f"Gemelli distance failed for {stem}.")
    return _as_distance_frame(dm)


def build_gene_model(table: pd.DataFrame, group_map: pd.Series, args: argparse.Namespace, stem: str) -> Dict:
    """Gemelli (Phylogenetic-RPCA) distance on the pilot table -> PCoA -> coordinate model."""
    dm = gene_pilot_distance(table, args, stem)
    pool_df = args.pool_df if (isinstance(args.pool_df, str) and args.pool_df == "auto") else float(args.pool_df)
    return _build_ordination_model(dm, group_map,
                                   center_mode=args.center_mode, cov_estimator=args.cov_estimator,
                                   n_axes=args.embed_dim, cov_eb_pool=args.cov_eb_pool,
                                   pool_dist=args.pool_dist, pool_df=pool_df)


def _resolve_pcam_gene_blocks(value: int | str, n_features: int) -> int:
    """Resolve the PCAM phylogenetic block count.

    ``auto`` scales with the number of feature/tree tips while keeping the
    block count in a conservative range. This avoids a fixed tree cut count
    becoming too coarse for large tables or too fragmented for small ones.
    """
    if str(value).strip().lower() == "auto":
        return int(np.clip(round(np.sqrt(max(int(n_features), 1))), 6, 24))
    resolved = int(value)
    if resolved < 1:
        raise ValueError("pcam_gene_blocks must be a positive integer or 'auto'.")
    return resolved


def _build_gene_raw_dict(table: pd.DataFrame, group_map: pd.Series, args: argparse.Namespace) -> dict:
    samples = [c for c in table.columns if c in group_map.index]
    table = table[samples].copy()
    abund = table.to_numpy(dtype=float)
    unit = table.index.astype(str).to_numpy()
    tree = TreeNode.read(str(args.tree))
    pcam_gene_blocks = _resolve_pcam_gene_blocks(args.pcam_gene_blocks, len(unit))
    args.pcam_gene_blocks = pcam_gene_blocks
    blocks = P.clade_assign(tree, unit, pcam_gene_blocks)
    rows = [np.where(blocks == c)[0] for c in range(pcam_gene_blocks)]
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
        "modality": "gene",
        "abund": abund,
        "L": L,
        "dev": dev,
        "unit": unit,
        "uid": np.arange(len(unit)),
        "rows": rows,
        "gs": gs,
        "groups": groups,
        "other": other,
        "libs": abund[:, pall].sum(axis=0),
        "meta": None,
        "tree_path": str(args.tree),
        "post": table.index,
    }


def _as_gene_raw_dict(base: dict, tab: pd.DataFrame, sgm: pd.Series) -> dict:
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


def _gene_pilot_raw_view(base: dict, pilot_n: int, seed: int, args: argparse.Namespace) -> tuple[dict, str]:
    observed_n = min(len(base["gs"][g]) for g in base["groups"])
    if pilot_n <= observed_n:
        return P.pilot_view(base, pilot_n, seed), "real"
    tab, sgm = P.pcam_pool(base, pilot_n, seed, pi=1.0, scale=1.0, ndon=args.pcam_ndon)
    return _as_gene_raw_dict(base, tab, sgm), "simulated"


def raw_pcam_curve(d: dict, args: argparse.Namespace, pilot_n: int, seed: int,
                   eval_n: Optional[int] = None) -> pd.DataFrame:
    eval_n = int(args.eval_n if eval_n is None else eval_n)
    scenarios = precompute_pcam_scenarios(d, args, pilot_n, seed)
    return evaluate_precomputed_pcam_scenarios(scenarios, args, eval_n, seed)


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
            from _fig4_curve_plotting import null_started_hill_sigmoid
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
    ax.set_title(f"{modality}: raw synthetic pool (eval_n={args.eval_n})")
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
        args, eval_n, model, psize, pool_gm, seed=seed, workflow="taxon"
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
    p = argparse.ArgumentParser(description="Gene (Gemelli) raw-pool power workflow.")
    p.add_argument("--table", type=Path, default=core.DATAGENE_DIR / "table.csv")
    p.add_argument("--tree", type=Path, default=core.DATAGENE_DIR / "rooted-tree.nwk")
    p.add_argument("--taxonomy", type=Path, default=core.DATAGENE_DIR / "taxonomy.csv")
    p.add_argument("--use-phylogeny", dest="use_phylogeny", action="store_true", default=True,
                   help="Gemelli phylogenetic-RPCA (default). Use --no-use-phylogeny for plain RPCA.")
    p.add_argument("--no-use-phylogeny", dest="use_phylogeny", action="store_false",
                   help="Plain (non-phylogenetic) Gemelli RPCA.")
    p.add_argument("--group", type=Path, default=core.DATAGENE_DIR / "group.csv")
    p.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    p.add_argument("--pilot-ns", type=str, default="4,7,10")
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
                   help="Default raw-pool uses PCAM synthetic feature tables and recomputes Gemelli. "
                        "Use ordination for the MVN/PCoA reference workflow.")
    p.add_argument("--pool-size-per-group", type=int, default=1000)
    p.add_argument("--boot-number", type=int, default=100)
    p.add_argument("--permutations", type=int, default=99)
    p.add_argument("--increase-num", type=int, default=18, help="Half of the power-uniform target count.")
    p.add_argument("--decrease-num", type=int, default=18)
    p.add_argument("--center-mode", type=str, default="omega-calibrated",
                   choices=["observed", "debiased", "empirical-bayes", "omega-calibrated"])
    p.add_argument("--cov-estimator", type=str, default="ledoit-wolf",
                   choices=["ledoit-wolf", "ledoit-wolf-ddof", "nonlinear-shrinkage"],
                   help="Per-group covariance shrinkage: linear Ledoit-Wolf (default); "
                        "'ledoit-wolf-ddof' adds an n/(n-1) trace de-bias that removes small-pilot "
                        "optimism; 'nonlinear-shrinkage' (LW 2020) not recommended for small n.")
    p.add_argument("--embed-dim", type=int, default=None,
                   help="Standardise the PCoA embedding to this many top axes across all pilots. "
                        "Removes the pilot-size dimensionality artefact (small pilot -> few axes -> "
                        "inflated power). Must be <= the axes a small pilot yields (~2*pilot_n-1).")
    p.add_argument("--cov-eb-pool", action="store_true",
                   help="Empirical-Bayes PARTIAL pooling: shrink each group's covariance toward the "
                        "pooled one by lambda=k/(k+n) (data-driven; not full pooling).")
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
    p.add_argument("--ordination-enhance-max", type=float, default=3.0)
    p.add_argument("--omega-calibrated-max-scale", type=float, default=20.0)
    p.add_argument("--adaptive-reference-n", type=float, default=17.0)
    p.add_argument("--power-preview-boot-number", type=int, default=20)
    p.add_argument("--power-preview-permutations", type=int, default=49)
    p.add_argument("--omega-grid-candidates", type=int, default=60)
    p.add_argument("--pcam-gene-blocks", type=str, default="auto")
    p.add_argument("--pcam-ndon", type=int, default=1)
    p.add_argument("--pcam-grid", type=str, default="auto", choices=["auto", "omega-uniform", "fixed"],
                   help="PCAM formal effect grid. auto/omega-uniform previews candidate effects and "
                        "selects formal points by realized omega^2; fixed uses the predefined GENE_GRID.")
    p.add_argument("--pcam-power-points", type=int, default=15,
                   help="Number of omega-uniform formal PCAM points in auto grid mode.")
    p.add_argument("--pcam-near-zero-points", type=int, default=4,
                   help="Number of formal PCAM points reserved for omega^2 values near zero.")
    p.add_argument("--pcam-near-zero-omega-max", type=str, default="auto",
                   help="Upper omega^2 bound for near-zero support points, or 'auto'.")
    p.add_argument("--pcam-pi-candidates", type=int, default=17,
                   help="Number of pi values in the PCAM auto preview grid.")
    p.add_argument("--pcam-scale-candidates", type=int, default=6,
                   help="Number of scale values in the PCAM auto preview grid.")
    p.add_argument("--pcam-scale-max", type=float, default=1.7,
                   help="Maximum PCAM scale in the ordinary auto preview grid.")
    p.add_argument("--pcam-scale-extend-max", type=float, default=3.0,
                   help="Optional upper scale used only if the ordinary preview does not bracket the target omega^2.")
    p.add_argument("--fit-bin-width", type=float, default=0.003)
    p.add_argument("--fit-power-min", type=float, default=0.15)
    p.add_argument("--omega2-floor", type=float, default=0.0)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--scenario-n-jobs", type=int, default=4,
                   help="Parallel workers for raw-pool scenario precomputation.")
    p.add_argument("--random-seed", type=int, default=20260614)
    p.add_argument("--sim-seed", type=int, default=20260614, help="Seed for simulating pilots > observed n.")
    p.add_argument("--out", type=Path, required=True)
    # Keep effect-grid identity for ordination helpers.
    p.set_defaults(effect_grid="power-uniform")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    group_map = _read_group_map(args.group)
    table, agm = _read_taxon_feature_table(args.table, group_map)
    observed_n = int(agm.value_counts().min())
    if args.eval_n is None:
        args.eval_n = observed_n

    if args.engine == "raw-pool":
        if args.eval_ns:
            raise ValueError("gene raw-pool currently supports --eval-n; use --engine ordination for --eval-ns.")
        pilot_ns = [int(x) for x in str(args.pilot_ns).split(",") if x.strip()]
        print(f"[gene raw] observed n/group={observed_n}, eval_n={args.eval_n}, pilots={pilot_ns}", flush=True)
        d = _build_gene_raw_dict(table, agm, args)
        frames = [raw_pcam_curve(d, args, pn, args.random_seed + pn * 1009) for pn in pilot_ns]
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(args.out / "gene_power_curves_raw.csv", index=False)
        _plot_raw_power(raw, args, modality="Gene (Gemelli)", stem="gene")
        (args.out / "summary.json").write_text(json.dumps({
            "modality": "Gene (Gemelli)",
            "engine": "raw-pool",
            "generator": "PCAM",
            "eval_n": int(args.eval_n),
            "pilot_ns": pilot_ns,
            "target_power": args.target_power,
            "target_omega2": args.target_omega2,
            "pool_size_per_group": args.pool_size_per_group,
            "boot_number": args.boot_number,
            "permutations": args.permutations,
            "pcam_ndon": args.pcam_ndon,
        }, indent=2), encoding="utf-8")
        print(f"[gene raw] done -> {args.out}", flush=True)
        return

    def _build_for_pilot(pn: int):
        seed = args.random_seed + pn * 1009
        if pn <= observed_n:
            sub = core.np.random.default_rng(seed)
            picked: List[str] = []
            for g in sorted(agm.unique()):
                members = agm[agm == g].index.to_numpy()
                picked.extend(sub.choice(members, size=pn, replace=False).tolist())
            return build_gene_model(table[picked].copy(), agm.loc[picked], args, stem=f"pilot{pn}"), "real", seed
        stable, pgm, _ = generate_taxon_pool(
            table, agm, pool_size_per_group=pn, random_seed=args.sim_seed,
            between_scale=1.0, residual_scale=1.0, noise_multiplier=0.10,
        )
        return build_gene_model(stable, pgm, args, stem=f"pilot{pn}"), "simulated", seed

    if args.eval_ns:
        eval_ns = [int(x) for x in str(args.eval_ns).split(",") if x.strip()]
        pn = int(args.sweep_pilot if args.sweep_pilot is not None else observed_n)
        print(f"[gene] EVAL-SWEEP: pilot={pn}, eval_ns={eval_ns}", flush=True)
        model, source, seed = _build_for_pilot(pn)
        sweep_rows: List[pd.DataFrame] = []
        for en in eval_ns:
            df = pilot_curve(model, args, seed + en * 17, eval_n=en)
            df.insert(0, "eval_n", en)
            sweep_rows.append(df)
            ntrans = int(((df["power"] >= 0.15) & (df["power"] <= 0.95)).sum())
            print(f"[gene] eval_n={en}: {len(df)} points, {ntrans} in transition", flush=True)
        raw = pd.concat(sweep_rows, ignore_index=True)
        raw["pilot_n"] = pn
        raw["pilot_source"] = source
        raw.to_csv(args.out / "gene_evalsweep_raw.csv", index=False)
        _plot_eval_sweep(raw, args, pn, float(model.get("pilot_target_omega2") or np.nan), modality="Gene (Gemelli)")
        print(f"[gene] done (eval-sweep) -> {args.out}", flush=True)
        return

    pilot_ns = [int(x) for x in str(args.pilot_ns).split(",") if x.strip()]
    print(f"[gene] observed n/group={observed_n}, eval_n={args.eval_n}, pilots={pilot_ns}", flush=True)

    all_rows: List[pd.DataFrame] = []
    for pn in pilot_ns:
        seed = args.random_seed + pn * 1009
        if pn <= observed_n:
            sub = core.np.random.default_rng(seed)
            picked: List[str] = []
            for g in sorted(agm.unique()):
                members = agm[agm == g].index.to_numpy()
                picked.extend(sub.choice(members, size=pn, replace=False).tolist())
            ptable = table[picked].copy()
            pgm = agm.loc[picked]
            source = "real"
        else:
            stable, pgm, _ = generate_taxon_pool(
                table, agm, pool_size_per_group=pn, random_seed=args.sim_seed,
                between_scale=1.0, residual_scale=1.0, noise_multiplier=0.10,
            )
            ptable = stable
            source = "simulated"
        model = build_gene_model(ptable, pgm, args, stem=f"pilot{pn}")
        df = pilot_curve(model, args, seed)
        df.insert(0, "pilot_n", pn)
        df.insert(1, "pilot_source", source)
        df["k_axes"] = model["k"]
        df["center_shrinkage"] = model.get("center_shrinkage")
        df["pilot_target_omega2"] = model.get("pilot_target_omega2")
        all_rows.append(df)
        ntrans = int(((df["power"] >= 0.15) & (df["power"] <= 0.95)).sum())
        print(f"[gene] pilot_n={pn} ({source}): {len(df)} points, {ntrans} in transition", flush=True)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(args.out / "gene_power_curves_raw.csv", index=False)
    _summarize_and_plot(raw, args, modality="Gene (Gemelli)")
    print(f"[gene] done -> {args.out}", flush=True)


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
    fig.savefig(args.out / "gene_power_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(args.out / "gene_sigmoid_summary.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps({
        "modality": modality, "engine": "ordination", "effect_grid": "power-uniform",
        "eval_n": int(args.eval_n), "pilot_ns": [int(x) for x in pilots], "target_power": args.target_power,
        "target_omega2": args.target_omega2, "pool_size_per_group": args.pool_size_per_group,
        "boot_number": args.boot_number, "permutations": args.permutations,
        "center_mode": args.center_mode,
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
    fig.savefig(args.out / "gene_evalsweep_curves.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(fit_rows).to_csv(args.out / "gene_evalsweep_sigmoid_summary.csv", index=False)


if __name__ == "__main__":
    main()
