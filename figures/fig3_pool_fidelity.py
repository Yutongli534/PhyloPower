#!/usr/bin/env python3
"""Manuscript Figure 3 - statistical fidelity audit for synthetic sample pools.

The main figure is intentionally compact:
  A) sample-level overlap after recomputing the modality-specific distance
  B) real-split calibrated fidelity dashboard
  C) power-relevant distance/effect fidelity
  D) metaproteomic taxon-function structure fidelity

This script rebuilds the synthetic pools and recomputes every metric (heavy;
the gene side needs the QIIME 2 / Gemelli environment):

  /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig3_pool_fidelity.py

Figure files land in figures/output/; the validation tables and the standalone
protein-structure panel go to data/archived_runs/fig2_fidelity_audit/.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from phylopower import core  # noqa: E402  (import first: installs the embedded-module finder)
import figstyle  # noqa: E402
import pcam_gen as P  # noqa: E402
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
from semisynthetic_power import _pcoa_coords  # noqa: E402

core.load_core_runtime()
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
    "font.size": 12.8,
    "axes.titlesize": 13.8,
    "axes.labelsize": 12.8,
    "xtick.labelsize": 11.6,
    "ytick.labelsize": 11.6,
    "legend.fontsize": 10.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

REAL = "#2f3742"
SYN = "#e07b39"
PROTEIN = "#159c76"
GENE = "#756bb1"
REAL_DOT = "#111827"
SYN_DOT = "#f97316"
WITHIN = "#0072B2"
BETWEEN = "#D55E00"
GROUP_COLORS = ["#2f6db0", "#b03a3a"]
GRID = "#e5e7eb"
STRUCT_TAXON = "#009E73"
STRUCT_FUNCTION = "#CC79A7"
STRUCT_EDGE = "#4B5563"


@dataclass(frozen=True)
class AuditMetric:
    key: str
    label: str
    modality: str
    family: str
    comparator: Callable[[dict, dict], float]
    floor: float


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(abs(den), 1e-12))


def _ratio_error(value: float) -> float:
    return abs(float(np.log2(max(value, 1e-12))))


# --- fidelity metrics (inlined from the retired analysis/fidelity_common.py) ---


def _within_between(dm: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ids = list(dm.index)
    arr = dm.loc[ids, ids].to_numpy(dtype=float)
    lab = labels.loc[ids].to_numpy()
    iu = np.triu_indices(arr.shape[0], 1)
    vals = arr[iu]
    same = lab[iu[0]] == lab[iu[1]]
    return vals[same], vals[~same]


def _eigenspectrum(dm: pd.DataFrame, sgm: pd.Series) -> dict[str, np.ndarray | float]:
    ids = list(dm.index)
    coords = _pcoa_coords(dm.loc[ids, ids]).to_numpy()
    labels = sgm.loc[ids].to_numpy()
    resid = []
    for g in pd.unique(labels):
        sub = coords[labels == g]
        resid.append(sub - sub.mean(axis=0, keepdims=True))
    R = np.vstack(resid)
    cov = np.cov(R, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.sort(eig[eig > 1e-12])[::-1]
    if eig.size == 0:
        return {"eig": np.array([]), "prop": np.array([]), "deff": float("nan")}
    prop = eig / eig.sum()
    deff = float((eig.sum() ** 2) / np.sum(eig ** 2))
    return {"eig": eig, "prop": prop, "deff": deff}


def _distance_metrics(real_dm: pd.DataFrame, real_sgm: pd.Series, syn_dm: pd.DataFrame,
                      syn_sgm: pd.Series) -> dict[str, float | np.ndarray | dict]:
    rw, rb = _within_between(real_dm, real_sgm)
    sw, sb = _within_between(syn_dm, syn_sgm)
    real_spec = _eigenspectrum(real_dm, real_sgm)
    syn_spec = _eigenspectrum(syn_dm, syn_sgm)
    om_real = max(0.0, float(core.compute_omega2(real_dm, real_sgm)))
    om_syn = max(0.0, float(core.compute_omega2(syn_dm, syn_sgm)))
    return {
        "within_real": rw,
        "within_syn": sw,
        "between_real": rb,
        "between_syn": sb,
        "within_ks": float(ks_2samp(rw, sw).statistic),
        "between_ks": float(ks_2samp(rb, sb).statistic),
        "omega2_real": om_real,
        "omega2_syn": om_syn,
        "omega2_ratio": _safe_ratio(om_syn, om_real),
        "real_spectrum": real_spec,
        "syn_spectrum": syn_spec,
        "deff_real": float(real_spec["deff"]),
        "deff_syn": float(syn_spec["deff"]),
        "deff_ratio": _safe_ratio(float(syn_spec["deff"]), float(real_spec["deff"])),
    }


def _sample_syn_columns(sgm: pd.Series, n_per_group: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    cols: list[str] = []
    for g in pd.unique(sgm):
        members = sgm[sgm == g].index.to_numpy()
        pick = rng.choice(members, size=min(n_per_group, len(members)), replace=False)
        cols.extend(pick.tolist())
    return cols


def _positive_log_jitter(tab: pd.DataFrame, sd: float, seed: int) -> pd.DataFrame:
    """Break repeated protein templates without changing support or library size."""
    if float(sd) <= 0:
        return tab.copy()
    rng = np.random.default_rng(int(seed))
    arr = tab.to_numpy(dtype=float).copy()
    totals = arr.sum(axis=0)
    mask = arr > 0
    noise = rng.normal(0.0, float(sd), size=arr.shape)
    arr[mask] = np.maximum(0.0, np.expm1(np.log1p(arr[mask]) + noise[mask]))
    new_totals = arr.sum(axis=0)
    ok = new_totals > 0
    arr[:, ok] *= totals[ok] / new_totals[ok]
    return pd.DataFrame(arr, index=tab.index, columns=tab.columns)


def _pseudo_r2(dm: pd.DataFrame, labels: pd.Series) -> float:
    labels = labels.loc[list(dm.index)]
    sst, _, ssa = core._summarize_ss(dm, labels)  # noqa: SLF001 - reuse package's PERMANOVA SS.
    return 0.0 if sst <= 0 else float(max(0.0, ssa / sst))


def _permanova_r2_p(dm: pd.DataFrame, labels: pd.Series, permutations: int = 199) -> tuple[float, float]:
    labels = labels.loc[list(dm.index)]
    r2 = _pseudo_r2(dm, labels)
    p_value = core.compute_permanova_p_value(dm, labels, permutations=permutations)
    return r2, float(p_value)


def _format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def _feature_stats(tab: pd.DataFrame) -> dict[str, np.ndarray | float]:
    arr = tab.to_numpy(dtype=float)
    return {
        "mean": arr.mean(axis=1),
        "variance": arr.var(axis=1),
        "prevalence": (arr > 0).mean(axis=1),
        "library_size": arr.sum(axis=0),
        "sample_sparsity": (arr == 0).mean(axis=0),
    }


def _distance_to_centroid(dm: pd.DataFrame, sgm: pd.Series) -> np.ndarray:
    ids = list(dm.index)
    coords = _pcoa_coords(dm.loc[ids, ids]).to_numpy()
    labels = sgm.loc[ids].to_numpy()
    vals: list[np.ndarray] = []
    for g in pd.unique(labels):
        idx = labels == g
        if idx.sum() <= 1:
            continue
        sub = coords[idx]
        vals.append(np.linalg.norm(sub - sub.mean(axis=0, keepdims=True), axis=1))
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def _protein_topology_summary(d: dict, tab: pd.DataFrame) -> dict[str, np.ndarray | float]:
    taxa = d["uid"].astype(int)
    funcs = d["meta"]["Function"].astype(str).to_numpy()
    _, func_ids = np.unique(funcs, return_inverse=True)
    n_taxa = int(taxa.max()) + 1
    n_funcs = int(func_ids.max()) + 1
    mask = tab.to_numpy(dtype=float) > 0
    tax_deg: list[int] = []
    fun_deg: list[int] = []
    tax_prev = np.zeros(n_taxa, dtype=float)
    fun_prev = np.zeros(n_funcs, dtype=float)
    for i in range(mask.shape[1]):
        present = mask[:, i]
        if not np.any(present):
            continue
        p_taxa = taxa[present]
        p_funcs = func_ids[present]
        tax_prev += np.bincount(p_taxa, minlength=n_taxa) > 0
        fun_prev += np.bincount(p_funcs, minlength=n_funcs) > 0
        for t in np.unique(p_taxa):
            tax_deg.append(int(np.unique(p_funcs[p_taxa == t]).size))
        for f in np.unique(p_funcs):
            fun_deg.append(int(np.unique(p_taxa[p_funcs == f]).size))
    edge_count = mask.sum(axis=0).astype(float)
    return {
        "taxon_degree": np.asarray(tax_deg, dtype=float),
        "function_degree": np.asarray(fun_deg, dtype=float),
        "edge_count": edge_count,
        "connectance": float(edge_count.mean() / max(mask.shape[0], 1)),
        "taxon_prevalence": tax_prev / max(mask.shape[1], 1),
        "function_prevalence": fun_prev / max(mask.shape[1], 1),
    }


def _split_real_ids(sgm: pd.Series, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    a: list[str] = []
    b: list[str] = []
    for group in pd.unique(sgm):
        ids = sgm[sgm == group].index.to_numpy()
        ids = rng.permutation(ids)
        cut = max(2, len(ids) // 2)
        if len(ids) - cut < 2:
            cut = len(ids) - 2
        a.extend(ids[:cut].tolist())
        b.extend(ids[cut:].tolist())
    return a, b


def _metric_bundle(d: dict | None, tab: pd.DataFrame, sgm: pd.Series, dm: pd.DataFrame) -> dict:
    stats = _feature_stats(tab)
    dist = _distance_metrics(dm, sgm, dm, sgm)
    return {
        "tab": tab,
        "sgm": sgm,
        "dm": dm,
        "stats": stats,
        "topology": _protein_topology_summary(d, tab) if d is not None else None,
        "within": dist["within_real"],
        "between": dist["between_real"],
        "omega2": max(0.0, float(core.compute_omega2(dm, sgm))),
        "deff": float(_eigenspectrum(dm, sgm)["deff"]),
        "dispersion": _distance_to_centroid(dm, sgm),
    }


def _bundle_pair_metrics(a: dict, b: dict, *, modality: str) -> dict[str, float]:
    out = {
        "abundance": max(
            float(ks_2samp(a["stats"]["mean"], b["stats"]["mean"]).statistic),
            float(ks_2samp(a["stats"]["variance"], b["stats"]["variance"]).statistic),
        ),
        "prevalence": float(ks_2samp(a["stats"]["prevalence"], b["stats"]["prevalence"]).statistic),
        "distance": max(
            float(ks_2samp(a["within"], b["within"]).statistic),
            float(ks_2samp(a["between"], b["between"]).statistic),
        ),
        "dispersion": float(ks_2samp(a["dispersion"], b["dispersion"]).statistic),
        "omega2": _ratio_error(_safe_ratio(float(b["omega2"]), float(a["omega2"]))),
        "deff": _ratio_error(_safe_ratio(float(b["deff"]), float(a["deff"]))),
    }
    if modality == "protein":
        ta = a["topology"]
        tb = b["topology"]
        out["tf_topology"] = max(
            float(ks_2samp(ta["taxon_degree"], tb["taxon_degree"]).statistic),
            float(ks_2samp(ta["function_degree"], tb["function_degree"]).statistic),
            float(ks_2samp(ta["edge_count"], tb["edge_count"]).statistic),
            _ratio_error(_safe_ratio(float(tb["connectance"]), float(ta["connectance"]))),
        )
    else:
        out["library_sparsity"] = max(
            float(ks_2samp(a["stats"]["library_size"], b["stats"]["library_size"]).statistic),
            float(ks_2samp(a["stats"]["sample_sparsity"], b["stats"]["sample_sparsity"]).statistic),
        )
    return out


def _real_split_baseline(d: dict | None, real: dict, modality: str, n_splits: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_splits):
        ids_a, ids_b = _split_real_ids(real["sgm"], rng)
        bundle_a = _metric_bundle(
            d,
            real["tab"].loc[:, ids_a],
            real["sgm"].loc[ids_a],
            real["dm"].loc[ids_a, ids_a],
        )
        bundle_b = _metric_bundle(
            d,
            real["tab"].loc[:, ids_b],
            real["sgm"].loc[ids_b],
            real["dm"].loc[ids_b, ids_b],
        )
        vals = _bundle_pair_metrics(bundle_a, bundle_b, modality=modality)
        src = pd.Series(["split_a" if x in set(ids_a) else "split_b" for x in real["dm"].index], index=real["dm"].index)
        vals["source_r2"] = _pseudo_r2(real["dm"], src)
        rows.extend({"modality": modality, "split": i, "metric": k, "value": v} for k, v in vals.items())
    return pd.DataFrame(rows)


def _allowed_thresholds(baseline: pd.DataFrame) -> dict[tuple[str, str], float]:
    floors = {
        "abundance": 0.05,
        "prevalence": 0.05,
        "tf_topology": 0.05,
        "library_sparsity": 0.05,
        "distance": 0.10,
        "dispersion": 0.10,
        "omega2": np.log2(1.10),
        "deff": np.log2(1.10),
        "source_r2": 0.02,
    }
    out: dict[tuple[str, str], float] = {}
    for (modality, metric), sub in baseline.groupby(["modality", "metric"]):
        out[(modality, metric)] = float(max(np.quantile(sub["value"], 0.95), floors.get(metric, 0.05)))
    return out


def _real_objects(modality: str) -> tuple[dict, dict]:
    d = P.load_modality(modality)
    tab, sgm = P.real_table(d)
    dm = P.recompute_distance(d, tab)
    return d, _metric_bundle(d if modality == "protein" else None, tab, sgm, dm)


def _protein_pool(d: dict, m: int, seed: int, params: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    tab, sgm = mdctf_mc_pool(
        d,
        m,
        seed,
        1.0,
        edge_fraction=float(params["edge_fraction"]),
        marginal_strength=params.get("marginal_strength", "auto"),
        eb_k=params.get("eb_k", "auto"),
        residual_mode=params.get("residual_mode", "template"),
    )
    tab = _positive_log_jitter(tab, float(params.get("jitter_sd", 0.0)), seed + 991)
    return tab, sgm, P.recompute_distance(d, tab)


def _gene_pool(d: dict, m: int, seed: int, params: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    tab, sgm = P.pcam_pool(d, m, seed, float(params["pi"]), float(params["scale"]), ndon=1)
    return tab, sgm, P.recompute_distance(d, tab)


def _score_candidate(
    real: dict,
    syn_bundle: dict,
    modality: str,
    allowed: dict[tuple[str, str], float],
) -> dict[str, float]:
    vals = _bundle_pair_metrics(real, syn_bundle, modality=modality)
    return {k: float(v / max(allowed[(modality, k)], 1e-12)) for k, v in vals.items()}


def _select_params(
    modality: str,
    d: dict,
    real: dict,
    allowed: dict[tuple[str, str], float],
    *,
    pool_m: int,
    seeds: list[int],
) -> dict:
    if modality == "protein":
        candidates = [
            {"edge_fraction": ef, "residual_mode": "template", "marginal_strength": "auto", "eb_k": "auto", "jitter_sd": js}
            for ef in [0.75, 0.9, 1.0]
            for js in [0.08, 0.15, 0.20, 0.30]
        ]
        candidates.extend([
            {"edge_fraction": ef, "residual_mode": "random", "marginal_strength": "auto", "eb_k": "auto", "jitter_sd": 0.0}
            for ef in [0.5, 0.75, 1.0]
        ])
        maker = _protein_pool
    else:
        candidates = [{"pi": pi, "scale": sc} for pi, sc in [(0.95, 1.0), (1.0, 1.0), (1.0, 1.2), (1.0, 1.4), (1.0, 1.7), (1.0, 2.0), (1.0, 3.0)]]
        maker = _gene_pool

    best: tuple[float, dict, list[dict]] | None = None
    for cand in candidates:
        seed_scores = []
        print(f"[fig2-audit] optimize {modality} candidate={cand}", flush=True)
        for seed in seeds:
            tab, sgm, dm = maker(d, pool_m, seed, cand)
            syn_bundle = _metric_bundle(d if modality == "protein" else None, tab, sgm, dm)
            scores = _score_candidate(real, syn_bundle, modality, allowed)
            if modality == "protein":
                rw, _ = _within_between(real["dm"], real["sgm"])
                sw, _ = _within_between(dm, sgm)
                real_q02 = float(np.quantile(rw, 0.02))
                syn_q02 = float(np.quantile(sw, 0.02))
                lowq_score = max(0.0, (0.75 * real_q02 - syn_q02) / max(0.75 * real_q02, 1e-12))
                rounded = np.round(tab.T.to_numpy(dtype=float), 8)
                unique_fraction = float(pd.DataFrame(rounded).drop_duplicates().shape[0] / max(rounded.shape[0], 1))
                duplicate_score = max(0.0, (0.98 - unique_fraction) / 0.98)
                scores["lowq_distance"] = lowq_score
                scores["uniqueness"] = duplicate_score
            seed_scores.append({"max_score": max(scores.values()), **scores})
        med = float(np.median([x["max_score"] for x in seed_scores]))
        if best is None or med < best[0]:
            best = (med, cand, seed_scores)
    assert best is not None
    print(f"[fig2-audit] selected {modality}: {best[1]} median_max_score={best[0]:.3f}", flush=True)
    return {"params": best[1], "median_max_score": best[0], "seed_scores": best[2]}


def _bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(values, size=values.size, replace=True)) for _ in range(n_boot)]
    return float(np.median(values)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def _validate_pools(
    modality: str,
    d: dict,
    real: dict,
    params: dict,
    allowed: dict[tuple[str, str], float],
    *,
    pool_m: int,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict, list[dict]]:
    maker = _protein_pool if modality == "protein" else _gene_pool
    rows: list[dict] = []
    bundles: list[dict] = []
    for i, seed in enumerate(seeds):
        print(f"[fig2-audit] validate {modality} pool {i + 1}/{len(seeds)} seed={seed}", flush=True)
        tab, sgm, dm = maker(d, pool_m, seed, params)
        syn = _metric_bundle(d if modality == "protein" else None, tab, sgm, dm)
        syn["seed"] = seed
        bundles.append(syn)
        vals = _bundle_pair_metrics(real, syn, modality=modality)
        source = _coembed(d, real, syn, seed + 7000)
        source_r2, source_p = _permanova_r2_p(source["dm"], source["source"], permutations=199)
        vals["source_r2"] = source_r2
        syn["source_audit"] = {
            "r2": source_r2,
            "p_value": source_p,
            "auc": _knn_auc(source["coords"][:, : min(6, source["coords"].shape[1])], source["source"].to_numpy(), seed),
            **_support_metrics(source["coords"], source["source"].to_numpy(), k=5),
        }
        for metric, raw in vals.items():
            limit = allowed[(modality, metric)]
            rows.append({
                "modality": modality,
                "seed": seed,
                "metric": metric,
                "raw_value": float(raw),
                "allowed": float(limit),
                "normalized": float(raw / max(limit, 1e-12)),
                "pass": bool(raw <= limit),
                "p_value": float(source_p) if metric == "source_r2" else np.nan,
            })
    return pd.DataFrame(rows), bundles[0], bundles


def _representative_bundle(validation: pd.DataFrame, bundles: list[dict], modality: str) -> dict:
    sub = validation[validation["modality"] == modality]
    max_by_seed = sub.groupby("seed")["normalized"].max()
    target = float(max_by_seed.median())
    seed = int((max_by_seed - target).abs().sort_values().index[0])
    for bundle in bundles:
        if int(bundle.get("seed", -1)) == seed:
            return bundle
    return bundles[0]


def _coembed(d: dict, real: dict, syn: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real_cols = list(real["tab"].columns)
    n_real_by_group = real["sgm"].value_counts().to_dict()
    syn_cols: list[str] = []
    for g, n in n_real_by_group.items():
        members = syn["sgm"][syn["sgm"] == g].index.to_numpy()
        take = min(len(members), int(n))
        syn_cols.extend(rng.choice(members, size=take, replace=False).tolist())
    syn_tab = syn["tab"].loc[:, syn_cols].copy()
    rename = {c: f"syn_{i:03d}" for i, c in enumerate(syn_cols)}
    syn_tab = syn_tab.rename(columns=rename)
    syn_sgm = syn["sgm"].loc[syn_cols].rename(index=rename)
    comb = pd.concat([real["tab"], syn_tab], axis=1)
    dm = P.recompute_distance(d, comb)
    order = list(comb.columns)
    coords = _pcoa_coords(dm.loc[order, order]).to_numpy()
    source = pd.Series(["real"] * len(real_cols) + ["synthetic"] * len(rename), index=order)
    labels = pd.concat([real["sgm"], syn_sgm]).loc[order]
    return {"dm": dm.loc[order, order], "coords": coords, "source": source, "labels": labels}


def _support_metrics(coords: np.ndarray, source: np.ndarray, k: int = 5) -> dict[str, float]:
    real = coords[source == "real"]
    syn = coords[source == "synthetic"]
    if len(real) <= k + 1 or len(syn) <= k + 1:
        return {"precision": float("nan"), "recall": float("nan"), "authenticity": float("nan")}
    rr = NearestNeighbors(n_neighbors=min(k + 1, len(real))).fit(real)
    real_radius = rr.kneighbors(real, return_distance=True)[0][:, -1]
    ss = NearestNeighbors(n_neighbors=min(k + 1, len(syn))).fit(syn)
    syn_radius = ss.kneighbors(syn, return_distance=True)[0][:, -1]
    sr_dist, sr_idx = rr.kneighbors(syn, n_neighbors=1, return_distance=True)
    rs_dist, rs_idx = ss.kneighbors(real, n_neighbors=1, return_distance=True)
    precision = float(np.mean(sr_dist[:, 0] <= real_radius[sr_idx[:, 0]]))
    recall = float(np.mean(rs_dist[:, 0] <= syn_radius[rs_idx[:, 0]]))
    real_nn = rr.kneighbors(real, return_distance=True)[0][:, 1]
    authenticity = float(np.mean(sr_dist[:, 0] >= np.quantile(real_nn, 0.01)))
    return {"precision": precision, "recall": recall, "authenticity": authenticity}


def _knn_auc(coords: np.ndarray, source: np.ndarray, seed: int) -> float:
    y = (source == "synthetic").astype(int)
    if min(np.bincount(y)) < 4:
        return float("nan")
    n_splits = int(min(5, np.bincount(y).min()))
    k = int(min(5, max(1, np.bincount(y).min() - 1)))
    clf = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        return float(np.mean(cross_val_score(clf, coords, y, cv=cv, scoring="roc_auc")))
    except Exception:
        clf.fit(coords, y)
        return float(roc_auc_score(y, clf.predict_proba(coords)[:, 1]))


def _density_contour(ax, pts: np.ndarray, color: str, ls: str) -> None:
    if len(pts) < 8:
        return
    try:
        kde = gaussian_kde(pts[:, :2].T)
    except Exception:
        return
    xpad = (pts[:, 0].max() - pts[:, 0].min()) * 0.15 + 1e-9
    ypad = (pts[:, 1].max() - pts[:, 1].min()) * 0.15 + 1e-9
    xx, yy = np.mgrid[
        pts[:, 0].min() - xpad : pts[:, 0].max() + xpad : 80j,
        pts[:, 1].min() - ypad : pts[:, 1].max() + ypad : 80j,
    ]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz[zz > 0], [0.70, 0.86, 0.94])
    ax.contour(xx, yy, zz, levels=np.unique(levels), colors=color, linewidths=1.0, linestyles=ls, alpha=0.85)


def _letter(ax, letter: str, title: str) -> None:
    ax.text(-0.06, 1.095, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=20, fontweight="bold")
    ax.text(0.065, 1.105, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold")


def _plot_panel_a(fig: plt.Figure, spec, embeddings: dict, sample_stats: dict) -> None:
    outer = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.22)
    first_ax = None
    for i, (modality, color, title) in enumerate([("protein", PROTEIN, "Protein / PhyloFunc"), ("gene", GENE, "Gene / Gemelli")]):
        ax = fig.add_subplot(outer[0, i])
        if first_ax is None:
            first_ax = ax
            _letter(ax, "a", "Sample-level overlap")
        emb = embeddings[modality]
        coords = emb["coords"]
        source = emb["source"].to_numpy()
        groups = emb["labels"].to_numpy()
        for gi, group in enumerate(pd.unique(groups)):
            for src, marker, size, alpha, edge in [
                ("synthetic", "o", 15, 0.22, "none"),
                ("real", "o", 36, 0.95, "white"),
            ]:
                mask = (groups == group) & (source == src)
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=size,
                    marker=marker,
                    color=GROUP_COLORS[gi % 2],
                    alpha=alpha,
                    edgecolors=edge,
                    linewidths=0.6,
                    label=f"{src} {group}" if i == 0 else None,
                    rasterized=True,
                )
        _density_contour(ax, coords[source == "real"], REAL, "-")
        _density_contour(ax, coords[source == "synthetic"], SYN, "--")
        st = sample_stats[modality]
        txt = (
            f"source R2 {st['source_r2']:.3f}\n"
            f"kNN AUC {st['auc']:.2f}\n"
            f"prec/rec {st['precision']:.2f}/{st['recall']:.2f}"
        )
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#dddddd", alpha=0.86))
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("PCoA 1")
        ax.set_ylabel("PCoA 2" if i == 0 else "")
        ax.grid(color=GRID, lw=0.5, alpha=0.7)
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        first_ax.legend(handles[:4], labels[:4], loc="lower left", fontsize=6.5, frameon=False)


def _mean_var_limits(real: dict, syn: dict) -> tuple[float, float, float, float]:
    xs = np.concatenate([np.asarray(real["stats"]["mean"], dtype=float), np.asarray(syn["stats"]["mean"], dtype=float)])
    ys = np.concatenate([np.asarray(real["stats"]["variance"], dtype=float), np.asarray(syn["stats"]["variance"], dtype=float)])
    pos = np.concatenate([xs[xs > 0], ys[ys > 0]])
    eps = float(np.quantile(pos, 0.01) * 0.5) if pos.size else 1e-9
    x = xs + eps
    y = ys + eps
    return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))


def _plot_panel_a_feature(fig: plt.Figure, spec, real: dict, reps: dict) -> None:
    outer = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.22)
    for i, (modality, color, title) in enumerate([("gene", GENE, "Gene abundance fidelity"), ("protein", PROTEIN, "Protein abundance fidelity")]):
        ax = fig.add_subplot(outer[0, i])
        if i == 0:
            _letter(ax, "a", "Feature-level fidelity")
        rb = real[modality]
        sb = reps[modality]
        xmin, xmax, ymin, ymax = _mean_var_limits(rb, sb)
        eps = max(xmin, ymin, 1e-12)
        rng = np.random.default_rng(11 + i)
        for source, bundle, c, alpha, z in [("real", rb, REAL_DOT, 0.28, 1), ("synthetic", sb, SYN_DOT, 0.30, 2)]:
            mu = np.asarray(bundle["stats"]["mean"], dtype=float)
            var = np.asarray(bundle["stats"]["variance"], dtype=float)
            idx = np.arange(mu.size)
            if idx.size > 3000:
                idx = rng.choice(idx, size=3000, replace=False)
            ax.scatter(mu[idx] + eps, var[idx] + eps, s=7, color=c, alpha=alpha, linewidths=0, label=source, rasterized=True, zorder=z)
        mean_ks = float(ks_2samp(rb["stats"]["mean"], sb["stats"]["mean"]).statistic)
        var_ks = float(ks_2samp(rb["stats"]["variance"], sb["stats"]["variance"]).statistic)
        prev_ks = float(ks_2samp(rb["stats"]["prevalence"], sb["stats"]["prevalence"]).statistic)
        ax.text(0.02, 0.98, f"mean KS {mean_ks:.2f}\nvariance KS {var_ks:.2f}\nprevalence KS {prev_ks:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#dddddd", alpha=0.86))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=14.0)
        ax.set_xlabel("feature mean abundance")
        ax.set_ylabel("feature variance" if i == 0 else "")
        ax.grid(color=GRID, lw=0.5, alpha=0.7)
        if i == 0:
            ax.legend(loc="lower right", fontsize=11.0, frameon=False)


def _plot_panel_b(ax, validation: pd.DataFrame) -> None:
    _letter(ax, "b", "Real-split calibrated fidelity")
    order = [
        ("gene", "source_r2", "Gene source R2"),
        ("gene", "abundance", "Gene abundance"),
        ("gene", "prevalence", "Gene prevalence"),
        ("gene", "library_sparsity", "Gene library/sparsity"),
        ("gene", "distance", "Gene distance"),
        ("gene", "dispersion", "Gene dispersion"),
        ("gene", "omega2", "Gene omega2"),
        ("gene", "deff", "Gene d_eff"),
        ("protein", "source_r2", "Protein source R2"),
        ("protein", "abundance", "Protein abundance"),
        ("protein", "prevalence", "Protein prevalence"),
        ("protein", "tf_topology", "TF topology"),
        ("protein", "distance", "Protein distance"),
        ("protein", "dispersion", "Protein dispersion"),
        ("protein", "omega2", "Protein omega2"),
        ("protein", "deff", "Protein d_eff"),
    ]
    rows = []
    for modality, metric, label in order:
        vals = validation[(validation["modality"] == modality) & (validation["metric"] == metric)]["normalized"].to_numpy(float)
        med, lo, hi = _bootstrap_ci(vals, seed=17)
        rows.append((modality, metric, label, med, lo, hi))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0, 1.0, color="#e8f3e7", zorder=0)
    ax.axvspan(1.0, 1.02, color="#fbf1cf", zorder=0)
    ax.axvline(1.0, color="#565656", lw=1.0, ls="--")
    for yi, (modality, _, label, med, lo, hi) in zip(y, rows):
        color = PROTEIN if modality == "protein" else GENE
        ax.hlines(yi, lo, hi, color=color, lw=2.0, alpha=0.65)
        ax.scatter(med, yi, s=34, color=color, edgecolors="white", linewidths=0.7, zorder=3)
        ax.text(min(max(med, hi) + 0.030, 0.97), yi, f"{med:.2f}", ha="left", va="center", fontsize=11.0)
    ax.set_yticks(y, [r[2] for r in rows], fontsize=11.3)
    ax.set_xlabel("synthetic discrepancy / allowed real-split discrepancy")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.55, alpha=0.75)
    ax.text(0.98, 0.03, "pass <= 1", transform=ax.transAxes, ha="right", va="bottom", fontsize=11.0, color="#425342")


def _density_line(ax, values: np.ndarray, yoff: float, color: str, ls: str, label: str) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max() + 1e-6)
    x = np.linspace(lo, hi, 260)
    try:
        y = gaussian_kde(values)(x)
    except Exception:
        hist, edges = np.histogram(values, bins=30, range=(lo, hi), density=True)
        x = (edges[:-1] + edges[1:]) / 2
        y = hist
    y = y / max(y.max(), 1e-12) * 0.34
    ax.plot(x, yoff + y, color=color, lw=1.8, ls=ls, label=label)


def _ratio_label(validation: pd.DataFrame, modality: str, metric: str, *, as_ratio: bool) -> str:
    sub = validation[(validation["modality"] == modality) & (validation["metric"] == metric)]
    raw = sub["raw_value"].to_numpy(float)
    if as_ratio:
        ratios = np.power(2.0, raw)
        med, lo, hi = _bootstrap_ci(ratios, seed=31)
        return f"{metric}: {med:.2f} [{lo:.2f}, {hi:.2f}]"
    med, lo, hi = _bootstrap_ci(raw, seed=31)
    return f"{metric}: {med:.2f} [{lo:.2f}, {hi:.2f}]"


def _plot_panel_c(fig: plt.Figure, spec, real: dict, reps: dict, validation: pd.DataFrame) -> None:
    outer = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=0.50)
    first = None
    for i, (modality, color, title) in enumerate([("gene", GENE, "Gene distance distributions"), ("protein", PROTEIN, "Protein distance distributions")]):
        ax = fig.add_subplot(outer[i, 0])
        if first is None:
            first = ax
            _letter(ax, "c", "Power-relevant fidelity")
        rb = real[modality]
        sb = reps[modality]
        qs = np.linspace(0.02, 0.98, 70)
        all_q = []
        for rkey, skey, label, marker, alpha in [
            ("within", "within", "within", "o", 0.72),
            ("between", "between", "between", "^", 0.72),
        ]:
            rq = np.quantile(np.asarray(rb[rkey], dtype=float), qs)
            sq = np.quantile(np.asarray(sb[skey], dtype=float), qs)
            all_q.extend([rq, sq])
            ax.scatter(rq, sq, s=12, color=WITHIN if label == "within" else BETWEEN,
                       marker=marker, alpha=alpha, linewidths=0, label=label, rasterized=True)
        lo = float(min(np.min(x) for x in all_q))
        hi = float(max(np.max(x) for x in all_q))
        pad = 0.06 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#666666", lw=1.0, ls="--", zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("real distance quantile")
        ax.set_ylabel("synthetic distance quantile")
        ax.set_title(title.replace("distributions", "quantile agreement"), fontsize=9.5)
        ax.grid(color=GRID, lw=0.55, alpha=0.7)
        w_ks = float(ks_2samp(rb["within"], sb["within"]).statistic)
        b_ks = float(ks_2samp(rb["between"], sb["between"]).statistic)
        txt = "\n".join([
            f"KS w/b {w_ks:.2f}/{b_ks:.2f}",
            _ratio_label(validation, modality, "omega2", as_ratio=True).replace("omega2", "ω² fold-error"),
            _ratio_label(validation, modality, "deff", as_ratio=True).replace("deff", "d_eff fold-error"),
        ])
        ax.text(0.98, 0.94, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.1,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#dddddd", alpha=0.86))
        if i == 0:
            ax.legend(loc="lower right", fontsize=6.5, ncol=2, frameon=False)


def _low_tail_summary(real_vals: np.ndarray, syn_vals_by_pool: list[np.ndarray]) -> tuple[float, float]:
    real_vals = np.asarray(real_vals, dtype=float)
    threshold = float(np.quantile(real_vals, 0.05))
    zero_fracs = []
    low_fracs = []
    for vals in syn_vals_by_pool:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        zero_fracs.append(float(np.mean(vals <= 1e-12)))
        low_fracs.append(float(np.mean(vals <= threshold)))
    return float(np.median(zero_fracs)), float(np.median(low_fracs))


def _plot_panel_c_validation(fig: plt.Figure, spec, real: dict, bundles: dict, validation: pd.DataFrame) -> None:
    """Distance agreement across all validation pools."""
    outer = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[1.0, 0.43],
        hspace=0.36,
        wspace=0.04,
    )
    first = None
    for i, (modality, title) in enumerate([("gene", "Gene distance agreement"), ("protein", "Protein distance agreement")]):
        ax = fig.add_subplot(outer[i, 0])
        tx = fig.add_subplot(outer[i, 1])
        tx.axis("off")
        if first is None:
            first = ax
            _letter(ax, "c", "Power-relevant fidelity")
        rb = real[modality]
        pool_bundles = bundles[modality]
        combined = np.concatenate([
            np.asarray(rb["within"], dtype=float),
            np.asarray(rb["between"], dtype=float),
            *[np.asarray(bundle["within"], dtype=float) for bundle in pool_bundles],
            *[np.asarray(bundle["between"], dtype=float) for bundle in pool_bundles],
        ])
        combined = combined[np.isfinite(combined)]
        x_max = float(np.quantile(combined, 0.995)) if combined.size else 1.0
        x = np.linspace(0.0, max(x_max, 1e-6), 260)
        for kind, label, color, yoff in [
            ("within", "within", WITHIN, 1.0),
            ("between", "between", BETWEEN, 0.0),
        ]:
            real_vals = np.asarray(rb[kind], dtype=float)
            syn_vals = [np.asarray(bundle[kind], dtype=float) for bundle in pool_bundles]
            try:
                real_density = gaussian_kde(real_vals[np.isfinite(real_vals)])(x)
            except Exception:
                real_density, edges = np.histogram(real_vals, bins=50, range=(0, x[-1]), density=True)
                x = (edges[:-1] + edges[1:]) / 2
            syn_density_rows = []
            for vals in syn_vals:
                vals = vals[np.isfinite(vals)]
                try:
                    syn_density_rows.append(gaussian_kde(vals)(x))
                except Exception:
                    hist, _ = np.histogram(vals, bins=len(x), range=(0, x[-1]), density=True)
                    syn_density_rows.append(hist[: len(x)])
            syn_density = np.vstack(syn_density_rows)
            scale = max(float(np.nanmax(real_density)), float(np.nanmax(syn_density)), 1e-12)
            real_density = real_density / scale * 0.55
            syn_med = np.nanmedian(syn_density / scale * 0.55, axis=0)
            syn_lo = np.nanquantile(syn_density / scale * 0.55, 0.025, axis=0)
            syn_hi = np.nanquantile(syn_density / scale * 0.55, 0.975, axis=0)
            ax.fill_between(x, yoff + syn_lo, yoff + syn_hi, color=color, alpha=0.16, linewidth=0)
            ax.plot(x, yoff + real_density, color=color, lw=1.65, ls="-", label=f"real {label}")
            ax.plot(x, yoff + syn_med, color=color, lw=1.8, ls="--", label=f"synthetic {label}")
            ax.axhline(yoff, color="#d1d5db", lw=0.7, zorder=0)
        ax.set_xlim(0, x[-1])
        ax.set_ylim(-0.08, 1.72)
        ax.set_yticks([0.0, 1.0], ["between", "within"], fontsize=11.5)
        ax.set_xlabel("pairwise distance" if i == 1 else "")
        ax.set_ylabel("")
        ax.set_title("")
        ax.text(0.02, 1.035, title.replace(" agreement", ""), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13.0, fontweight="bold")
        ax.grid(axis="x", color=GRID, lw=0.55, alpha=0.7)
        w_ks = float(np.median([
            ks_2samp(rb["within"], bundle["within"]).statistic
            for bundle in pool_bundles
        ]))
        b_ks = float(np.median([
            ks_2samp(rb["between"], bundle["between"]).statistic
            for bundle in pool_bundles
        ]))
        zero_frac, low_frac = _low_tail_summary(rb["within"], [bundle["within"] for bundle in pool_bundles])
        txt = "\n".join([
            f"KS w/b {w_ks:.2f}/{b_ks:.2f}",
            f"low-tail {100 * low_frac:.1f}%\nzero {100 * zero_frac:.1f}%",
            _ratio_label(validation, modality, "omega2", as_ratio=True).replace("omega2", "ω² FE"),
            _ratio_label(validation, modality, "deff", as_ratio=True).replace("deff", "d_eff FE"),
        ])
        tx.text(0.01, 0.90, txt, transform=tx.transAxes, ha="left", va="top", fontsize=10.8,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#dddddd", alpha=0.92))
        if i == 0:
            y0 = 0.38
            legend_items = [
                (WITHIN, "-", "real within"),
                (WITHIN, "--", "synthetic within"),
                (BETWEEN, "-", "real between"),
                (BETWEEN, "--", "synthetic between"),
            ]
            for j, (c, ls, label) in enumerate(legend_items):
                y = y0 - 0.085 * j
                tx.plot([0.02, 0.21], [y, y], transform=tx.transAxes, color=c, ls=ls, lw=1.8, clip_on=False)
                tx.text(0.27, y, label, transform=tx.transAxes, ha="left", va="center", fontsize=10.2)


def _distance_tail_diagnostics(real: dict, bundles: dict) -> dict:
    out = {}
    for modality, rb in real.items():
        pool_bundles = bundles[modality]
        zero_frac, low_frac = _low_tail_summary(rb["within"], [bundle["within"] for bundle in pool_bundles])
        out[modality] = {
            "within_zero_fraction_median": zero_frac,
            "within_below_real_5pct_fraction_median": low_frac,
            "visual_display": "ridge density overlay clipped at the pooled 99.5th distance percentile",
            "ks_uses_full_pairwise_distribution": True,
        }
    return out


def _append_ks_test(rows: list[dict], modality: str, seed: int, block: str, metric: str, real_vals, syn_vals) -> None:
    res = ks_2samp(np.asarray(real_vals, dtype=float), np.asarray(syn_vals, dtype=float))
    rows.append({
        "modality": modality,
        "seed": seed,
        "block": block,
        "metric": metric,
        "test": "two-sample KS",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "p_gt_0.05": bool(float(res.pvalue) > 0.05),
    })


def _literature_standard_tests(real: dict, bundles: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for modality, rb in real.items():
        for bundle in bundles[modality]:
            seed = int(bundle.get("seed", -1))
            for metric in ["mean", "variance", "prevalence"]:
                _append_ks_test(rows, modality, seed, "feature", metric, rb["stats"][metric], bundle["stats"][metric])
            for metric in ["within", "between"]:
                _append_ks_test(rows, modality, seed, "distance", metric, rb[metric], bundle[metric])
            _append_ks_test(rows, modality, seed, "dispersion", "distance_to_centroid", rb["dispersion"], bundle["dispersion"])
            if modality == "gene":
                for metric in ["library_size", "sample_sparsity"]:
                    _append_ks_test(rows, modality, seed, "sample", metric, rb["stats"][metric], bundle["stats"][metric])
            else:
                rt = rb["topology"]
                st = bundle["topology"]
                for metric in ["taxon_degree", "function_degree", "edge_count"]:
                    _append_ks_test(rows, modality, seed, "tf_topology", metric, rt[metric], st[metric])
            source = bundle.get("source_audit", {})
            if source:
                rows.append({
                    "modality": modality,
                    "seed": seed,
                    "block": "sample",
                    "metric": "source_label",
                    "test": "PERMANOVA",
                    "statistic": float(source["r2"]),
                    "p_value": float(source["p_value"]),
                    "p_gt_0.05": bool(float(source["p_value"]) > 0.05),
                    "auc": float(source["auc"]),
                    "precision": float(source["precision"]),
                    "recall": float(source["recall"]),
                    "authenticity": float(source["authenticity"]),
                })
    tests = pd.DataFrame(rows)
    summary_rows = []
    for (modality, block, metric, test), sub in tests.groupby(["modality", "block", "metric", "test"]):
        stat_med, stat_lo, stat_hi = _bootstrap_ci(sub["statistic"].to_numpy(float), seed=83)
        p_med, p_lo, p_hi = _bootstrap_ci(sub["p_value"].to_numpy(float), seed=89)
        row = {
            "modality": modality,
            "block": block,
            "metric": metric,
            "test": test,
            "statistic_median": stat_med,
            "statistic_ci_low": stat_lo,
            "statistic_ci_high": stat_hi,
            "p_median": p_med,
            "p_ci_low": p_lo,
            "p_ci_high": p_hi,
            "p_gt_0.05_rate": float(sub["p_gt_0.05"].mean()),
        }
        for extra in ["auc", "precision", "recall", "authenticity"]:
            if extra in sub:
                vals = sub[extra].dropna().to_numpy(float)
                if vals.size:
                    row[f"{extra}_median"] = float(np.median(vals))
        summary_rows.append(row)
    return tests, pd.DataFrame(summary_rows).sort_values(["modality", "block", "metric"])


def _write_tables(
    out: Path,
    baseline: pd.DataFrame,
    validation: pd.DataFrame,
    optimization: dict,
    sample_stats: dict,
    distance_tail: dict,
    literature_tests: pd.DataFrame,
    literature_summary: pd.DataFrame,
) -> None:
    baseline.to_csv(out / "real_split_baseline_long.csv", index=False)
    validation.to_csv(out / "synthetic_validation_long.csv", index=False)
    literature_tests.to_csv(out / "literature_standard_tests_long.csv", index=False)
    literature_summary.to_csv(out / "literature_standard_tests_summary.csv", index=False)
    summary_rows = []
    for (modality, metric), sub in validation.groupby(["modality", "metric"]):
        med, lo, hi = _bootstrap_ci(sub["raw_value"].to_numpy(float), seed=23)
        nmed, nlo, nhi = _bootstrap_ci(sub["normalized"].to_numpy(float), seed=23)
        summary_rows.append({
            "modality": modality,
            "metric": metric,
            "raw_median": med,
            "raw_ci_low": lo,
            "raw_ci_high": hi,
            "normalized_median": nmed,
            "normalized_ci_low": nlo,
            "normalized_ci_high": nhi,
            "pass_all_validation_pools": bool(sub["pass"].all()),
        })
    pd.DataFrame(summary_rows).to_csv(out / "synthetic_validation_summary.csv", index=False)
    (out / "fig2_fidelity_audit_summary.json").write_text(
        json.dumps({"optimization": optimization, "sample_level": sample_stats, "distance_low_tail": distance_tail}, indent=2),
        encoding="utf-8",
    )


def _quantile_scatter(ax, real_vals: np.ndarray, syn_vals: np.ndarray, color: str, marker: str, label: str) -> float:
    qs = np.linspace(0.02, 0.98, 80)
    rq = np.quantile(np.asarray(real_vals, dtype=float), qs)
    sq = np.quantile(np.asarray(syn_vals, dtype=float), qs)
    ax.scatter(rq, sq, s=14, color=color, marker=marker, alpha=0.74, linewidths=0, label=label, rasterized=True)
    return float(ks_2samp(real_vals, syn_vals).statistic)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy()
    b = pd.Series(np.asarray(b, dtype=float)).rank(method="average").to_numpy()
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _plot_protein_structure(out: Path, real: dict, syn: dict) -> None:
    rt = real["topology"]
    st = syn["topology"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax = axes[0]
    tax_ks = _quantile_scatter(ax, rt["taxon_degree"], st["taxon_degree"], PROTEIN, "o", "taxon degree")
    fun_ks = _quantile_scatter(ax, rt["function_degree"], st["function_degree"], "#e07b39", "^", "function degree")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#666666", ls="--", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Node-degree quantiles")
    ax.set_xlabel("real quantile")
    ax.set_ylabel("synthetic quantile")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(0.02, 0.98, f"KS taxon/function\n{tax_ks:.3f}/{fun_ks:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.88))

    ax = axes[1]
    n_tax = min(len(rt["taxon_prevalence"]), len(st["taxon_prevalence"]))
    n_fun = min(len(rt["function_prevalence"]), len(st["function_prevalence"]))
    ax.scatter(rt["taxon_prevalence"][:n_tax], st["taxon_prevalence"][:n_tax], s=13, color=PROTEIN, alpha=0.68, linewidths=0, label="taxa", rasterized=True)
    ax.scatter(rt["function_prevalence"][:n_fun], st["function_prevalence"][:n_fun], s=13, color="#e07b39", alpha=0.68, linewidths=0, label="functions", rasterized=True)
    ax.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1.0)
    tax_r = _rank_corr(rt["taxon_prevalence"][:n_tax], st["taxon_prevalence"][:n_tax])
    fun_r = _rank_corr(rt["function_prevalence"][:n_fun], st["function_prevalence"][:n_fun])
    ax.set_title("Node prevalence agreement")
    ax.set_xlabel("real prevalence")
    ax.set_ylabel("synthetic prevalence")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.text(0.02, 0.98, f"rank r taxon/function\n{tax_r:.3f}/{fun_r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.2, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.88))

    ax = axes[2]
    edge_ks = _quantile_scatter(ax, rt["edge_count"], st["edge_count"], PROTEIN, "o", "edge count")
    real_conn = float(rt["connectance"])
    syn_conn = float(st["connectance"])
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#666666", ls="--", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Edge support / connectance")
    ax.set_xlabel("real edge-count quantile")
    ax.set_ylabel("synthetic edge-count quantile")
    ax.text(0.02, 0.98, f"edge KS {edge_ks:.3f}\nconnectance ratio {syn_conn / max(real_conn, 1e-12):.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.88))
    for i, label in enumerate(["a", "b", "c"]):
        axes[i].text(-0.15, 1.08, label, transform=axes[i].transAxes, fontsize=13, fontweight="bold")
        axes[i].grid(color=GRID, lw=0.55, alpha=0.7)
    fig.tight_layout(w_pad=2.1)
    fig.savefig(out / "protein_tf_structure.png", dpi=320, bbox_inches="tight")
    fig.savefig(out / "protein_tf_structure.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_panel_d_structure(fig: plt.Figure, spec, real: dict, syn: dict) -> None:
    rt = real["topology"]
    st = syn["topology"]
    outer = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.34)

    ax = fig.add_subplot(outer[0, 0])
    _letter(ax, "d", "Protein taxon-function structure")
    tax_ks = _quantile_scatter(ax, rt["taxon_degree"], st["taxon_degree"], STRUCT_TAXON, "o", "taxon degree")
    fun_ks = _quantile_scatter(ax, rt["function_degree"], st["function_degree"], STRUCT_FUNCTION, "^", "function degree")
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#666666", ls="--", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("node degree", fontsize=13.0)
    ax.set_xlabel("real quantile")
    ax.set_ylabel("synthetic quantile")
    ax.legend(frameon=False, fontsize=10.3, loc="lower right")
    ax.text(0.03, 0.97, f"KS taxon/function\n{tax_ks:.3f}/{fun_ks:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.2, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.88))
    ax.grid(color=GRID, lw=0.55, alpha=0.7)

    ax = fig.add_subplot(outer[0, 1])
    n_tax = min(len(rt["taxon_prevalence"]), len(st["taxon_prevalence"]))
    n_fun = min(len(rt["function_prevalence"]), len(st["function_prevalence"]))
    ax.scatter(rt["taxon_prevalence"][:n_tax], st["taxon_prevalence"][:n_tax],
               s=12, color=STRUCT_TAXON, alpha=0.68, linewidths=0, label="taxa", rasterized=True)
    ax.scatter(rt["function_prevalence"][:n_fun], st["function_prevalence"][:n_fun],
               s=12, color=STRUCT_FUNCTION, alpha=0.68, linewidths=0, label="functions", rasterized=True)
    ax.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1.0)
    tax_r = _rank_corr(rt["taxon_prevalence"][:n_tax], st["taxon_prevalence"][:n_tax])
    fun_r = _rank_corr(rt["function_prevalence"][:n_fun], st["function_prevalence"][:n_fun])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("node prevalence", fontsize=13.0)
    ax.set_xlabel("real prevalence")
    ax.set_ylabel("synthetic prevalence")
    ax.legend(frameon=False, fontsize=10.3, loc="lower right")
    ax.text(0.03, 0.97, f"rank r taxon/function\n{tax_r:.3f}/{fun_r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10.2, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.88))
    ax.grid(color=GRID, lw=0.55, alpha=0.7)

    ax = fig.add_subplot(outer[0, 2])
    edge_ks = _quantile_scatter(ax, rt["edge_count"], st["edge_count"], STRUCT_EDGE, "o", "edge count")
    real_conn = float(rt["connectance"])
    syn_conn = float(st["connectance"])
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#666666", ls="--", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("edge support", fontsize=13.0)
    ax.set_xlabel("real quantile")
    ax.set_ylabel("synthetic quantile")
    ax.text(0.03, 0.97, f"edge KS {edge_ks:.3f}\nconnectance ratio {syn_conn / max(real_conn, 1e-12):.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#dddddd", alpha=0.88))
    ax.grid(color=GRID, lw=0.55, alpha=0.7)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Statistical fidelity audit figure for synthetic sample pools.")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "output")
    p.add_argument(
        "--tables-out",
        type=Path,
        default=ROOT / "data" / "archived_runs" / "fig2_fidelity_audit",
        help="directory for the validation tables and the standalone protein-structure panel",
    )
    p.add_argument("--pool-M", type=int, default=120)
    p.add_argument("--opt-pool-M", type=int, default=60)
    p.add_argument("--baseline-splits", type=int, default=500)
    p.add_argument("--validation-pools", type=int, default=20)
    p.add_argument("--opt-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260701)
    p.add_argument("--skip-optimization", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.tables_out.mkdir(parents=True, exist_ok=True)

    print("[fig2-audit] loading real objects", flush=True)
    protein_d, protein_real = _real_objects("protein")
    gene_d, gene_real = _real_objects("gene")
    real = {"protein": protein_real, "gene": gene_real}

    print("[fig2-audit] real-vs-real split baselines", flush=True)
    baseline = pd.concat([
        _real_split_baseline(protein_d, protein_real, "protein", args.baseline_splits, args.seed + 1),
        _real_split_baseline(None, gene_real, "gene", args.baseline_splits, args.seed + 2),
    ], ignore_index=True)
    allowed = _allowed_thresholds(baseline)

    opt_seed_list = [args.seed + 100 + i * 17 for i in range(args.opt_seeds)]
    if args.skip_optimization:
        optimization = {
            "protein": {"params": {"edge_fraction": 1.0, "residual_mode": "template", "marginal_strength": "auto", "eb_k": "auto", "jitter_sd": 0.3}},
            "gene": {"params": {"pi": 1.0, "scale": 1.0}},
        }
    else:
        optimization = {
            "protein": _select_params("protein", protein_d, protein_real, allowed, pool_m=args.opt_pool_M, seeds=opt_seed_list),
            "gene": _select_params("gene", gene_d, gene_real, allowed, pool_m=args.opt_pool_M, seeds=opt_seed_list),
        }

    val_seed_list = [args.seed + 1000 + i * 31 for i in range(args.validation_pools)]
    protein_val, _, protein_bundles = _validate_pools(
        "protein",
        protein_d,
        protein_real,
        optimization["protein"]["params"],
        allowed,
        pool_m=args.pool_M,
        seeds=val_seed_list,
    )
    gene_val, _, gene_bundles = _validate_pools(
        "gene",
        gene_d,
        gene_real,
        optimization["gene"]["params"],
        allowed,
        pool_m=args.pool_M,
        seeds=val_seed_list,
    )
    validation = pd.concat([protein_val, gene_val], ignore_index=True)
    protein_rep = _representative_bundle(validation, protein_bundles, "protein")
    gene_rep = _representative_bundle(validation, gene_bundles, "gene")

    print("[fig2-audit] coembedding representative pools", flush=True)
    embeddings = {
        "protein": _coembed(protein_d, protein_real, protein_rep, args.seed + 3000),
        "gene": _coembed(gene_d, gene_real, gene_rep, args.seed + 4000),
    }
    sample_stats = {}
    for modality, emb in embeddings.items():
        coords = emb["coords"]
        source = emb["source"].to_numpy()
        support = _support_metrics(coords, source, k=5)
        source_r2, source_p = _permanova_r2_p(emb["dm"], emb["source"], permutations=199)
        sample_stats[modality] = {
            "source_r2": source_r2,
            "source_p": source_p,
            "auc": _knn_auc(coords[:, : min(6, coords.shape[1])], source, args.seed),
            **support,
        }

    print("[fig2-audit] plotting", flush=True)
    fig = plt.figure(figsize=(16.2, 10.2))
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.92, 1.18], hspace=0.45)
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], width_ratios=[1.16, 1.08], wspace=0.32)
    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], width_ratios=[0.62, 1.62], wspace=0.20)
    _plot_panel_a_feature(fig, top[0, 0], real, {"protein": protein_rep, "gene": gene_rep})
    ax_b = fig.add_subplot(top[0, 1])
    _plot_panel_b(ax_b, validation)
    _plot_panel_c_validation(
        fig,
        bottom[0, 0],
        real,
        {"protein": protein_bundles, "gene": gene_bundles},
        validation,
    )
    _plot_panel_d_structure(fig, bottom[0, 1], protein_real, protein_rep)
    fig.savefig(args.out / "fig3_pool_fidelity.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "fig3_pool_fidelity.pdf", bbox_inches="tight")
    plt.close(fig)

    distance_tail = _distance_tail_diagnostics(real, {"protein": protein_bundles, "gene": gene_bundles})
    literature_tests, literature_summary = _literature_standard_tests(real, {"protein": protein_bundles, "gene": gene_bundles})
    _write_tables(args.tables_out, baseline, validation, optimization, sample_stats, distance_tail, literature_tests, literature_summary)
    _plot_protein_structure(args.tables_out, protein_real, protein_rep)
    print(f"[fig3-audit] done -> {args.out / 'fig3_pool_fidelity.png'}", flush=True)


if __name__ == "__main__":
    main()
