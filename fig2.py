#!/usr/bin/env python3
"""Figure 2 -- fidelity validation of synthetic sample pools.

Layout:
  a) condensed scorecard
  b) feature-level abundance fidelity
  c) modality-specific structure
  d) global geometry after recomputing distances
  e) downstream distance residuals
  f) within-group eigenspectrum and d_eff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import figstyle  # noqa: E402
import pcam_gen as P  # noqa: E402
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
from _protein_mdctf_optimized_curve import _rank_corr  # noqa: E402
from phylopower import core  # noqa: E402
from semisynthetic_power import _pcoa_coords  # noqa: E402

core.load_core_runtime()
figstyle.apply_style()
FIGDATA = ROOT / "figdata"

DARK = "#303642"
REAL = "#2f3742"
PROTEIN = "#159c76"
GENE = "#756bb1"
GROUP_COLORS = ["#2f6db0", "#b03a3a"]
GRID = "#e7e7e7"

plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
})


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(abs(den), 1e-12))


def _ratio_error(value: float) -> float:
    return abs(float(np.log2(max(value, 1e-12))))


def _within_between(dm: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ids = list(dm.index)
    arr = dm.loc[ids, ids].to_numpy(dtype=float)
    lab = labels.loc[ids].to_numpy()
    iu = np.triu_indices(arr.shape[0], 1)
    vals = arr[iu]
    same = lab[iu[0]] == lab[iu[1]]
    return vals[same], vals[~same]


def _ecdf_delta_grid(real: np.ndarray, syn: np.ndarray, x: np.ndarray) -> np.ndarray:
    real = np.sort(np.asarray(real, dtype=float))
    syn = np.sort(np.asarray(syn, dtype=float))
    ry = np.searchsorted(real, x, side="right") / max(real.size, 1)
    sy = np.searchsorted(syn, x, side="right") / max(syn.size, 1)
    return sy - ry


def _axis(ax, *, zero: bool = False, xgrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(length=3, width=0.8, color="#555555")
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.55, alpha=0.75)
    if zero:
        ax.axhline(0, color="#606060", lw=0.9, ls=(0, (1.2, 1.8)), zorder=1)


def _letter(ax, letter: str, title: str) -> None:
    ax.text(-0.08, 1.07, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, fontweight="bold")
    ax.text(0.0, 1.08, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.2, fontweight="bold")


def _note(ax, text: str, loc: str = "upper left") -> None:
    x, y, ha, va = {
        "upper left": (0.02, 0.96, "left", "top"),
        "upper right": (0.98, 0.96, "right", "top"),
        "lower left": (0.02, 0.05, "left", "bottom"),
        "lower right": (0.98, 0.05, "right", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="0.88", alpha=0.78))


def _feature_stats(tab: pd.DataFrame) -> dict[str, np.ndarray | float]:
    arr = tab.to_numpy(dtype=float)
    return {
        "mean": arr.mean(axis=1),
        "variance": arr.var(axis=1),
        "prevalence": (arr > 0).mean(axis=1),
        "library_size": arr.sum(axis=0),
        "sample_sparsity": (arr == 0).mean(axis=0),
        "overall_sparsity": float((arr == 0).mean()),
    }


def _protein_topology(d: dict, syn_tab: pd.DataFrame, syn_sgm: pd.Series) -> tuple[dict, dict]:
    taxa = d["uid"].astype(int)
    funcs = d["meta"]["Function"].astype(str).to_numpy()
    _, func_ids = np.unique(funcs, return_inverse=True)
    n_taxa = int(taxa.max()) + 1
    n_funcs = int(func_ids.max()) + 1
    real_cols = np.concatenate([d["gs"][g] for g in d["groups"]])
    real = d["abund"][:, real_cols] > 0
    syn = syn_tab.loc[:, list(syn_sgm.index)].to_numpy() > 0

    def summarize(mask: np.ndarray) -> dict[str, np.ndarray | float]:
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

    return summarize(real), summarize(syn)


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


def _calibrate_protein_pool(d: dict, real_dm: pd.DataFrame, real_sgm: pd.Series, M: int, seed: int,
                            edge_fraction: str, marginal_strength: str,
                            eb_k: str, residual_mode: str, edge_min: float,
                            edge_max: float, edge_candidates: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, float]:
    real_tab, _ = P.real_table(d)
    if edge_fraction not in {"auto", "auto-fidelity"}:
        ef = float(edge_fraction)
        tab, sgm = mdctf_mc_pool(
            d,
            M,
            seed,
            1.0,
            edge_fraction=ef,
            marginal_strength=marginal_strength,
            eb_k=eb_k,
            residual_mode=residual_mode,
        )
        return tab, sgm, P.recompute_distance(d, tab), ef
    target = max(0.0, float(core.compute_omega2(real_dm, real_sgm)))
    if edge_fraction == "auto-fidelity":
        grid = np.linspace(float(edge_min), float(edge_max), int(edge_candidates))
    else:
        grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1]
    best = (float("inf"), grid[0], None, None, None)
    for ef in grid:
        tab, sgm = mdctf_mc_pool(
            d,
            M,
            seed,
            1.0,
            edge_fraction=ef,
            marginal_strength=marginal_strength,
            eb_k=eb_k,
            residual_mode=residual_mode,
        )
        dm = P.recompute_distance(d, tab)
        om = max(0.0, float(core.compute_omega2(dm, sgm)))
        if edge_fraction == "auto-fidelity":
            real_stats, syn_stats = _feature_stats(real_tab), _feature_stats(tab)
            real_top, syn_top = _protein_topology(d, tab, sgm)
            dist = _distance_metrics(real_dm, real_sgm, dm, sgm)
            topo = max(
                float(ks_2samp(real_top["taxon_degree"], syn_top["taxon_degree"]).statistic),
                float(ks_2samp(real_top["function_degree"], syn_top["function_degree"]).statistic),
                float(ks_2samp(real_top["edge_count"], syn_top["edge_count"]).statistic),
                abs(_safe_ratio(float(syn_top["connectance"]), float(real_top["connectance"])) - 1.0),
            )
            loss = max(
                float(ks_2samp(real_stats["mean"], syn_stats["mean"]).statistic),
                float(ks_2samp(real_stats["variance"], syn_stats["variance"]).statistic),
                float(ks_2samp(real_stats["prevalence"], syn_stats["prevalence"]).statistic),
                topo,
                float(dist["within_ks"]),
                float(dist["between_ks"]),
                _ratio_error(float(dist["omega2_ratio"])),
                _ratio_error(float(dist["deff_ratio"])),
            )
        else:
            loss = abs(om - target)
        print(f"[fig2] protein edge_fraction={float(ef):.3f} loss={loss:.3f} omega={om:.3f}", flush=True)
        if loss < best[0]:
            best = (loss, ef, tab, sgm, dm)
    print(f"[fig2] protein selected edge_fraction={best[1]:.3f} ({edge_fraction})", flush=True)
    return best[2], best[3], best[4], float(best[1])


def _make_protein(M: int, seed: int, edge_fraction: str, marginal_strength: str, eb_k: str,
                  residual_mode: str, edge_min: float, edge_max: float, edge_candidates: int) -> dict:
    d = P.load_modality("protein")
    real_tab, real_sgm = P.real_table(d)
    real_dm = P.recompute_distance(d, real_tab)
    syn_tab, syn_sgm, syn_dm, ef = _calibrate_protein_pool(
        d, real_dm, real_sgm, M, seed, edge_fraction, marginal_strength, eb_k,
        residual_mode, edge_min, edge_max, edge_candidates
    )
    return {
        "d": d,
        "real_tab": real_tab,
        "real_sgm": real_sgm,
        "real_dm": real_dm,
        "syn_tab": syn_tab,
        "syn_sgm": syn_sgm,
        "syn_dm": syn_dm,
        "stats": (_feature_stats(real_tab), _feature_stats(syn_tab)),
        "topology": _protein_topology(d, syn_tab, syn_sgm),
        "distance": _distance_metrics(real_dm, real_sgm, syn_dm, syn_sgm),
        "generator": {"method": "MDC-TF-MC", "edge_fraction": ef,
                      "marginal_strength": marginal_strength, "eb_k": eb_k,
                      "residual_mode": residual_mode},
    }


def _make_gene(M: int, seed: int) -> dict:
    d = P.load_modality("gene")
    real_tab, real_sgm = P.real_table(d)
    real_dm = P.recompute_distance(d, real_tab)
    om_real = max(0.0, float(core.compute_omega2(real_dm, real_sgm)))
    cands = [(pi, 1.0) for pi in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]]
    cands += [(1.0, s) for s in [1.2, 1.4, 1.7, 2.0, 2.5, 3.0]]
    best = (float("inf"), 1.0, 1.0, 0.0)
    for pi0, scale0 in cands:
        tab0, sgm0 = P.pcam_pool(d, M, seed, pi0, scale0, ndon=1)
        dm0 = P.recompute_distance(d, tab0)
        om0 = max(0.0, float(core.compute_omega2(dm0, sgm0)))
        if abs(om0 - om_real) < best[0]:
            best = (abs(om0 - om_real), pi0, scale0, om0)
    _, pi, scale, achieved = best
    print(f"[fig2] gene calibrate omega2 {om_real:.3f}: pi={pi:.2f}, scale={scale:.2f}, preview={achieved:.3f}", flush=True)
    syn_tab, syn_sgm = P.pcam_pool(d, M, seed, pi, scale, ndon=1)
    syn_dm = P.recompute_distance(d, syn_tab)
    return {
        "d": d,
        "real_tab": real_tab,
        "real_sgm": real_sgm,
        "real_dm": real_dm,
        "syn_tab": syn_tab,
        "syn_sgm": syn_sgm,
        "syn_dm": syn_dm,
        "stats": (_feature_stats(real_tab), _feature_stats(syn_tab)),
        "distance": _distance_metrics(real_dm, real_sgm, syn_dm, syn_sgm),
        "generator": {"pi": pi, "scale": scale},
    }


def _sample_syn_columns(sgm: pd.Series, n_per_group: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    cols: list[str] = []
    for g in pd.unique(sgm):
        members = sgm[sgm == g].index.to_numpy()
        pick = rng.choice(members, size=min(n_per_group, len(members)), replace=False)
        cols.extend(pick.tolist())
    return cols


def _coembed(data: dict, seed: int, max_syn_per_group: int = 70) -> dict:
    syn_cols = _sample_syn_columns(data["syn_sgm"], max_syn_per_group, seed)
    syn = data["syn_tab"].loc[:, syn_cols].copy()
    rename = {c: f"syn_{i:04d}" for i, c in enumerate(syn.columns)}
    syn = syn.rename(columns=rename)
    syn_sgm = data["syn_sgm"].loc[syn_cols].rename(index=rename)
    real = data["real_tab"]
    comb = pd.concat([real, syn], axis=1)
    dm = P.recompute_distance(data["d"], comb)
    order = list(comb.columns)
    coords = _pcoa_coords(dm.loc[order, order]).to_numpy()
    labels = pd.concat([data["real_sgm"], syn_sgm]).loc[order]
    kind = np.array(["real"] * real.shape[1] + ["synthetic"] * syn.shape[1])
    return {"coords": coords, "labels": labels.to_numpy(), "kind": kind}


def _mean_var_inset(ax, stats: tuple[dict, dict], title: str, color: str, seed: int) -> dict:
    real, syn = stats
    rng = np.random.default_rng(seed)
    out = {}
    for source, st, c, alpha in [("real", real, REAL, 0.28), ("synthetic", syn, color, 0.24)]:
        mu = np.asarray(st["mean"], dtype=float)
        var = np.asarray(st["variance"], dtype=float)
        pos = np.concatenate([mu[mu > 0], var[var > 0]])
        eps = float(np.quantile(pos, 0.01) * 0.5) if len(pos) else 1e-9
        idx = np.arange(mu.size)
        if idx.size > 2600:
            idx = rng.choice(idx, size=2600, replace=False)
        ax.scatter(mu[idx] + eps, var[idx] + eps, s=6, color=c, alpha=alpha, linewidths=0, rasterized=True, label=source)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=8.8, pad=2)
    ax.set_xlabel("mean", fontsize=7.8)
    ax.set_ylabel("variance", fontsize=7.8)
    ax.tick_params(labelsize=7)
    ax.grid(color=GRID, lw=0.5, alpha=0.65)
    out["mean_ks"] = float(ks_2samp(real["mean"], syn["mean"]).statistic)
    out["variance_ks"] = float(ks_2samp(real["variance"], syn["variance"]).statistic)
    _note(ax, f"KS {out['mean_ks']:.2f}/{out['variance_ks']:.2f}")
    return out


def _panel_scorecard(ax, protein: dict, gene: dict) -> dict:
    _letter(ax, "a", "Fidelity scorecard")
    p_stats, g_stats = protein["stats"], gene["stats"]
    p_topo = protein["topology"]
    p_dist, g_dist = protein["distance"], gene["distance"]
    p_real, p_syn = p_stats
    g_real, g_syn = g_stats
    p_topo_real, p_topo_syn = p_topo
    p_abund = max(float(ks_2samp(p_real["mean"], p_syn["mean"]).statistic),
                  float(ks_2samp(p_real["variance"], p_syn["variance"]).statistic))
    g_abund = max(float(ks_2samp(g_real["mean"], g_syn["mean"]).statistic),
                  float(ks_2samp(g_real["variance"], g_syn["variance"]).statistic))
    p_topo_err = max(
        float(ks_2samp(p_topo_real["taxon_degree"], p_topo_syn["taxon_degree"]).statistic),
        float(ks_2samp(p_topo_real["function_degree"], p_topo_syn["function_degree"]).statistic),
        float(ks_2samp(p_topo_real["edge_count"], p_topo_syn["edge_count"]).statistic),
    )
    rows = [
        ("Protein abundance", p_abund, PROTEIN, "o"),
        ("Protein prevalence", float(ks_2samp(p_real["prevalence"], p_syn["prevalence"]).statistic), PROTEIN, "s"),
        ("TF topology", p_topo_err, PROTEIN, "D"),
        ("Protein distance", max(p_dist["within_ks"], p_dist["between_ks"]), PROTEIN, "v"),
        ("Protein d_eff", _ratio_error(p_dist["deff_ratio"]), PROTEIN, "^"),
        ("Protein omega2", _ratio_error(p_dist["omega2_ratio"]), PROTEIN, "^"),
        ("Gene abundance", g_abund, GENE, "o"),
        ("Gene prevalence/lib", max(float(ks_2samp(g_real["prevalence"], g_syn["prevalence"]).statistic),
                                    float(ks_2samp(g_real["library_size"], g_syn["library_size"]).statistic),
                                    float(ks_2samp(g_real["sample_sparsity"], g_syn["sample_sparsity"]).statistic)), GENE, "s"),
        ("Gene distance", max(g_dist["within_ks"], g_dist["between_ks"]), GENE, "v"),
        ("Gene d_eff", _ratio_error(g_dist["deff_ratio"]), GENE, "^"),
        ("Gene omega2", _ratio_error(g_dist["omega2_ratio"]), GENE, "^"),
    ]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0.0, 0.05, color="#e6f1e3", zorder=0)
    ax.axvspan(0.05, 0.15, color="#f6edcf", zorder=0)
    for yi, (_, err, color, marker) in zip(y, rows):
        ax.hlines(yi, 0, err, color=color, lw=1.6, alpha=0.72)
        ax.scatter(err, yi, s=38, color=color, marker=marker, edgecolors="white", linewidths=0.7, zorder=3)
        ax.text(err + 0.008, yi, f"{err:.2f}", va="center", ha="left", fontsize=6.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.6)
    ax.set_xlabel("deviation from ideal")
    ax.set_xlim(-0.005, max(0.35, max(r[1] for r in rows) * 1.16))
    ax.set_ylim(-0.7, len(rows) - 0.3)
    _axis(ax, xgrid=True)
    return {"protein": {r[0]: r[1] for r in rows if r[0].startswith(("Protein", "TF"))},
            "gene": {r[0]: r[1] for r in rows if r[0].startswith("Gene")}}


def _panel_feature(ax, protein: dict, gene: dict) -> dict:
    ax.axis("off")
    _letter(ax, "b", "Feature-level abundance fidelity")
    left = ax.inset_axes([0.06, 0.13, 0.41, 0.78])
    right = ax.inset_axes([0.56, 0.13, 0.41, 0.78])
    metrics = {
        "protein": _mean_var_inset(left, protein["stats"], "Protein", PROTEIN, 1),
        "gene": _mean_var_inset(right, gene["stats"], "Gene", GENE, 2),
    }
    left.legend(frameon=False, fontsize=6.5, loc="lower right")
    return metrics


def _panel_structure(ax, protein: dict, gene: dict) -> dict:
    ax.axis("off")
    _letter(ax, "c", "Modality-specific structure")
    left = ax.inset_axes([0.06, 0.15, 0.41, 0.76])
    right = ax.inset_axes([0.56, 0.15, 0.41, 0.76])
    real_topo, syn_topo = protein["topology"]
    metrics = {"protein": {}, "gene": {}}
    for key, label, ls in [("taxon_degree", "taxon", "-"), ("function_degree", "function", "--")]:
        lo = max(1.0, min(np.min(real_topo[key]), np.min(syn_topo[key])))
        hi = max(np.max(real_topo[key]), np.max(syn_topo[key]))
        x = np.geomspace(lo, hi, 240)
        left.plot(x, _ecdf_delta_grid(real_topo[key], syn_topo[key], x), color=PROTEIN, lw=2.0, ls=ls, label=label)
        metrics["protein"][f"{key}_ks"] = float(ks_2samp(real_topo[key], syn_topo[key]).statistic)
    metrics["protein"]["edge_count_ks"] = float(ks_2samp(real_topo["edge_count"], syn_topo["edge_count"]).statistic)
    metrics["protein"]["connectance_ratio"] = _safe_ratio(float(syn_topo["connectance"]), float(real_topo["connectance"]))
    left.set_xscale("log")
    left.set_xlabel("TF node degree", fontsize=8)
    left.set_ylabel("ECDF residual", fontsize=8)
    left.set_ylim(-0.045, 0.045)
    left.legend(frameon=False, loc="lower right")
    _axis(left, zero=True)
    _note(left, f"degree KS {metrics['protein']['taxon_degree_ks']:.3f}/{metrics['protein']['function_degree_ks']:.3f}\nconnectance {metrics['protein']['connectance_ratio']:.3f}")

    greal, gsyn = gene["stats"]
    for key, label, ls in [("library_size", "library", "-"), ("sample_sparsity", "sparsity", "--")]:
        lo = min(np.min(greal[key]), np.min(gsyn[key]))
        hi = max(np.max(greal[key]), np.max(gsyn[key]))
        x = np.linspace(lo, hi, 220)
        x_scaled = (x - lo) / max(hi - lo, 1e-12)
        right.plot(x_scaled, _ecdf_delta_grid(greal[key], gsyn[key], x), color=GENE, lw=2.0, ls=ls, label=label)
        metrics["gene"][f"{key}_ks"] = float(ks_2samp(greal[key], gsyn[key]).statistic)
    right.set_xlabel("scaled sample QC variable", fontsize=8)
    right.set_ylabel("", fontsize=8)
    right.set_ylim(-0.12, 0.12)
    right.legend(frameon=False, loc="lower right")
    _axis(right, zero=True)
    _note(right, f"library KS {metrics['gene']['library_size_ks']:.2f}\nsparsity KS {metrics['gene']['sample_sparsity_ks']:.2f}")
    return metrics


def _geometry_inset(ax, emb: dict, title: str) -> None:
    coords = emb["coords"]
    labels = emb["labels"]
    kind = emb["kind"]
    groups = list(pd.unique(labels))
    for gi, g in enumerate(groups):
        syn = (labels == g) & (kind == "synthetic")
        real = (labels == g) & (kind == "real")
        ax.scatter(coords[syn, 0], coords[syn, 1], s=8, color=GROUP_COLORS[gi % 2], alpha=0.17, linewidths=0)
        ax.scatter(coords[real, 0], coords[real, 1], s=25, color=GROUP_COLORS[gi % 2],
                   edgecolors="white", linewidths=0.6, zorder=4)
        for mask, ls, alpha in [(real, "-", 0.95), (syn, "--", 0.70)]:
            sub = coords[mask]
            if len(sub) >= 4:
                cov = np.cov(sub[:, :2], rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                vals = np.maximum(vals, 0)
                t = np.linspace(0, 2 * np.pi, 160)
                ell = (vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T + sub[:, :2].mean(axis=0)
                ax.plot(ell[:, 0], ell[:, 1], color=GROUP_COLORS[gi % 2], lw=1.0, ls=ls, alpha=alpha)
    ax.set_title(title, fontsize=8.8, pad=2)
    ax.set_xlabel("PCoA 1", fontsize=8)
    ax.set_ylabel("PCoA 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(color=GRID, lw=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_geometry(ax, protein: dict, gene: dict, seed: int) -> dict:
    ax.axis("off")
    _letter(ax, "d", "Global geometry after distance recomputation")
    left = ax.inset_axes([0.06, 0.14, 0.41, 0.78])
    right = ax.inset_axes([0.56, 0.14, 0.41, 0.78])
    p_emb = _coembed(protein, seed, max_syn_per_group=65)
    g_emb = _coembed(gene, seed + 11, max_syn_per_group=65)
    _geometry_inset(left, p_emb, "Protein")
    _geometry_inset(right, g_emb, "Gene")
    return {"protein": {}, "gene": {}}


def _distance_inset(ax, dist: dict, title: str, color: str, *, ylabel: bool) -> None:
    for rkey, skey, label, ls in [
        ("within_real", "within_syn", "within", "-"),
        ("between_real", "between_syn", "between", "--"),
    ]:
        lo = min(np.min(dist[rkey]), np.min(dist[skey]))
        hi = max(np.max(dist[rkey]), np.max(dist[skey]))
        x = np.linspace(lo, hi, 250)
        ax.plot(x, _ecdf_delta_grid(dist[rkey], dist[skey], x), color=color, lw=2.0, ls=ls, label=label)
    ax.set_title(title, fontsize=8.8, pad=2)
    ax.set_xlabel("distance", fontsize=8)
    ax.set_ylabel("ECDF residual" if ylabel else "", fontsize=8)
    lim = 1.08 * max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), 0.1)
    ax.set_ylim(-lim, lim)
    _axis(ax, zero=True)
    _note(ax, f"KS w/b {dist['within_ks']:.2f}/{dist['between_ks']:.2f}\nomega2 {dist['omega2_real']:.2f}/{dist['omega2_syn']:.2f}")


def _panel_distance(ax, protein: dict, gene: dict) -> dict:
    ax.axis("off")
    _letter(ax, "e", "Downstream distance fidelity")
    left = ax.inset_axes([0.06, 0.14, 0.41, 0.78])
    right = ax.inset_axes([0.56, 0.14, 0.41, 0.78])
    _distance_inset(left, protein["distance"], "Protein", PROTEIN, ylabel=True)
    _distance_inset(right, gene["distance"], "Gene", GENE, ylabel=False)
    right.legend(frameon=False, loc="lower right")
    return {
        "protein": {k: v for k, v in protein["distance"].items() if isinstance(v, (float, int, np.floating, np.integer))},
        "gene": {k: v for k, v in gene["distance"].items() if isinstance(v, (float, int, np.floating, np.integer))},
    }


def _spectrum_inset(ax, dist: dict, title: str, color: str, *, ylabel: bool) -> None:
    for key, c, ls, label in [
        ("real_spectrum", REAL, "-", "real"),
        ("syn_spectrum", color, "--", "synthetic"),
    ]:
        prop = np.asarray(dist[key]["prop"], dtype=float)
        r = np.arange(1, min(len(prop), 18) + 1)
        ax.plot(r, prop[:len(r)], color=c, lw=2.0, ls=ls, marker="o", ms=2.5, label=label)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=8.8, pad=2)
    ax.set_xlabel("eigenvalue rank", fontsize=8)
    ax.set_ylabel("variance fraction" if ylabel else "", fontsize=8)
    ax.grid(color=GRID, lw=0.5, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _note(ax, f"d_eff {dist['deff_real']:.1f}/{dist['deff_syn']:.1f}\nomega2 {dist['omega2_real']:.2f}/{dist['omega2_syn']:.2f}", "upper right")


def _panel_spectrum(ax, protein: dict, gene: dict) -> dict:
    ax.axis("off")
    _letter(ax, "f", "Power-relevant eigenspectrum")
    left = ax.inset_axes([0.06, 0.14, 0.41, 0.78])
    right = ax.inset_axes([0.56, 0.14, 0.41, 0.78])
    _spectrum_inset(left, protein["distance"], "Protein", PROTEIN, ylabel=True)
    _spectrum_inset(right, gene["distance"], "Gene", GENE, ylabel=False)
    right.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    return {
        "protein": {"deff_real": protein["distance"]["deff_real"], "deff_syn": protein["distance"]["deff_syn"]},
        "gene": {"deff_real": gene["distance"]["deff_real"], "deff_syn": gene["distance"]["deff_syn"]},
    }


def _save_distance_table(distances: dict[str, dict]) -> None:
    rows = []
    for modality, dist in distances.items():
        for kind in ["within_real", "within_syn", "between_real", "between_syn"]:
            rows.append(pd.DataFrame({"modality": modality, "kind": kind, "distance": dist[kind]}))
    pd.concat(rows, ignore_index=True).to_csv(FIGDATA / "fig2_fidelity_distances.csv", index=False)


def _save_metric_table(summary: dict[str, dict]) -> None:
    rows = []
    for block, payload in summary.items():
        for modality, metrics in payload.items():
            for key, value in metrics.items():
                if isinstance(value, (float, int, np.floating, np.integer)):
                    rows.append({"block": block, "modality": modality, "metric": key, "value": float(value)})
    pd.DataFrame(rows).to_csv(FIGDATA / "fig2_fidelity_metrics.csv", index=False)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Figure 2: synthetic-pool fidelity validation.")
    p.add_argument("--pool-M", type=int, default=220)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--edge-fraction", default="1.0",
                   help="MDC-TF-MC edge residual fraction, 'auto', or 'auto-fidelity'.")
    p.add_argument("--edge-min", type=float, default=0.5)
    p.add_argument("--edge-max", type=float, default=2.0)
    p.add_argument("--edge-candidates", type=int, default=7)
    p.add_argument("--protein-marginal-strength", default="auto")
    p.add_argument("--protein-eb-k", default="auto")
    p.add_argument("--protein-residual-mode", choices=["random", "template"], default="random")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    FIGDATA.mkdir(parents=True, exist_ok=True)

    print("[fig2] protein MDC-TF-MC fidelity...", flush=True)
    protein = _make_protein(
        args.pool_M,
        args.seed,
        args.edge_fraction,
        args.protein_marginal_strength,
        args.protein_eb_k,
        args.protein_residual_mode,
        args.edge_min,
        args.edge_max,
        args.edge_candidates,
    )
    print("[fig2] gene PCAM fidelity...", flush=True)
    gene = _make_gene(args.pool_M, args.seed + 7)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.8))
    summary: dict[str, dict] = {
        "scorecard": _panel_scorecard(axes[0, 0], protein, gene),
        "feature": _panel_feature(axes[0, 1], protein, gene),
        "structure": _panel_structure(axes[0, 2], protein, gene),
        "geometry": _panel_geometry(axes[1, 0], protein, gene, args.seed + 31),
        "distance": _panel_distance(axes[1, 1], protein, gene),
        "spectrum": _panel_spectrum(axes[1, 2], protein, gene),
        "generator": {"protein": protein["generator"], "gene": gene["generator"]},
    }

    fig.tight_layout(rect=(0, 0, 1, 1), h_pad=2.1, w_pad=2.0)
    out_png = args.out / "fig2.png"
    out_pdf = args.out / "fig2.pdf"
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    _save_distance_table({"protein": protein["distance"], "gene": gene["distance"]})
    _save_metric_table(summary)
    (args.out / "fig2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[fig2] protein omega2 {protein['distance']['omega2_real']:.3f}/"
        f"{protein['distance']['omega2_syn']:.3f}; gene omega2 "
        f"{gene['distance']['omega2_real']:.3f}/{gene['distance']['omega2_syn']:.3f}",
        flush=True,
    )
    print(
        f"[fig2] protein d_eff {protein['distance']['deff_real']:.2f}/"
        f"{protein['distance']['deff_syn']:.2f}; gene d_eff "
        f"{gene['distance']['deff_real']:.2f}/{gene['distance']['deff_syn']:.2f}",
        flush=True,
    )
    print(f"[fig2] done -> {out_png}", flush=True)
    print(f"[fig2] pdf -> {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
