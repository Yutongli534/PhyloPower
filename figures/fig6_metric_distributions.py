#!/usr/bin/env python3
"""Manuscript Figure 6: metric separation under original, increased, and decreased effects.

A 3 x 2 panel figure: rows are effect settings (original / enhanced / diluted),
columns are metagenomics and metaproteomics, and each panel contrasts
within-group vs between-group pairwise distances across four metrics.

Synthetic panels use the current production generators:
  gene    -> PCAM, donor-coherent mode (ndon=1)
  protein -> MDC-TF-MC, template-residual + auto-fidelity settings

This script rebuilds the synthetic panels (heavy; the gene side needs the
QIIME 2 / Gemelli environment):

  /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig6_metric_distributions.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from phylopower import core  # noqa: E402  (import first: installs the embedded-module finder)
import figstyle  # noqa: E402
import pcam_gen as P  # noqa: E402
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402

core.load_core_runtime()
figstyle.apply_style()
font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial.ttf")
font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 16,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    }
)

C_WITHIN = figstyle.GROUP[0]
C_BETWEEN = figstyle.GROUP[1]

SCENARIOS = [
    ("Original", "original"),
    ("Effect enhanced", "high"),
    ("Effect diluted", "low"),
]


def _bray(sample_by_feature: pd.DataFrame) -> pd.DataFrame:
    mat = squareform(pdist(sample_by_feature.to_numpy(float), metric="braycurtis"))
    return pd.DataFrame(mat, index=sample_by_feature.index, columns=sample_by_feature.index)


def _jaccard(sample_by_feature: pd.DataFrame) -> pd.DataFrame:
    mat = squareform(pdist((sample_by_feature.to_numpy(float) > 0).astype(int), metric="jaccard"))
    return pd.DataFrame(mat, index=sample_by_feature.index, columns=sample_by_feature.index)


def _weighted_unifrac(sample_by_taxon: pd.DataFrame, tree_path: str) -> pd.DataFrame:
    from skbio import TreeNode
    from skbio.diversity import beta_diversity

    table = sample_by_taxon.copy()
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    table[table < 0] = 0.0

    tree = TreeNode.read(str(tree_path)).root_at_midpoint()
    tips = {tip.name for tip in tree.tips() if tip.name is not None}
    taxa = [c for c in table.columns if c in tips and table[c].sum() > 0]
    table = table.loc[table[taxa].sum(axis=1) > 0, taxa]
    dm = beta_diversity(
        "weighted_unifrac",
        np.ascontiguousarray(table.to_numpy(float)),
        ids=table.index.tolist(),
        taxa=taxa,
        tree=tree,
        normalized=True,
    )
    return dm.to_data_frame()


def _candidate_omega(d: dict, table: pd.DataFrame, group_map: pd.Series) -> float:
    dm = P.recompute_distance(d, table)
    return max(0.0, float(core.compute_omega2(dm, group_map)))


def _low_effect_table(d: dict, modality: str, M: int, seed: int, target_omega: float = 0.04):
    best = None
    if modality == "gene":
        candidates = [(pi, scale) for pi in [0.50, 0.52, 0.55, 0.60, 0.65] for scale in [0.05, 0.10, 0.20, 0.35, 0.50]]
        for pi, scale in candidates:
            table, group_map = P.pcam_pool(d, M=M, seed=seed + int(pi * 1000 + scale * 100), pi=pi, scale=scale, ndon=1)
            omega = _candidate_omega(d, table, group_map)
            if omega <= 1e-5:
                continue
            score = abs(target_omega - omega)
            if best is None or score < best[0]:
                best = (score, omega, table, group_map, f"pi={pi}, scale={scale}")
    else:
        for strength in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]:
            table, group_map = mdctf_mc_pool(
                d,
                M=M,
                seed=seed + int(strength * 1000),
                effect_strength=strength,
                edge_fraction="auto",
                marginal_strength="auto",
                eb_k="auto",
                residual_mode="template",
            )
            omega = _candidate_omega(d, table, group_map)
            if omega <= 1e-5:
                continue
            score = abs(target_omega - omega)
            if best is None or score < best[0]:
                best = (score, omega, table, group_map, f"strength={strength}")
    if best is None:
        print(f"[fig6] warning: no positive low-effect candidate for {modality}; using pooled null", flush=True)
        if modality == "gene":
            return P.pcam_null_pool(d, M=M, seed=seed, ndon=1)
        return mdctf_mc_pool(
            d,
            M=M,
            seed=seed,
            effect_strength=0.0,
            edge_fraction="auto",
            marginal_strength="auto",
            eb_k="auto",
            residual_mode="template",
        )
    print(f"[fig6] selected low {modality}: {best[4]} omega={best[1]:.3f}", flush=True)
    return best[2], best[3]


def _high_effect_table(d: dict, modality: str, M: int, seed: int, target_omega: float = 0.80):
    best = None
    if modality == "gene":
        candidates = [(1.0, s) for s in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]]
        for pi, scale in candidates:
            table, group_map = P.pcam_pool(d, M=M, seed=seed + int(scale * 100), pi=pi, scale=scale, ndon=1)
            omega = _candidate_omega(d, table, group_map)
            score = abs(target_omega - omega)
            if best is None or score < best[0]:
                best = (score, omega, table, group_map, f"pi={pi}, scale={scale}")
    else:
        for strength in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
            table, group_map = mdctf_mc_pool(
                d,
                M=M,
                seed=seed + int(strength * 100),
                effect_strength=strength,
                edge_fraction="auto",
                marginal_strength="auto",
                eb_k="auto",
                residual_mode="template",
            )
            omega = _candidate_omega(d, table, group_map)
            score = abs(target_omega - omega)
            if best is None or score < best[0]:
                best = (score, omega, table, group_map, f"strength={strength}")
    assert best is not None
    print(f"[fig6] selected high {modality}: {best[4]} omega={best[1]:.3f}", flush=True)
    return best[2], best[3]


def _scenario_table(d: dict, modality: str, scenario: str, M: int, seed: int):
    if scenario == "original":
        return P.real_table(d)
    if scenario == "low":
        return _low_effect_table(d, modality, M=M, seed=seed)
    if scenario == "high":
        return _high_effect_table(d, modality, M=M, seed=seed)
    raise ValueError(f"unknown scenario: {scenario}")


def _metric_distances(d: dict, modality: str, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    table = table.copy()
    if modality == "gene":
        table.index = [str(x) for x in d["post"]]
        sbf = table.T.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return {
            "Gemelli": P.recompute_distance(d, table),
            "Weighted UniFrac": _weighted_unifrac(sbf, d["tree_path"]),
            "Bray-Curtis": _bray(sbf),
            "Jaccard": _jaccard(sbf),
        }

    meta = d["meta"].reset_index(drop=True)
    samples = list(table.columns)
    long_df = pd.concat([meta, table.reset_index(drop=True)], axis=1)
    labels = meta["Taxon"].astype(str) + "|" + meta["Function"].astype(str)
    sbf = table.copy()
    sbf.index = labels
    sbf = sbf.T.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    taxa_ab = pd.concat([meta[["Taxon"]], table.reset_index(drop=True)], axis=1)
    taxa_ab[samples] = taxa_ab[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    taxa_ab = taxa_ab.groupby("Taxon", sort=False)[samples].sum().T
    return {
        "PhyloFunc": P.recompute_distance(d, long_df[samples]),
        "Weighted UniFrac": _weighted_unifrac(taxa_ab, d["tree_path"]),
        "Bray-Curtis": _bray(sbf),
        "Jaccard": _jaccard(sbf),
    }


def _within_between(dm_df: pd.DataFrame, group_map: pd.Series):
    common = dm_df.index.intersection(dm_df.columns).intersection(group_map.index)
    dm = dm_df.loc[common, common].to_numpy(float)
    g = group_map.loc[common].astype(str).to_numpy()
    iu = np.triu_indices(len(common), 1)
    vals = dm[iu]
    same = g[iu[0]] == g[iu[1]]
    return vals[same], vals[~same]


def _p_text(p: float) -> tuple[str, str]:
    if not np.isfinite(p):
        return r"$P=\mathrm{NA}$", ""
    if p <= 0:
        return r"$P<1.0\times10^{-300}$", ""
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / (10 ** exponent)
    return rf"$P={mantissa:.2f}\times10^{{{exponent}}}$", ""


def _draw_panel(ax, title: str, metric_dms: dict[str, pd.DataFrame], group_map: pd.Series, stats_rows: list[dict]):
    rng = np.random.default_rng(12345)
    spacing = 1.5
    box_w = 0.28
    jit_w = 0.055
    metrics = list(metric_dms)
    y_peak = 0.0
    brackets = []

    for i, metric in enumerate(metrics):
        base = i * spacing
        within, between = _within_between(metric_dms[metric], group_map)
        p = mannwhitneyu(between, within, alternative="two-sided").pvalue
        stats_rows.append(
            {
                "panel": title.replace("\n", " "),
                "metric": metric,
                "within_mean": float(np.mean(within)),
                "between_mean": float(np.mean(between)),
                "p_value": float(p),
                "n_within": int(len(within)),
                "n_between": int(len(between)),
            }
        )
        for vals, x, color in (
            (within, base - 0.18, C_WITHIN),
            (between, base + 0.18, C_BETWEEN),
        ):
            bp = ax.boxplot(
                [vals],
                positions=[x],
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
            )
            bp["boxes"][0].set(facecolor=color, edgecolor="#333333", alpha=0.62, linewidth=0.9)
            bp["medians"][0].set(color="black", linewidth=1.2)
            for line in bp["whiskers"] + bp["caps"]:
                line.set(color="#555555", linewidth=0.8)
            keep = vals if len(vals) <= 600 else rng.choice(vals, 600, replace=False)
            ax.scatter(
                x + rng.uniform(-jit_w, jit_w, len(keep)),
                keep,
                s=7,
                alpha=0.32,
                color=color,
                edgecolors="none",
                zorder=3,
            )
            y_peak = max(y_peak, float(np.percentile(vals, 97)))
        brackets.append((base - 0.18, base + 0.18, max(np.percentile(within, 97), np.percentile(between, 97)), p))
        print(
            f"[fig6] {title.replace(chr(10), ' ')} {metric}: "
            f"within={within.mean():.3f} between={between.mean():.3f} p={p:.2e}",
            flush=True,
        )

    h = max(y_peak * 0.045, 0.02)
    for x1, x2, y, p in brackets:
        top = y + h
        ax.plot([x1, x1, x2, x2], [y, top, top, y], lw=0.9, color="0.25", clip_on=False)
        p_line, star_line = _p_text(p)
        label = p_line if not star_line else f"{p_line}\n{star_line}"
        ax.text((x1 + x2) / 2, top + h * 0.28, label, ha="center", va="bottom", fontsize=13.2)

    ax.set_ylabel("Pairwise Distance")
    ax.set_xticks([i * spacing for i in range(len(metrics))])
    ax.set_xticklabels(metrics, rotation=28, ha="center", va="top")
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, max(y_peak * 1.28, 0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=C_WITHIN, alpha=0.7, label="Within-group"),
            mpatches.Patch(facecolor=C_BETWEEN, alpha=0.7, label="Between-group"),
        ],
        loc="upper right",
        fontsize=15,
        framealpha=0.8,
    )


def _run_modality(row_axes, modality: str, M: int, seed: int, stats_rows: list[dict]):
    d = P.load_modality(modality)
    production_metric = "Gemelli" if modality == "gene" else "PhyloFunc"
    for j, (title, scenario) in enumerate(SCENARIOS):
        table, group_map = _scenario_table(d, modality, scenario, M, seed + 1000 * j + (0 if modality == "gene" else 100))
        dms = _metric_distances(d, modality, table)
        omega = max(0.0, float(core.compute_omega2(dms[production_metric], group_map)))
        panel_title = f"{title}\nω²={omega:.3f}"
        _draw_panel(row_axes[j], panel_title, dms, group_map, stats_rows)


def add_panel_label(ax, letter: str) -> None:
    ax.text(
        -0.135,
        1.03,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=25,
        fontweight="bold",
        color="black",
    )


def draw_one(ax, modality: str, modality_label: str, scenario_title: str, scenario: str, m: int, seed: int, stats_rows: list[dict], protein_group_file: str | None = None, protein_table_file: str | None = None) -> None:
    print(f"[fig6-3x2] start {modality_label} {scenario_title}", flush=True)
    # group2.csv (terminal-ileum CD vs Control) pairs with cleaned2 table
    d = P.load_modality(modality, group_file=protein_group_file if modality == "protein" else None, table_file=protein_table_file if modality == "protein" else None)
    table, group_map = _scenario_table(d, modality, scenario, M=m, seed=seed)
    dms = _metric_distances(d, modality, table)
    production_metric = "Gemelli" if modality == "gene" else "PhyloFunc"
    omega = max(0.0, float(core.compute_omega2(dms[production_metric], group_map)))
    title = f"{modality_label} {scenario_title}\nω²={omega:.3f}"
    _draw_panel(ax, title, dms, group_map, stats_rows)
    ax.set_title(title, fontsize=15.0, fontweight="bold", color="black", pad=10)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    print(f"[fig6-3x2] done {modality_label} {scenario_title}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures" / "output")
    parser.add_argument("--pool-M", type=int, default=100, help="synthetic samples per group for effect panels")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--protein-group-file",
        default=str(ROOT / "phylopower" / "datapro" / "group2.csv"),
    )
    parser.add_argument(
        "--protein-table-file",
        default=str(ROOT / "phylopower" / "datapro" / "protein_taxon_function_cleaned2.csv"),
        help="protein group map; the manuscript panels use the terminal-ileum CD-vs-Control contrast (group2.csv)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 13.2,
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 10.5,
        }
    )
    scenarios = [
        ("Original effect", "original"),
        ("Effect enhanced", "high"),
        ("Effect diluted", "low"),
    ]
    modalities = [("gene", "Metagenomics"), ("protein", "Metaproteomics")]

    fig, axes = plt.subplots(3, 2, figsize=(16.8, 9.35), squeeze=False)
    stats_rows: list[dict] = []
    for r, (scenario_title, scenario) in enumerate(scenarios):
        for c, (modality, modality_label) in enumerate(modalities):
            seed = args.seed + r * 1000 + (0 if modality == "gene" else 10000)
            draw_one(axes[r, c], modality, modality_label, scenario_title, scenario, args.pool_M, seed, stats_rows, protein_group_file=args.protein_group_file, protein_table_file=args.protein_table_file)

    for letter, ax in zip("abcdef", axes.ravel()):
        add_panel_label(ax, letter)

    handles = [
        mpatches.Patch(facecolor=C_WITHIN, alpha=0.7, label="Within-group"),
        mpatches.Patch(facecolor=C_BETWEEN, alpha=0.7, label="Between-group"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=12.5)
    fig.tight_layout(rect=(0.025, 0.0, 0.998, 0.968), w_pad=2.25, h_pad=0.95)
    out_png = args.out / "fig6_metric_3x2_abcdef.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=260)
    fig.savefig(args.out / "fig6_metric_3x2_abcdef.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(stats_rows).to_csv(args.out / "fig6_metric_3x2_stats.csv", index=False)
    print(out_png)
    print(args.out / "fig6_metric_3x2_abcdef.pdf")


if __name__ == "__main__":
    main()
