#!/usr/bin/env python3
"""Figure 7: real-vs-synthetic contrast along the omega-power effect axis.

Synthetic samples are drawn as a pale background cloud; real pilot samples are
overlaid as larger saturated points in the same recomputed production-distance
PCoA space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import figstyle  # noqa: E402
import pcam_gen as P  # noqa: E402
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
from phylopower import core  # noqa: E402
from semisynthetic_power import _pcoa_coords  # noqa: E402

core.load_core_runtime()
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
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 19,
        "axes.labelsize": 19,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = ["#2B6CB0", "#B33A3A"]
SCENARIOS = [("dilution", "low"), ("original", "original"), ("enhancement", "high")]


def _omega(d: dict, table: pd.DataFrame, group_map: pd.Series) -> float:
    dm = P.recompute_distance(d, table)
    return max(0.0, float(core.compute_omega2(dm, group_map)))


def _make_table(d: dict, modality: str, config: tuple, M: int, seed: int):
    if modality == "gene":
        _, pi, scale = config
        return P.pcam_pool(d, M=M, seed=seed, pi=pi, scale=scale, ndon=1)
    _, strength = config
    return mdctf_mc_pool(
        d,
        M=M,
        seed=seed,
        effect_strength=strength,
        edge_fraction=1.25,
        marginal_strength="auto",
        eb_k="auto",
        residual_mode="template",
    )


def _select_low_config(d: dict, modality: str, M: int, seed: int, target_low: float = 0.04):
    best = None
    if modality == "gene":
        candidates = [(pi, scale) for pi in [0.50, 0.52, 0.55, 0.60, 0.65] for scale in [0.05, 0.10, 0.20, 0.35, 0.50]]
        for pi, scale in candidates:
            config = ("gene", pi, scale)
            table, group_map = _make_table(d, modality, config, M=M, seed=seed + int(pi * 1000 + scale * 100))
            om = _omega(d, table, group_map)
            if om <= 1e-5:
                continue
            score = abs(target_low - om)
            if best is None or score < best[0]:
                best = (score, config, om, f"pi={pi}, scale={scale}")
    else:
        for strength in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]:
            config = ("protein", strength)
            table, group_map = _make_table(d, modality, config, M=M, seed=seed + int(strength * 1000))
            om = _omega(d, table, group_map)
            if om <= 1e-5:
                continue
            score = abs(target_low - om)
            if best is None or score < best[0]:
                best = (score, config, om, f"strength={strength}")
    if best is None:
        raise RuntimeError(f"no positive low-effect candidate for {modality}")
    print(f"[fig7] selected low {modality}: {best[3]} omega={best[2]:.3f}", flush=True)
    return best[1]


def _select_config(d: dict, modality: str, scenario: str, M: int, seed: int, target_high: float = 0.80):
    if scenario == "low":
        return _select_low_config(d, modality, M=M, seed=seed)
    if scenario == "original":
        real_table, real_group = P.real_table(d)
        target = _omega(d, real_table, real_group)
        best = None
        if modality == "gene":
            candidates = [(pi, scale) for pi in [0.85, 0.9, 0.95, 1.0] for scale in [0.8, 1.0, 1.2, 1.5, 2.0]]
            for pi, scale in candidates:
                config = ("gene", pi, scale)
                table, group_map = _make_table(d, modality, config, M=M, seed=seed + int(pi * 1000 + scale * 100))
                om = _omega(d, table, group_map)
                score = abs(target - om)
                if best is None or score < best[0]:
                    best = (score, config, om, f"pi={pi}, scale={scale}")
        else:
            for strength in [0.6, 0.8, 1.0, 1.2, 1.5]:
                config = ("protein", strength)
                table, group_map = _make_table(d, modality, config, M=M, seed=seed + int(strength * 1000))
                om = _omega(d, table, group_map)
                score = abs(target - om)
                if best is None or score < best[0]:
                    best = (score, config, om, f"strength={strength}")
        assert best is not None
        print(f"[fig7] selected original-like {modality}: {best[3]} omega={best[2]:.3f}", flush=True)
        return best[1]

    if scenario == "high":
        best = None
        if modality == "gene":
            for scale in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
                config = ("gene", 1.0, scale)
                table, group_map = _make_table(d, modality, config, M=M, seed=seed + int(scale * 100))
                om = _omega(d, table, group_map)
                score = abs(target_high - om)
                if best is None or score < best[0]:
                    best = (score, config, om, f"scale={scale}")
        else:
            for strength in [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]:
                config = ("protein", strength)
                table, group_map = _make_table(d, modality, config, M=M, seed=seed + int(strength * 100))
                om = _omega(d, table, group_map)
                score = abs(target_high - om)
                if best is None or score < best[0]:
                    best = (score, config, om, f"strength={strength}")
        assert best is not None
        print(f"[fig7] selected high {modality}: {best[3]} omega={best[2]:.3f}", flush=True)
        return best[1]

    raise ValueError(f"unknown scenario: {scenario}")


def _combined_table(d: dict, syn_table: pd.DataFrame, syn_group: pd.Series):
    real_table, real_group = P.real_table(d)
    real_table = real_table.copy()
    syn_table = syn_table.copy()
    real_table.columns = [f"real_{c}" for c in real_table.columns]
    syn_table.columns = [f"syn_{c}" for c in syn_table.columns]
    table = pd.concat([real_table, syn_table], axis=1)
    groups = pd.concat(
        [
            pd.Series(real_group.to_numpy(), index=real_table.columns),
            pd.Series(syn_group.to_numpy(), index=syn_table.columns),
        ]
    )
    source = pd.Series(
        ["Real"] * real_table.shape[1] + ["Synthetic"] * syn_table.shape[1],
        index=list(real_table.columns) + list(syn_table.columns),
    )
    return table, groups, source, list(syn_table.columns)


def _plot_panel(ax, title: str, d: dict, syn_table: pd.DataFrame, syn_group: pd.Series):
    table, groups, source, syn_ids = _combined_table(d, syn_table, syn_group)
    dm = P.recompute_distance(d, table)
    coords = _pcoa_coords(dm)
    if coords.shape[1] < 2:
        coords[1] = 0.0
    groups = groups.loc[coords.index].astype(str)
    source = source.loc[coords.index]
    syn_dm = dm.loc[syn_ids, syn_ids]
    syn_omega = max(0.0, float(core.compute_omega2(syn_dm, groups.loc[syn_ids])))

    group_order = list(pd.unique(groups))
    for i, group in enumerate(group_order):
        color = COLORS[i % len(COLORS)]
        syn_mask = (groups == group) & (source == "Synthetic")
        real_mask = (groups == group) & (source == "Real")
        ax.scatter(
            coords.loc[syn_mask, 0],
            coords.loc[syn_mask, 1],
            s=7,
            facecolors=color,
            alpha=0.22,
            edgecolors="none",
            linewidths=0,
            marker="o",
            antialiaseds=False,
            rasterized=True,
            zorder=1,
        )
        ax.scatter(
            coords.loc[real_mask, 0],
            coords.loc[real_mask, 1],
            s=42,
            color=color,
            alpha=0.96,
            edgecolors="#FFFFFF",
            linewidths=0.7,
            marker="o",
            label=str(group),
            zorder=4,
        )

    ax.set_title(f"{title}\n$\\omega^2$={syn_omega:.3f}", fontsize=19, fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.axhline(0, color="0.88", lw=0.7, zorder=0)
    ax.axvline(0, color="0.88", lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return syn_omega


def _run_modality(
    axes,
    modality: str,
    select_M: int,
    display_M: int,
    seed: int,
    rows: list[dict],
    protein_group_file: str | None = None,
):
    d = P.load_modality(modality, group_file=protein_group_file if modality == "protein" else None)
    label = "Protein" if modality == "protein" else "Gene"
    for j, (title, scenario) in enumerate(SCENARIOS):
        config = _select_config(d, modality, scenario, M=select_M, seed=seed + j * 1000)
        syn_table, syn_group = _make_table(d, modality, config, M=display_M, seed=seed + j * 10000 + 77)
        panel_title = f"{label} - {title}"
        om = _plot_panel(axes[j], panel_title, d, syn_table, syn_group)
        rows.append({
            "modality": modality,
            "scenario": title,
            "synthetic_omega2": om,
            "synthetic_per_group": display_M,
            "selection_per_group": select_M,
            "config": "|".join(map(str, config)),
        })
        print(f"[fig7] {modality} {title} synthetic omega2={om:.3f}", flush=True)
    legend_loc = "upper right" if modality == "protein" else "upper left"
    axes[0].legend(
        fontsize=16,
        frameon=False,
        loc=legend_loc,
        ncol=1,
        markerscale=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.25,
    )


def _sync_row_limits(row_axes):
    """Use shared x/y limits within one modality row."""
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for ax in row_axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        xmins.append(x0)
        xmaxs.append(x1)
        ymins.append(y0)
        ymaxs.append(y1)
    xmin, xmax = min(xmins), max(xmaxs)
    ymin, ymax = min(ymins), max(ymaxs)
    xpad = 0.04 * max(xmax - xmin, 1e-9)
    ypad = 0.04 * max(ymax - ymin, 1e-9)
    for ax in row_axes:
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool-M", type=int, default=100, help="synthetic samples per group shown in each panel")
    parser.add_argument("--select-M", type=int, default=60, help="synthetic samples per group used for effect-parameter selection")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--skip-gene", action="store_true")
    parser.add_argument("--skip-protein", action="store_true")
    parser.add_argument("--protein-group-file", default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    modalities = []
    if not args.skip_protein:
        modalities.append(("protein", "a"))
    if not args.skip_gene:
        modalities.append(("gene", "b"))

    fig, axes = plt.subplots(len(modalities), 3, figsize=(11.4, 4.45 * len(modalities)), squeeze=False)
    rows: list[dict] = []
    for i, (modality, letter) in enumerate(modalities):
        _run_modality(
            axes[i],
            modality,
            select_M=args.select_M,
            display_M=args.pool_M,
            seed=args.seed + i * 10000,
            rows=rows,
            protein_group_file=args.protein_group_file,
        )
        _sync_row_limits(axes[i])
        axes[i, 0].text(-0.13, 1.08, letter, transform=axes[i, 0].transAxes, fontsize=38, fontweight="bold")

    fig.suptitle("Figure 7 - Effect modulation: real pilot over synthetic pool", y=1.01, fontsize=23)
    fig.tight_layout(w_pad=0.02, h_pad=1.15)
    out_png = args.out / "fig7.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.out / "fig7_omega2.csv", index=False)
    print(f"[fig7] done -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
