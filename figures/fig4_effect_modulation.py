#!/usr/bin/env python3
"""Manuscript Figure 4 - bidirectional effect modulation and sample geometry.

Panels:
  (a,b) Realized PERMANOVA omega^2 across effect levels for (a) metagenomic
        PCAM pools and (b) metaproteomic MDC-TF-MC pools, from the archived
        dense grid (data/figdata/parameter_omega_dense_current_method_v4.csv).
  (c,d) Real pilot samples overlaid on synthetic pools in the recomputed
        production-distance PCoA space at three effect settings
        (dilution / original / enhancement) for (c) metagenomics and
        (d) metaproteomics. These panels rebuild the pools (heavy; the gene
        side needs the QIIME 2 / Gemelli environment).

Run with the QIIME/Gemelli environment:

  /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig4_effect_modulation.py

Default invocation (no --compute) only plots, using the archived inputs above.

Compute submodes (self-contained gene-side data producers, ported verbatim
from the retired analysis/run_gene_fig4_panel_*.py scripts now kept in
_archive_scripts/):

  --compute dense     regenerate data/archived_runs/fig4_new/fig4_metagenomics_panel_a_dense.csv
  --compute refined   regenerate data/archived_runs/fig4_new/fig4_metagenomics_panel_a_4710_refined.csv
  --compute panel_c   regenerate data/archived_runs/fig4_new/fig4_metagenomics_panel_c_extra_pilots.csv
  --compute all       run all three producers

Each compute submode runs its producer(s) first, then the normal plotting.
Compute needs the QIIME env:
/opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "figures"))

from phylopower import core  # noqa: E402  (import first: installs the embedded-module finder)
import pcam_gen as P  # noqa: E402
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
from semisynthetic_power import _pcoa_coords  # noqa: E402

core.load_core_runtime()

DATA = ROOT / "data" / "figdata" / "parameter_omega_dense_current_method_v4.csv"
OUTDIR = ROOT / "figures" / "output"
OUT_PNG = OUTDIR / "fig4_effect_modulation.png"
OUT_PDF = OUTDIR / "fig4_effect_modulation.pdf"
OUT_CSV = OUTDIR / "fig4_effect_modulation_omega2.csv"

TREND_COLORS = {
    "gene": {
        "point": "#7570b3",
        "line": "#5e4fa2",
        "band": "#dad7f0",
    },
    "protein": {
        "point": "#1b9e77",
        "line": "#0f766e",
        "band": "#d9f0e7",
    },
}

GROUP_COLORS = ["#2B6CB0", "#B33A3A"]
SCENARIOS = [("dilution", "low"), ("original", "original"), ("enhancement", "high")]


def apply_style() -> None:
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
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.linewidth": 0.95,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.62,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Panels (a,b): parameter-to-realized-omega trajectories from the archived grid
# ---------------------------------------------------------------------------


def smooth_with_band(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    *,
    frac: float,
    seed: int,
    n_boot: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x = np.asarray(x, float)[order]
    y = np.asarray(y, float)[order]

    sm = lowess(y, x, frac=frac, it=1, return_sorted=True)
    center = np.interp(grid, sm[:, 0], sm[:, 1])

    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        b_order = np.argsort(xb)
        xb = xb[b_order]
        yb = yb[b_order]
        grouped = pd.DataFrame({"x": xb, "y": yb}).groupby("x", as_index=False)["y"].mean()
        if len(grouped) < 4:
            continue
        try:
            sb = lowess(grouped["y"], grouped["x"], frac=frac, it=1, return_sorted=True)
        except Exception:
            continue
        if len(sb) < 4:
            continue
        curves.append(np.interp(grid, sb[:, 0], sb[:, 1], left=sb[0, 1], right=sb[-1, 1]))

    if len(curves) >= 40:
        stack = np.vstack(curves)
        lo = np.nanpercentile(stack, 10, axis=0)
        hi = np.nanpercentile(stack, 90, axis=0)
    else:
        lo = center
        hi = center
    return center, lo, hi


def draw_trend_panel(
    ax,
    sub: pd.DataFrame,
    *,
    modality: str,
    frac: float,
    seed: int,
    show_ylabel: bool,
) -> None:
    colors = TREND_COLORS[modality]
    x = sub["effect_level"].to_numpy(float)
    y = sub["omega2"].to_numpy(float)
    xpad = 0.03 * (float(np.nanmax(x)) - float(np.nanmin(x)))
    grid = np.linspace(float(np.nanmin(x)) - xpad, float(np.nanmax(x)) + xpad, 500)
    center, lo, hi = smooth_with_band(x, y, grid, frac=frac, seed=seed)

    ax.fill_between(grid, lo, hi, color=colors["band"], alpha=0.55, lw=0, zorder=1)
    ax.plot(grid, center, color=colors["line"], lw=2.7, zorder=3)
    ax.scatter(x, y, s=36, color=colors["point"], alpha=0.72, edgecolor="white", linewidth=0.45, zorder=4)

    observed = sub.iloc[(sub["effect_level"].abs()).argsort().iloc[0]]
    ax.axvline(0, color="#7a8798", ls=":", lw=1.35, zorder=0)
    ax.scatter(
        [observed["effect_level"]],
        [observed["omega2"]],
        marker="D",
        s=96,
        color=colors["line"],
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

    ax.set_xlabel("Effect level")
    ax.set_ylabel("true ω²" if show_ylabel else "")
    ax.set_xlim(grid[0], grid[-1])
    ymax = float(np.nanmax([np.nanmax(y), np.nanmax(hi)]))
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(True)
    ax.tick_params(axis="both", length=4, width=0.9, color="#444444")


# ---------------------------------------------------------------------------
# Panels (c,d): real-vs-synthetic PCoA contrast along the effect axis
# ---------------------------------------------------------------------------


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
        edge_fraction="auto",
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
    print(f"[fig4] selected low {modality}: {best[3]} omega={best[2]:.3f}", flush=True)
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
        print(f"[fig4] selected original-like {modality}: {best[3]} omega={best[2]:.3f}", flush=True)
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
        print(f"[fig4] selected high {modality}: {best[3]} omega={best[2]:.3f}", flush=True)
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


def _plot_pcoa_panel(ax, title: str, d: dict, syn_table: pd.DataFrame, syn_group: pd.Series):
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
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
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

    ax.set_title(f"{title}\n$\\omega^2$={syn_omega:.3f}", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.axhline(0, color="0.88", lw=0.7, zorder=0)
    ax.axvline(0, color="0.88", lw=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return syn_omega


def _run_modality_pcoa(
    axes,
    modality: str,
    select_M: int,
    display_M: int,
    seed: int,
    rows: list[dict],
    protein_group_file: str | None = None,
    protein_table_file: str | None = None,
):
    # group2.csv (terminal-ileum CD vs Control) pairs with cleaned2 table
    d = P.load_modality(modality, group_file=protein_group_file if modality == "protein" else None, table_file=protein_table_file if modality == "protein" else None)
    label = "Protein" if modality == "protein" else "Gene"
    for j, (title, scenario) in enumerate(SCENARIOS):
        config = _select_config(d, modality, scenario, M=select_M, seed=seed + j * 1000)
        syn_table, syn_group = _make_table(d, modality, config, M=display_M, seed=seed + j * 10000 + 77)
        panel_title = f"{label} - {title}"
        om = _plot_pcoa_panel(axes[j], panel_title, d, syn_table, syn_group)
        rows.append({
            "modality": modality,
            "scenario": title,
            "synthetic_omega2": om,
            "synthetic_per_group": display_M,
            "selection_per_group": select_M,
            "config": "|".join(map(str, config)),
        })
        print(f"[fig4] {modality} {title} synthetic omega2={om:.3f}", flush=True)
    legend_loc = "upper right" if modality == "protein" else "upper left"
    axes[0].legend(
        frameon=False,
        loc=legend_loc,
        ncol=1,
        markerscale=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.25,
    )


def _sync_row_limits(row_axes) -> None:
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


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.13,
        1.05,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=26,
        fontweight="bold",
        color="black",
    )


# ---------------------------------------------------------------------------
# --compute submodes: gene-side data producers
#
# Ported verbatim (seeds, defaults, output filenames, CSV columns unchanged)
# from the retired scripts analysis/run_gene_fig4_panel_a_dense.py,
# analysis/run_gene_fig4_panel_a_4710_refined.py, and
# analysis/run_gene_fig4_panel_c_extra_pilots.py (now in _archive_scripts/).
# ---------------------------------------------------------------------------

FIG4_NEW_DIR = ROOT / "data" / "archived_runs" / "fig4_new"

_PANEL_A_DENSE_GRID = [
    (0.50, 1.00),
    (0.55, 1.00),
    (0.60, 1.00),
    (0.65, 1.00),
    (0.70, 1.00),
    (0.75, 1.00),
    (0.80, 1.00),
    (0.84, 1.00),
    (0.88, 1.00),
    (0.91, 1.00),
    (0.94, 1.00),
    (0.96, 1.00),
    (0.98, 1.00),
    (1.00, 1.00),
    (1.00, 1.10),
    (1.00, 1.20),
    (1.00, 1.30),
    (1.00, 1.45),
    (1.00, 1.60),
    (1.00, 1.80),
]


def _compute_panel_a_dense() -> None:
    """Generate a dense metagenomic Fig.4 panel-A study-size table."""
    out = FIG4_NEW_DIR / "fig4_metagenomics_panel_a_dense.csv"
    pilot_n = 10
    pilot_seed = 1000 + pilot_n * 1009 + 777
    pool_m = 300
    boot = 100
    eval_ns = (4, 7, 10, 30, 50, 80)

    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (pi, scale) in enumerate(_PANEL_A_DENSE_GRID):
        jobs.append(
            (
                float(pi),
                float(scale),
                pool_m,
                pilot_seed + 7000 + i * 131,
                eval_ns,
                boot,
                pilot_n,
                pilot_seed,
            )
        )
    print(f"[gene-panel-a-dense] jobs={len(jobs)} eval_ns={eval_ns}", flush=True)
    results = P.eval_pilot("gene", jobs, n_workers=6)

    rows = []
    for pn, pseed, pi, scale, omega, powers in results:
        for en in eval_ns:
            rows.append(
                {
                    "modality": "gene",
                    "panel": "b_dense",
                    "pilot": int(pn),
                    "pilot_seed": int(pseed),
                    "pi": float(pi),
                    "scale": float(scale),
                    "eval_n": int(en),
                    "true_omega2": float(omega),
                    "power": float(powers[en]),
                }
            )
    df = pd.DataFrame(rows).sort_values(["eval_n", "true_omega2", "pi", "scale"])
    df.to_csv(out, index=False)
    print(f"[gene-panel-a-dense] wrote {out} shape={df.shape}", flush=True)


def _compute_panel_a_4710_refined() -> None:
    """Refine metagenomic Fig.4 panel-A low-n study-size estimates."""
    out = FIG4_NEW_DIR / "fig4_metagenomics_panel_a_4710_refined.csv"
    pilot_n = 10
    pilot_seed = 1000 + pilot_n * 1009 + 777
    pool_m = 300
    boot = 500
    eval_ns = (4, 7, 10)

    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (pi, scale) in enumerate(_PANEL_A_DENSE_GRID):
        jobs.append(
            (
                float(pi),
                float(scale),
                pool_m,
                pilot_seed + 7000 + i * 131,
                eval_ns,
                boot,
                pilot_n,
                pilot_seed,
            )
        )
    print(f"[gene-panel-a-4710] jobs={len(jobs)} eval_ns={eval_ns} boot={boot}", flush=True)
    results = P.eval_pilot("gene", jobs, n_workers=6)

    rows = []
    for pn, pseed, pi, scale, omega, powers in results:
        for en in eval_ns:
            rows.append(
                {
                    "modality": "gene",
                    "panel": "b_dense_refined",
                    "pilot": int(pn),
                    "pilot_seed": int(pseed),
                    "pi": float(pi),
                    "scale": float(scale),
                    "eval_n": int(en),
                    "true_omega2": float(omega),
                    "power": float(powers[en]),
                    "boot": int(boot),
                }
            )
    df = pd.DataFrame(rows).sort_values(["eval_n", "true_omega2", "pi", "scale"])
    df.to_csv(out, index=False)
    print(f"[gene-panel-a-4710] wrote {out} shape={df.shape}", flush=True)


# --- panel-C extra pilots (process-pool worker state) -----------------------

_PANEL_C_BASE = None
_PANEL_C_PILOTS: dict = {}


def _panel_c_as_gene_raw_dict(base: dict, tab: pd.DataFrame, sgm: pd.Series) -> dict:
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


def _panel_c_make_gene_pilot(base: dict, pilot_n: int, seed: int) -> dict:
    observed_n = min(len(base["gs"][g]) for g in base["groups"])
    if pilot_n <= observed_n:
        return P.pilot_view(base, pilot_n, seed)
    tab, sgm = P.pcam_pool(base, pilot_n, seed, pi=1.0, scale=1.0, ndon=1)
    return _panel_c_as_gene_raw_dict(base, tab, sgm)


def _panel_c_init() -> None:
    import os

    global _PANEL_C_BASE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    core.load_core_runtime()
    _PANEL_C_BASE = P.load_modality("gene")


def _panel_c_task(job: tuple[int, int, float, float, int]) -> dict:
    from semisynthetic_power import summarize_distance_metrics_with_replacement

    core.load_core_runtime()
    pilot_n, pilot_seed, pi, scale, gen_seed = job
    key = (pilot_n, pilot_seed)
    if key not in _PANEL_C_PILOTS:
        _PANEL_C_PILOTS[key] = _panel_c_make_gene_pilot(_PANEL_C_BASE, pilot_n, pilot_seed)
    pilot = _PANEL_C_PILOTS[key]
    tab, sgm = P.pcam_pool(pilot, 300, gen_seed, pi=pi, scale=scale, ndon=1)
    dm = P.recompute_distance(pilot, tab)
    omega = max(0.0, float(core.compute_omega2(dm, sgm)))
    metrics = summarize_distance_metrics_with_replacement(
        dm=dm,
        group_map=sgm,
        boot_number=100,
        alpha=0.05,
        n_jobs=1,
        random_seed=gen_seed + 80 + 31,
        n_per_group=80,
        permutations=99,
        omega2_floor=0.0,
    )
    return {
        "modality": "gene",
        "panel": "c_extra",
        "pilot": int(pilot_n),
        "pilot_seed": int(pilot_seed),
        "pi": float(pi),
        "scale": float(scale),
        "eval_n": int(80),
        "true_omega2": float(omega),
        "power": float(metrics["power"]),
    }


def _compute_panel_c_extra_pilots() -> None:
    """Generate extra metagenomic Fig.4 panel-C pilot-extrapolation curves."""
    from concurrent.futures import ProcessPoolExecutor

    out = FIG4_NEW_DIR / "fig4_metagenomics_panel_c_extra_pilots.csv"
    pilots = (30, 50, 80)
    seed0 = 1000
    gene_grid = [
        (0.50, 1.00),
        (0.60, 1.00),
        (0.68, 1.00),
        (0.75, 1.00),
        (0.82, 1.00),
        (0.88, 1.00),
        (0.93, 1.00),
        (0.97, 1.00),
        (1.00, 1.00),
        (1.00, 1.30),
        (1.00, 1.70),
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pn in pilots:
        pilot_seed = seed0 + pn * 1009
        for i, (pi, scale) in enumerate(gene_grid):
            jobs.append((pn, pilot_seed, float(pi), float(scale), seed0 + pn * 10000 + i * 131))
    print(f"[gene-panel-c-extra] jobs={len(jobs)} pilots={pilots}", flush=True)
    with ProcessPoolExecutor(max_workers=6, initializer=_panel_c_init) as ex:
        rows = list(ex.map(_panel_c_task, jobs))
    df = pd.DataFrame(rows).sort_values(["pilot", "true_omega2", "pi", "scale"])
    df.to_csv(out, index=False)
    print(f"[gene-panel-c-extra] wrote {out} shape={df.shape}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTDIR)
    parser.add_argument(
        "--compute",
        choices=["dense", "refined", "panel_c", "all"],
        default=None,
        help="run a gene-side data producer first (writes data/archived_runs/fig4_new/), then plot as usual",
    )
    parser.add_argument("--pool-M", type=int, default=100, help="synthetic samples per group shown in each PCoA panel")
    parser.add_argument("--select-M", type=int, default=60, help="synthetic samples per group used for effect-parameter selection")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--protein-group-file",
        default=str(ROOT / "phylopower" / "datapro" / "group2.csv"),
        help="protein group map; the manuscript panels use the terminal-ileum CD-vs-Control contrast (group2.csv)",
    )
    parser.add_argument(
        "--protein-table-file",
        default=str(ROOT / "phylopower" / "datapro" / "protein_taxon_function_cleaned2.csv"),
        help="protein taxon-function table matching the group map (cleaned2.csv pairs with group2.csv)",
    )
    args = parser.parse_args(argv)

    if args.compute in ("dense", "all"):
        _compute_panel_a_dense()
    if args.compute in ("refined", "all"):
        _compute_panel_a_4710_refined()
    if args.compute in ("panel_c", "all"):
        _compute_panel_c_extra_pilots()

    args.out.mkdir(parents=True, exist_ok=True)

    apply_style()
    df = pd.read_csv(DATA)

    fig = plt.figure(figsize=(17.2, 9.9))
    outer = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.12], hspace=0.30, wspace=0.16)

    # (a,b) realized-omega trajectories from the archived dense grid.
    ax_a = fig.add_subplot(outer[0, 0])
    draw_trend_panel(ax_a, df[df["modality"].eq("gene")], modality="gene", frac=0.24, seed=4100, show_ylabel=True)
    ax_b = fig.add_subplot(outer[0, 1])
    draw_trend_panel(ax_b, df[df["modality"].eq("protein")], modality="protein", frac=0.28, seed=8200, show_ylabel=False)
    protein_colors = TREND_COLORS["protein"]
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=protein_colors["point"],
               markeredgecolor="white", markersize=7.5, label="Synthetic pool"),
        Line2D([0], [0], color=protein_colors["line"], lw=2.7, label="Smoothed trend"),
        Patch(facecolor=protein_colors["band"], edgecolor="none", alpha=0.55, label="Local trend band"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor=protein_colors["line"],
               markeredgewidth=1.0, markersize=8, linestyle="None", label="Observed setting"),
    ]
    ax_b.legend(handles=legend_handles, loc="lower right", handlelength=2.8)
    _panel_letter(ax_a, "a")
    _panel_letter(ax_b, "b")

    # (c,d) PCoA triptychs: gene on the left, protein on the right.
    rows: list[dict] = []
    for col, (modality, letter) in enumerate([("gene", "c"), ("protein", "d")]):
        sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, col], wspace=0.22)
        axes = [fig.add_subplot(sub[0, j]) for j in range(3)]
        _run_modality_pcoa(
            axes,
            modality,
            select_M=args.select_M,
            display_M=args.pool_M,
            seed=args.seed + (0 if modality == "protein" else 10000),
            rows=rows,
            protein_group_file=args.protein_group_file,
            protein_table_file=args.protein_table_file,
        )
        _sync_row_limits(axes)
        _panel_letter(axes[0], letter)

    fig.savefig(args.out / OUT_PNG.name, bbox_inches="tight", dpi=220)
    fig.savefig(args.out / OUT_PDF.name, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.out / OUT_CSV.name, index=False)
    print(f"[fig4] done -> {args.out / OUT_PNG.name}", flush=True)


if __name__ == "__main__":
    main()
