#!/usr/bin/env python3
"""Redraw manuscript Figure 5 as a 3 x 2 panel figure.

Rows follow the narrative order: study-size family, pilot consistency, and
pilot extrapolation. Columns are metagenomics followed by metaproteomics.

Default (no arguments): plot from the archived CSVs
(`data/figdata/fig4_power_curves.csv` and
`data/archived_runs/fig4_new/fig4_mdctf_mc_power_curves.csv`); this works in
the base env and is unchanged from before. If either CSV is missing (e.g. a
release checkout without the archived data), default mode automatically
falls back to recomputing the missing side(s) as with `--compute`, and
prints a notice.

`--compute gene|protein|all`: recompute the corresponding CSV(s) first, then
plot. The compute code is ported verbatim from the retired producers
`analysis/produce_power_curves_gene.py` (PCAM pipeline ->
`data/figdata/fig4_power_curves.csv`) and
`analysis/produce_power_curves_protein.py` (MDC-TF-MC ->
`data/archived_runs/fig4_new/fig4_mdctf_mc_power_curves.csv`), both archived
under `_archive_scripts/`. The gene side needs the QIIME env (Gemelli/skbio):

    /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python figures/fig5_power_curves.py --compute gene
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _power_panels_gene as gene_fig
import _power_panels_protein as protein_fig

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures" / "output"
OUT_PNG = OUTDIR / "fig5_power_3x2_abcdef.png"
OUT_PDF = OUTDIR / "fig5_power_3x2_abcdef.pdf"

FIGDATA = ROOT / "data" / "figdata"

# ---------------------------------------------------------------------------
# Compute path, ported verbatim from analysis/produce_power_curves_gene.py
# (PCAM pipeline; heavy imports are done lazily inside compute_gene_power_curves).
# ---------------------------------------------------------------------------

GRID = {"gene": [(0.5, 1), (0.6, 1), (0.68, 1), (0.75, 1), (0.82, 1), (0.88, 1), (0.93, 1), (0.97, 1),
                 (1.0, 1), (1.0, 1.3), (1.0, 1.7)],
        "protein": [(0.5, 1), (0.6, 1), (0.68, 1), (0.75, 1), (0.82, 1), (0.88, 1), (0.93, 1), (0.97, 1),
                    (1.0, 1), (1.0, 1.3), (1.0, 1.7), (1.0, 2.1)]}


def _band(ax, x, curves, color, label):
    C = np.vstack(curves); ax.plot(x, np.median(C, 0), color=color, lw=2, zorder=3, label=label)
    if C.shape[0] > 1:
        ax.fill_between(x, np.percentile(C, 10, 0), np.percentile(C, 90, 0), color=color, alpha=0.18, zorder=1)


def _fit_curve(rows, xg, floor=0.0):    # free fit (floor=0): curve starts where the data honestly is, not forced
    df = pd.DataFrame(rows, columns=["true_omega2", "power"]).dropna()
    if len(df) < 4: return None
    fit = fit_logistic(df, floor=floor); pr = fit.get("params")
    return logistic_curve(xg, pr["k"], pr["x0"], floor=floor) if pr else None


def run_modality(axes, modality, pilots, eval_truth, study_sizes, xmax, c_xmax, n_sub, seed0, M, boot):
    grid = GRID[modality]; cof, _, _ = figstyle.seq_colors(pilots)
    # ---- build all jobs ----
    jobs = []
    # (a)+(c): each pilot subsample, sweep grid, eval at (truth, 80)
    for pn in pilots:
        for r in range(n_sub):
            pseed = seed0 + pn * 1009 + r * 131
            for gi, (pi, sc) in enumerate(grid):
                jobs.append((pi, sc, M, pseed + 9000, (eval_truth, 80), boot, pn, pseed))
    # (b): full pilot (one subsample), sweep grid, eval at study sizes
    full = max(pilots); fseed = seed0 + full * 1009; bseed = fseed + 777  # distinct key so (b) doesn't collide with (a)+(c)
    for gi, (pi, sc) in enumerate(grid):
        jobs.append((pi, sc, M, bseed + 7000, tuple(study_sizes), boot, full, bseed))
    print(f"[fig4-{modality}] {len(jobs)} jobs...", flush=True)
    res = P.eval_pilot(modality, jobs, n_workers=6)

    # ---- assemble ----
    # index results: key (pn,pseed) -> list of (om, powers dict)
    from collections import defaultdict
    byp = defaultdict(list)
    for pn, pseed, pi, sc, om, powers in res:
        byp[(pn, pseed)].append((om, powers))
    xg = np.linspace(0, xmax, 300); xgc = np.linspace(0, c_xmax, 300)
    saved = []

    # (a) consistency @ eval_truth
    for pn in pilots:
        curves = []
        for r in range(n_sub):
            pseed = seed0 + pn * 1009 + r * 131
            rows = [(om, pw[eval_truth]) for om, pw in byp[(pn, pseed)]]
            c = _fit_curve(rows, xg)
            if c is not None: curves.append(c)
            for om, pw in byp[(pn, pseed)]:
                saved.append({"modality": modality, "panel": "a", "pilot": pn, "eval_n": eval_truth,
                              "true_omega2": om, "power": pw[eval_truth]})
        if curves: _band(axes[0], xg, curves, cof[pn], f"pilot {pn}")
        print(f"[fig4-{modality}] (a) pilot {pn} done", flush=True)
    axes[0].set_title(f"{modality} (a) consistency @ eval_n={eval_truth}")
    axes[0].set_xlabel("true omega^2"); axes[0].set_ylabel("power"); axes[0].set_xlim(0, xmax); axes[0].set_ylim(-.02, 1.03)
    axes[0].legend(fontsize=7)

    # (b) study-size family from full pilot
    full = max(pilots); fseed = seed0 + full * 1009; bseed = fseed + 777
    cofe, _, _ = figstyle.seq_colors(study_sizes)
    # collect (b) rows from the dedicated full-pilot jobs (keyed by bseed, separate from (a)+(c))
    brows = defaultdict(list)
    for pn, pseed, pi, sc, om, powers in res:
        if pn == full and pseed == bseed:
            for en in study_sizes:
                if en in powers: brows[en].append((om, powers[en]))
    for en in study_sizes:
        c = _fit_curve(brows[en], xg)
        if c is not None: axes[1].plot(xg, c, color=cofe[en], lw=2, label=f"n={en}")
        for om, pw in brows[en]:
            saved.append({"modality": modality, "panel": "b", "pilot": full, "eval_n": en, "true_omega2": om, "power": pw})
    axes[1].axhline(0.8, color=figstyle.NEUTRAL, ls=":", lw=1)
    axes[1].set_title(f"{modality} (b) power curves by study size n (pilot {full})")
    axes[1].set_xlabel("true omega^2"); axes[1].set_ylabel("power"); axes[1].set_xlim(0, xmax); axes[1].set_ylim(-.02, 1.03)
    axes[1].legend(fontsize=7, title="study n")

    # (c) eval_n=80 extrapolation, banded over pilots
    for pn in pilots:
        curves = []
        for r in range(n_sub):
            pseed = seed0 + pn * 1009 + r * 131
            rows = [(om, pw[80]) for om, pw in byp[(pn, pseed)]]
            c = _fit_curve(rows, xgc)
            if c is not None: curves.append(c)
            for om, pw in byp[(pn, pseed)]:
                saved.append({"modality": modality, "panel": "c", "pilot": pn, "eval_n": 80,
                              "true_omega2": om, "power": pw[80]})
        if curves: _band(axes[2], xgc, curves, cof[pn], f"pilot {pn}")
        print(f"[fig4-{modality}] (c) pilot {pn} done", flush=True)
    axes[2].axhline(0.8, color=figstyle.NEUTRAL, ls=":", lw=1)
    axes[2].set_title(f"{modality} (c) omega^2-power @ eval_n=80")
    axes[2].set_xlabel("true omega^2"); axes[2].set_ylabel("power"); axes[2].set_xlim(0, c_xmax); axes[2].set_ylim(-.02, 1.03)
    axes[2].legend(fontsize=7)
    return saved


def compute_gene_power_curves(args) -> None:
    """Verbatim port of produce_power_curves_gene.py main() (PCAM pipeline).

    Writes `data/figdata/fig4_power_curves.csv` (and the producer's diagnostic
    `fig4.png` into --gene-out). Needs the QIIME env (Gemelli/skbio via pcam_gen).
    """
    global P, figstyle, fit_logistic, logistic_curve
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "analysis"))
    sys.path.insert(0, str(ROOT / "figures"))  # shared figstyle
    # Wiring note (not in the retired producer): phylopower must be imported before
    # semisynthetic_power (its `from phylopower import core` would otherwise re-enter
    # the embedded _protein_mdctf_curve mid-import -> circular ImportError), and fork
    # is required so pcam_gen's ProcessPoolExecutor workers inherit these
    # fully-initialized modules. Without this the retired producer dies with
    # BrokenProcessPool under macOS's default spawn start method.
    import multiprocessing as mp
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)
    from phylopower import core  # noqa: E402,F401  (import first: installs the embedded-module finder)
    core.load_core_runtime()
    import semisynthetic_power  # noqa: E402,F401
    import pcam_gen as P  # noqa: E402
    from logistic_fit import fit_logistic, logistic_curve  # noqa: E402
    import figstyle  # noqa: E402
    figstyle.apply_style()

    args.gene_out.mkdir(parents=True, exist_ok=True); FIGDATA.mkdir(parents=True, exist_ok=True)

    mods = []
    if not args.skip_protein:
        mods.append(("protein", [int(x) for x in args.protein_pilots.split(",")], args.protein_eval_truth,
                     [int(x) for x in args.protein_sizes.split(",")], args.protein_xmax, args.protein_c_xmax))
    if not args.skip_gene:
        mods.append(("gene", [int(x) for x in args.gene_pilots.split(",")], args.gene_eval_truth,
                     [int(x) for x in args.gene_sizes.split(",")], args.gene_xmax, args.gene_c_xmax))

    nrows = len(mods); fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.6 * nrows), squeeze=False)
    allrows = []
    for r, (modality, pilots, etr, sizes, xmax, cxmax) in enumerate(mods):
        allrows += run_modality(axes[r], modality, pilots, etr, sizes, xmax, cxmax,
                                args.n_sub, args.gene_seed, args.gene_pool_M, args.gene_boot)
    fig.suptitle("Figure 4 — Power-curve consistency, extrapolation, and sample-size planning (PCAM)",
                 y=1.005, fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(args.gene_out / "fig4.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(allrows).to_csv(FIGDATA / "fig4_power_curves.csv", index=False)
    print(f"[fig4] data table -> {FIGDATA}/fig4_power_curves.csv", flush=True)
    print(f"[fig4] done -> {args.gene_out}/fig4.png", flush=True)


# ---------------------------------------------------------------------------
# Compute path, ported verbatim from analysis/produce_power_curves_protein.py
# (MDC-TF-MC with transition-omega-uniform sampling; heavy imports are done
# lazily inside compute_protein_power_curves).
# ---------------------------------------------------------------------------


def _as_pilot_dict(base: dict, tab: pd.DataFrame, sgm: pd.Series) -> dict:
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


def _make_pilot(base: dict, pilot_n: int, seed: int, args: argparse.Namespace) -> tuple[dict, str]:
    observed = min(len(base["gs"][g]) for g in base["groups"])
    if pilot_n <= observed:
        return P.pilot_view(base, pilot_n, seed), "real"
    tab, sgm = mdctf_mc_pool(
        base,
        pilot_n,
        seed,
        1.0,
        edge_fraction=args.edge_fraction,
        marginal_strength=args.marginal_strength,
        eb_k=args.eb_k,
        residual_mode=args.residual_mode,
    )
    return _as_pilot_dict(base, tab, sgm), "simulated"


def _preview_strengths(d: dict, args: argparse.Namespace, eval_ns: list[int], seed: int) -> list[float]:
    coarse = np.linspace(0.0, 1.0, int(args.strength_candidates))
    rows: list[dict] = []
    for i, s in enumerate(coarse):
        point_seed = seed + i * 7919
        tab, sgm = mdctf_mc_pool(
            d,
            args.preview_M,
            point_seed,
            float(s),
            edge_fraction=args.edge_fraction,
            marginal_strength=args.marginal_strength,
            eb_k=args.eb_k,
            residual_mode=args.residual_mode,
        )
        dm = P.recompute_distance(d, tab)
        omega = 0.0 if np.isclose(s, 0.0) else max(0.0, float(core.compute_omega2(dm, sgm)))
        powers = []
        for en in sorted(set(eval_ns)):
            if np.isclose(s, 0.0):
                power = _exchangeable_null_power(
                    dm, eval_n=en, boot=args.preview_boot, perms=args.preview_perms, seed=point_seed + en + 41
                )
            else:
                metrics = summarize_distance_metrics_with_replacement(
                    dm=dm,
                    group_map=sgm,
                    boot_number=args.preview_boot,
                    alpha=0.05,
                    n_jobs=1,
                    random_seed=point_seed + en + 31,
                    n_per_group=en,
                    permutations=args.preview_perms,
                    omega2_floor=0.0,
                )
                power = float(metrics["power"])
            rows.append({"strength": float(s), "eval_n": int(en), "omega2": float(omega), "power": float(power)})
            powers.append(f"n{en}={power:.2f}")
        print(f"[preview] s={s:.3f} omega2={omega:.4f} " + " ".join(powers), flush=True)

    preview = pd.DataFrame(rows)
    selected: list[float] = [0.0, 1.0]
    for _, sub in preview.groupby("eval_n"):
        sub = sub.sort_values("strength")
        power = np.maximum.accumulate(sub["power"].to_numpy(float))
        omega = np.maximum.accumulate(sub["omega2"].to_numpy(float))
        strength = sub["strength"].to_numpy(float)
        if np.nanmax(omega) <= 1e-10:
            continue
        plateau_idx = np.flatnonzero(power >= 1.0 - 1e-12)
        transition_hi = float(omega[plateau_idx[0]]) if len(plateau_idx) and plateau_idx[0] > 0 else float(omega[-1])
        omega_targets = np.linspace(0.0, transition_hi, int(args.n_strengths))
        unique_omega, idx = np.unique(omega, return_index=True)
        unique_strength = strength[idx]
        selected.extend(
            np.interp(np.clip(omega_targets, unique_omega[0], unique_omega[-1]), unique_omega, unique_strength).tolist()
        )
        if int(args.plateau_points) > 0 and len(plateau_idx):
            selected.extend(np.linspace(float(strength[plateau_idx[0]]), 1.0, int(args.plateau_points)).tolist())
    out: list[float] = []
    for s in sorted(selected):
        val = float(np.clip(np.round(s, 4), 0.0, 1.0))
        if not out or abs(val - out[-1]) > 1e-4:
            out.append(val)
    if args.max_strengths and len(out) > int(args.max_strengths):
        keep = {0, len(out) - 1}
        keep.update(np.linspace(0, len(out) - 1, int(args.max_strengths)).round().astype(int).tolist())
        keep.update(range(max(0, len(out) - int(args.plateau_points)), len(out)))
        out = [out[i] for i in sorted(keep)]
    return out


def run_rows(args: argparse.Namespace) -> pd.DataFrame:
    base = P.load_modality("protein")
    pilots = [int(x) for x in args.pilots.split(",") if x.strip()]
    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    rows: list[dict] = []
    for pn in pilots:
        pilot, source = _make_pilot(base, pn, args.seed + pn * 1009, args)
        strengths = _preview_strengths(pilot, args, eval_ns, args.seed + pn * 10000)
        print(f"[fig4-mc] pilot={pn} source={source} strengths={strengths}", flush=True)
        for i, s in enumerate(strengths):
            point_seed = args.seed + pn * 100000 + i * 1291
            tab, sgm = mdctf_mc_pool(
                pilot,
                args.pool_M,
                point_seed,
                float(s),
                edge_fraction=args.edge_fraction,
                marginal_strength=args.marginal_strength,
                eb_k=args.eb_k,
                residual_mode=args.residual_mode,
            )
            dm = P.recompute_distance(pilot, tab)
            omega = 0.0 if np.isclose(s, 0.0) else max(0.0, float(core.compute_omega2(dm, sgm)))
            powers = []
            for en in eval_ns:
                if np.isclose(s, 0.0):
                    power = _exchangeable_null_power(
                        dm, eval_n=en, boot=args.boot, perms=args.perms, seed=point_seed + en + 41
                    )
                    mode = "exchangeable_null"
                else:
                    metrics = summarize_distance_metrics_with_replacement(
                        dm=dm,
                        group_map=sgm,
                        boot_number=args.boot,
                        alpha=0.05,
                        n_jobs=1,
                        random_seed=point_seed + en + 31,
                        n_per_group=en,
                        permutations=args.perms,
                        omega2_floor=0.0,
                    )
                    power = float(metrics["power"])
                    mode = "labeled_bootstrap"
                rows.append(
                    {
                        "pilot_n": int(pn),
                        "pilot_source": source,
                        "strength": float(s),
                        "eval_n": int(en),
                        "true_omega2": float(omega),
                        "power": float(power),
                        "eval_mode": mode,
                    }
                )
                powers.append(f"n{en}={power:.2f}")
            print(f"[fig4-mc] pilot={pn} s={s:.4f} omega2={omega:.4f} " + " ".join(powers), flush=True)
    return pd.DataFrame(rows)


def compute_protein_power_curves(args) -> None:
    """Verbatim port of produce_power_curves_protein.py main() (MDC-TF-MC).

    Writes `<--protein-out>/fig4_mdctf_mc_power_curves.csv` (default:
    `data/archived_runs/fig4_new/`, the archived grid this figure plots).
    """
    global P, figstyle, core, mdctf_mc_pool, _exchangeable_null_power, summarize_distance_metrics_with_replacement
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "analysis"))
    sys.path.insert(0, str(ROOT / "figures"))  # shared figstyle
    from phylopower import core  # noqa: E402  (import first: installs the embedded-module finder)
    import figstyle  # noqa: E402
    import pcam_gen as P  # noqa: E402
    from _fig4_curve_plotting import draw_binned_null_hill_group  # noqa: E402,F401
    from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
    from _protein_mdctf_optimized_curve import _exchangeable_null_power  # noqa: E402
    from semisynthetic_power import summarize_distance_metrics_with_replacement  # noqa: E402

    core.load_core_runtime()
    figstyle.apply_style()

    # Adapt the merged CLI back to the original producer's attribute names.
    pargs = argparse.Namespace(
        out=args.protein_out,
        pilots=args.pilots,
        eval_ns=args.eval_ns,
        combine_csv=args.combine_csv,
        pool_M=args.protein_pool_M,
        preview_M=args.preview_M,
        boot=args.protein_boot,
        perms=args.perms,
        preview_boot=args.preview_boot,
        preview_perms=args.preview_perms,
        edge_fraction=args.edge_fraction,
        marginal_strength=args.marginal_strength,
        eb_k=args.eb_k,
        residual_mode=args.residual_mode,
        strength_candidates=args.strength_candidates,
        n_strengths=args.n_strengths,
        plateau_points=args.plateau_points,
        max_strengths=args.max_strengths,
        seed=args.protein_seed,
    )
    pargs.out.mkdir(parents=True, exist_ok=True)
    if pargs.combine_csv:
        df = pd.concat([pd.read_csv(x) for x in pargs.combine_csv.split(",")], ignore_index=True)
    else:
        df = run_rows(pargs)
    df.to_csv(pargs.out / "fig4_mdctf_mc_power_curves.csv", index=False)
    print(f"[fig4-mc] saved {pargs.out / 'fig4_mdctf_mc_power_curves.csv'}", flush=True)


# ---------------------------------------------------------------------------
# Plotting (default behavior, unchanged).
# ---------------------------------------------------------------------------


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


def plot_figure() -> None:
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

    gene_df = pd.read_csv(gene_fig.DATA)
    gene_df = gene_df[gene_df["modality"].eq("gene")].copy()
    if gene_fig.PANEL_A_DENSE.exists():
        gene_panel_a = pd.read_csv(gene_fig.PANEL_A_DENSE)
    else:
        gene_panel_a = gene_df[
            (gene_df["panel"].eq("b"))
            & (gene_df["pilot"].eq(10))
            & (gene_df["eval_n"].isin(gene_fig.STUDY_SIZES))
        ].copy()
    if gene_fig.PANEL_A_4710_REFINED.exists():
        refined = pd.read_csv(gene_fig.PANEL_A_4710_REFINED)
        gene_panel_a = pd.concat(
            [gene_panel_a[~gene_panel_a["eval_n"].isin([4, 7, 10])], refined],
            ignore_index=True,
        )
    gene_panel_c = gene_df[
        (gene_df["panel"].eq("c"))
        & (gene_df["pilot"].isin(gene_fig.PILOTS))
        & (gene_df["eval_n"].eq(80))
    ].copy()
    gene_panel_c_keys = gene_fig.PILOTS
    gene_panel_c_colors = gene_fig.PILOT_COLORS
    if gene_fig.PANEL_C_EXTRA.exists():
        gene_panel_c = pd.concat([gene_panel_c, pd.read_csv(gene_fig.PANEL_C_EXTRA)], ignore_index=True)
        gene_panel_c_keys = gene_fig.PILOTS_EXTENDED
        gene_panel_c_colors = gene_fig.PILOT_EXTENDED_COLORS

    protein_df = pd.read_csv(protein_fig.DATA)

    fig, axes = plt.subplots(3, 2, figsize=(16.7, 9.35), squeeze=False)

    gene_fig.draw_points_panel(
        axes[0, 0],
        gene_panel_a[gene_panel_a["eval_n"].isin(gene_fig.STUDY_SIZES)],
        by="eval_n",
        keys=gene_fig.STUDY_SIZES,
        colors=gene_fig.SIZE_COLORS,
        title="Metagenomics study-size family (pilot n=10)",
        bin_width=0.008,
        legend_loc="lower right",
        legend_ncol=3,
        legend_frame=True,
    )
    protein_fig.draw_points_panel(
        axes[0, 1],
        protein_df[(protein_df["pilot_n"] == 17) & (protein_df["eval_n"].isin(protein_fig.PILOT_KEYS))],
        by="eval_n",
        keys=protein_fig.PILOT_KEYS,
        title="Metaproteomics study-size family (pilot n=17)",
        bin_width=0.004,
        legend_loc="lower right",
        legend_ncol=3,
        legend_frame=True,
    )

    gene_fig.draw_band_panel(
        axes[1, 0],
        gene_df[(gene_df["panel"].eq("a")) & (gene_df["pilot"].isin(gene_fig.PILOTS)) & (gene_df["eval_n"].eq(10))],
        by="pilot",
        keys=gene_fig.PILOTS,
        colors=gene_fig.PILOT_COLORS,
        title="Metagenomics pilot consistency (eval n=10)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[1, 1],
        protein_df[(protein_df["pilot_n"].isin([7, 10, 17])) & (protein_df["eval_n"] == 17)],
        by="pilot_n",
        keys=[7, 10, 17],
        title="Metaproteomics pilot consistency (eval n=17)",
        bin_width=0.004,
        seed_base=3500,
    )

    gene_fig.draw_band_panel(
        axes[2, 0],
        gene_panel_c[gene_panel_c["pilot"].isin(gene_fig.PILOTS_EXTENDED)],
        by="pilot",
        keys=gene_panel_c_keys,
        colors=gene_panel_c_colors,
        title="Metagenomics pilot extrapolation (eval n=80)",
        bin_width=0.008,
    )
    protein_fig.draw_band_panel(
        axes[2, 1],
        protein_df[(protein_df["pilot_n"].isin(protein_fig.PILOT_KEYS)) & (protein_df["eval_n"] == 80)],
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


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compute", choices=["gene", "protein", "all"], default=None,
                   help="recompute the corresponding power-curve CSV(s) before plotting "
                        "(default: plot from the archived CSVs only)")

    g = p.add_argument_group("gene compute knobs (PCAM producer; default output: data/figdata/fig4_power_curves.csv)")
    g.add_argument("--protein-pilots", default="7,10,19")
    g.add_argument("--protein-eval-truth", type=int, default=17)
    g.add_argument("--protein-sizes", default="10,17,30,50,80")
    g.add_argument("--protein-xmax", type=float, default=0.30)
    g.add_argument("--protein-c-xmax", type=float, default=0.08)
    g.add_argument("--gene-pilots", default="4,7,10")
    g.add_argument("--gene-eval-truth", type=int, default=10)
    g.add_argument("--gene-sizes", default="6,10,30,50,80")
    g.add_argument("--gene-xmax", type=float, default=0.50)
    g.add_argument("--gene-c-xmax", type=float, default=0.18)
    g.add_argument("--n-sub", type=int, default=1)          # one pilot draw = a real study's single realization
    g.add_argument("--gene-pool-M", type=int, default=300)  # large pool -> with-replacement bootstrap ~ i.i.d. at eval_n<=80
    g.add_argument("--gene-boot", type=int, default=500)
    g.add_argument("--skip-gene", action="store_true")
    g.add_argument("--skip-protein", action="store_true")
    g.add_argument("--gene-seed", type=int, default=1000)
    g.add_argument("--gene-out", type=Path, default=OUTDIR,
                   help="directory for the producer's diagnostic fig4.png")

    q = p.add_argument_group("protein compute knobs (MDC-TF-MC producer; default output: "
                             "data/archived_runs/fig4_new/fig4_mdctf_mc_power_curves.csv)")
    q.add_argument("--protein-out", type=Path, default=ROOT / "data" / "archived_runs" / "fig4_new")
    q.add_argument("--pilots", default="7,10,17")
    q.add_argument("--eval-ns", default="7,10,17,30,50,80")
    q.add_argument("--combine-csv", default=None)
    q.add_argument("--protein-pool-M", type=int, default=180)
    q.add_argument("--preview-M", type=int, default=70)
    q.add_argument("--protein-boot", type=int, default=100)
    q.add_argument("--perms", type=int, default=99)
    q.add_argument("--preview-boot", type=int, default=20)
    q.add_argument("--preview-perms", type=int, default=49)
    q.add_argument("--edge-fraction", type=float, default=1.0)
    q.add_argument("--marginal-strength", default="auto")
    q.add_argument("--eb-k", default="auto")
    q.add_argument("--residual-mode", choices=["random", "template"], default="random")
    q.add_argument("--strength-candidates", type=int, default=21)
    q.add_argument("--n-strengths", type=int, default=15)
    q.add_argument("--plateau-points", type=int, default=5)
    q.add_argument("--max-strengths", type=int, default=28)
    q.add_argument("--protein-seed", type=int, default=20260627)
    args = p.parse_args(argv)

    if args.compute is None:
        # Fallback: the release ships no archived CSVs, so when a required
        # plotting input is missing, recompute that side first (same as
        # --compute with default knobs) instead of dying with
        # FileNotFoundError. The fig4_metagenomics_panel_* CSVs stay
        # optional: plot_figure() already degrades gracefully without them.
        gene_csv = FIGDATA / "fig4_power_curves.csv"
        protein_csv = args.protein_out / "fig4_mdctf_mc_power_curves.csv"
        missing = [p for p in (gene_csv, protein_csv) if not p.exists()]
        if missing:
            args.compute = (
                "all" if len(missing) == 2 else ("gene" if missing[0] == gene_csv else "protein")
            )
            print(
                f"[fig5] archived data not found ({', '.join(str(m) for m in missing)}); "
                f"computing from scratch (--compute {args.compute}; this can take a while"
                + (" and needs the QIIME 2 env for gene" if args.compute in ("gene", "all") else "")
                + ") ...",
                flush=True,
            )
    if args.compute in ("gene", "all"):
        compute_gene_power_curves(args)
    if args.compute in ("protein", "all"):
        compute_protein_power_curves(args)
    plot_figure()


if __name__ == "__main__":
    main()
