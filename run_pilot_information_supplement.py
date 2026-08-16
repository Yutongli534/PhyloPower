#!/usr/bin/env python3
"""Repeated-pilot sensitivity analysis for a supplementary figure.

The empirical cohort, effect grid, evaluation size, and random numbers used to
generate each effect point are held fixed.  Only the pilot subset and its size
change.  Curves are compared with the curve obtained from the full empirical
cohort, so decreasing curve-to-reference error with pilot size is evidence for
finite-pilot information as the source of disagreement.

The default settings are a proof-of-concept metagenomics run.  Increase
``--reps``, ``--pool-size`` and ``--boot`` for a manuscript-quality run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phylopower-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import pcam_gen as pcam
from _fig4_curve_plotting import binned_monotone, fit_binned_null_hill


_WORKER_DATA = None


EFFECT_GRID = {
    "gene": [
        (0.50, 1.0), (0.60, 1.0), (0.68, 1.0), (0.75, 1.0),
        (0.82, 1.0), (0.88, 1.0), (0.93, 1.0), (0.97, 1.0),
        (1.00, 1.0), (1.00, 1.3), (1.00, 1.7),
    ],
    "protein": [
        (0.50, 1.0), (0.60, 1.0), (0.68, 1.0), (0.75, 1.0),
        (0.82, 1.0), (0.88, 1.0), (0.93, 1.0), (0.97, 1.0),
        (1.00, 1.0), (1.00, 1.3), (1.00, 1.7), (1.00, 2.1),
    ],
}

PALETTE = ["#4b006e", "#35679a", "#ffcf24", "#0f8b7c", "#b45309"]


def apply_style() -> None:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fit_curve(rows: pd.DataFrame, x: np.ndarray, bin_width: float) -> np.ndarray | None:
    rows = rows[["true_omega2", "power"]].dropna().sort_values("true_omega2")
    if len(rows) < 4 or rows["true_omega2"].nunique() < 3:
        return None
    binned = binned_monotone(rows, bin_width)
    y, _ = fit_binned_null_hill(binned, x)
    if y is not None and np.all(np.isfinite(y)):
        return np.clip(y, 0.0, 1.0)
    sx = rows["true_omega2"].clip(lower=0).to_numpy(float)
    sy = np.maximum.accumulate(rows["power"].clip(0, 1).to_numpy(float))
    ux, idx = np.unique(sx, return_index=True)
    if len(ux) < 2:
        return None
    return np.interp(x, ux, sy[idx], left=sy[idx][0], right=sy[idx][-1])


def crossing(x: np.ndarray, y: np.ndarray, target: float = 0.8) -> float:
    hit = np.flatnonzero(y >= target)
    if not len(hit):
        return np.nan
    i = int(hit[0])
    if i == 0:
        return float(x[0])
    x0, x1 = float(x[i - 1]), float(x[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 <= y0:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def sorted_pilot_view(data: dict, pilot_n: int, seed: int) -> dict:
    """Return a pilot view with deterministic donor ordering.

    Sorting is essential for a common-random-number experiment: if a pilot
    contains every available sample, changing the pilot seed must not merely
    permute donor positions and create artificial between-replicate variation.
    """
    rng = np.random.default_rng(seed)
    groups = data["groups"]
    selected = {
        group: np.sort(
            rng.choice(indices, min(pilot_n, len(indices)), replace=False)
        )
        for group, indices in data["gs"].items()
    }
    pooled = np.concatenate([selected[group] for group in groups])
    grand = data["L"][:, pooled].mean(axis=1)
    deviations = {
        group: data["L"][:, selected[group]].mean(axis=1) - grand
        for group in groups
    }
    view = dict(data)
    view["gs"] = selected
    view["dev"] = deviations
    view["libs"] = data["abund"][:, pooled].sum(axis=0)
    return view


def worker_init(modality: str) -> None:
    global _WORKER_DATA
    _WORKER_DATA = pcam.load_modality(modality)


def worker_task(job: tuple) -> tuple:
    from phylopower import core
    from semisynthetic_power import summarize_distance_metrics_with_replacement

    core.load_core_runtime()
    pi, scale, pool_size, generation_seed, eval_n, boot, pilot_n, pilot_seed = job
    pilot = sorted_pilot_view(_WORKER_DATA, pilot_n, pilot_seed)
    table, group_map = pcam.pcam_pool(
        pilot, pool_size, generation_seed, pi, scale, ndon=1
    )
    distance = pcam.recompute_distance(pilot, table)
    omega2 = max(0.0, float(core.compute_omega2(distance, group_map)))
    metrics = summarize_distance_metrics_with_replacement(
        dm=distance,
        group_map=group_map,
        boot_number=boot,
        alpha=0.05,
        n_jobs=1,
        random_seed=7,
        n_per_group=eval_n,
        permutations=99,
        omega2_floor=0.0,
    )
    return pilot_n, pilot_seed, pi, scale, omega2, float(metrics["power"])


def simulate_one_size(
    modality: str,
    pilot_n: int,
    reps: int,
    eval_n: int,
    pool_size: int,
    boot: int,
    seed: int,
    workers: int,
) -> pd.DataFrame:
    jobs = []
    for rep in range(reps):
        pilot_seed = seed + pilot_n * 1009 + rep * 131
        for gi, (pi, scale) in enumerate(EFFECT_GRID[modality]):
            # Common random numbers across pilot sizes/repetitions reduce Monte
            # Carlo noise unrelated to the pilot subset itself.
            generation_seed = seed + 50_000 + gi * 9973
            jobs.append(
                (pi, scale, pool_size, generation_seed, eval_n, boot, pilot_n, pilot_seed)
            )
    with ProcessPoolExecutor(
        max_workers=workers, initializer=worker_init, initargs=(modality,)
    ) as executor:
        result = list(executor.map(worker_task, jobs))
    records = []
    seed_to_rep = {
        seed + pilot_n * 1009 + rep * 131: rep for rep in range(reps)
    }
    for pn, pilot_seed, pi, scale, omega2, power in result:
        records.append(
            {
                "modality": modality,
                "pilot_n": pn,
                "rep": seed_to_rep[pilot_seed],
                "pi": pi,
                "scale": scale,
                "true_omega2": omega2,
                "power": power,
                "eval_n": eval_n,
            }
        )
    return pd.DataFrame.from_records(records)


def run_simulation(args: argparse.Namespace) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    parts = []
    for pilot_n in args.pilots:
        cache = args.out / f"raw_pilot_{pilot_n}.csv"
        if cache.exists() and not args.force:
            part = pd.read_csv(cache)
            print(f"[pilot-info] reused {cache}", flush=True)
        else:
            print(
                f"[pilot-info] {args.modality}: pilot n={pilot_n}, "
                f"reps={args.reps}, points={len(EFFECT_GRID[args.modality])}",
                flush=True,
            )
            part = simulate_one_size(
                args.modality, pilot_n, args.reps, args.eval_n,
                args.pool_size, args.boot, args.seed, args.workers,
            )
            part.to_csv(cache, index=False)
        parts.append(part)

    # A full-cohort reference.  A pilot_n much larger than either group causes
    # pilot_view() to retain every empirical sample.
    ref_cache = args.out / "raw_full_cohort_reference.csv"
    if ref_cache.exists() and not args.force:
        reference = pd.read_csv(ref_cache)
    else:
        print("[pilot-info] full-cohort reference", flush=True)
        reference = simulate_one_size(
            args.modality, 999, args.reference_reps, args.eval_n,
            # Keep the generation seed identical to the pilot runs.  Because
            # the full cohort is deterministically ordered, this makes the
            # largest possible pilot an exact common-random-number match to
            # the reference and isolates information loss from Monte Carlo
            # variation.
            args.pool_size, args.boot, args.seed, args.workers,
        )
        reference.to_csv(ref_cache, index=False)
    parts.append(reference)

    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(args.out / "pilot_information_raw.csv", index=False)
    return raw


def summarize_and_plot(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

    apply_style()
    xmax = args.xmax
    x = np.linspace(0.0, xmax, 700)
    bin_width = args.bin_width

    ref_all = raw[raw["pilot_n"].eq(999)]
    # Common-random-number reference replicates are intentionally identical.
    # Fit one copy so duplicated rows do not change the curve fitter's weights.
    ref_rows = ref_all[ref_all["rep"].eq(ref_all["rep"].min())]
    reference = fit_curve(ref_rows, x, bin_width)
    if reference is None:
        raise RuntimeError("Could not fit the full-cohort reference curve")

    curves: dict[int, list[tuple[int, np.ndarray]]] = {}
    summary = []
    for pilot_n in args.pilots:
        curves[pilot_n] = []
        subset = raw[raw["pilot_n"].eq(pilot_n)]
        for rep, rows in subset.groupby("rep"):
            curve = fit_curve(rows, x, bin_width)
            if curve is None:
                continue
            curves[pilot_n].append((int(rep), curve))
            summary.append(
                {
                    "pilot_n": pilot_n,
                    "rep": int(rep),
                    "mean_abs_curve_error": float(np.trapz(np.abs(curve - reference), x) / xmax),
                    "omega2_at_80_power": crossing(x, curve, 0.8),
                    "reference_omega2_at_80_power": crossing(x, reference, 0.8),
                }
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out / "pilot_information_curve_metrics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(args.pilots)}

    # A: median curves and repeated-pilot envelopes for every pilot size.
    ax = axes[0, 0]
    ax.plot(x, reference, color="#202020", lw=3.0, label="Full-cohort reference", zorder=5)
    for pilot_n in args.pilots:
        stack = np.vstack([curve for _, curve in curves[pilot_n]])
        color = colors[pilot_n]
        ax.fill_between(
            x, np.percentile(stack, 10, axis=0), np.percentile(stack, 90, axis=0),
            color=color, alpha=0.16, lw=0,
        )
        ax.plot(x, np.median(stack, axis=0), color=color, lw=2.0, label=f"pilot n={pilot_n}")
    ax.set_title("Repeated-pilot power curves")
    ax.set_xlabel("Realized effect size (ω²)")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", fontsize=9)

    # B: individual curves at the smallest and largest pilot sizes.
    ax = axes[0, 1]
    small, large = min(args.pilots), max(args.pilots)
    for pilot_n, linestyle in ((small, "-"), (large, "--")):
        color = colors[pilot_n]
        for j, (_, curve) in enumerate(curves[pilot_n]):
            ax.plot(
                x, curve, color=color, lw=1.0, alpha=0.38, ls=linestyle,
                label=f"pilot n={pilot_n}" if j == 0 else None,
            )
    ax.plot(x, reference, color="#202020", lw=3.0, label="Full-cohort reference", zorder=5)
    ax.set_title("Between-pilot disagreement")
    ax.set_xlabel("Realized effect size (ω²)")
    ax.set_ylabel("Power")
    ax.legend(loc="lower right", fontsize=9)

    # C: the primary quantitative test.
    ax = axes[1, 0]
    values = [
        summary_df.loc[summary_df["pilot_n"].eq(n), "mean_abs_curve_error"].dropna().to_numpy()
        for n in args.pilots
    ]
    bp = ax.boxplot(values, positions=np.arange(len(args.pilots)), widths=0.55, patch_artist=True)
    for patch, n in zip(bp["boxes"], args.pilots):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.40)
    rng = np.random.default_rng(args.seed)
    for i, (n, vals) in enumerate(zip(args.pilots, values)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=24, color=colors[n], alpha=0.80)
    ax.set_xticks(np.arange(len(args.pilots)), [str(n) for n in args.pilots])
    ax.set_title("Curve-to-reference disagreement")
    ax.set_xlabel("Pilot size per group")
    ax.set_ylabel("Mean absolute power difference")

    # D: decision-relevant stability of the 80% power threshold.
    ax = axes[1, 1]
    threshold_values = [
        summary_df.loc[summary_df["pilot_n"].eq(n), "omega2_at_80_power"].dropna().to_numpy()
        for n in args.pilots
    ]
    bp = ax.boxplot(
        threshold_values, positions=np.arange(len(args.pilots)), widths=0.55, patch_artist=True
    )
    for patch, n in zip(bp["boxes"], args.pilots):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.40)
    for i, (n, vals) in enumerate(zip(args.pilots, threshold_values)):
        ax.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals, s=24, color=colors[n], alpha=0.80)
    ref_cross = crossing(x, reference, 0.8)
    ax.axhline(ref_cross, color="#202020", lw=2.0, ls=":", label="Full-cohort reference")
    ax.set_xticks(np.arange(len(args.pilots)), [str(n) for n in args.pilots])
    ax.set_title("Stability of the 80% power threshold")
    ax.set_xlabel("Pilot size per group")
    ax.set_ylabel("ω² required for 80% power")
    ax.legend(loc="best", fontsize=9)

    for letter, ax in zip("abcd", axes.ravel()):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=20, fontweight="bold")
    for ax in axes.ravel()[:2]:
        ax.axhline(0.8, color="#7a8798", lw=1.1, ls=":")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.03)

    title_modality = "Metagenomics (Gemelli)" if args.modality == "gene" else "Metaproteomics"
    fig.suptitle(
        f"Finite-pilot disagreement and convergence — {title_modality}",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(args.out / "pilot_information_supplement.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "pilot_information_supplement.pdf", bbox_inches="tight")
    plt.close(fig)

    medians = summary_df.groupby("pilot_n", as_index=False).median(numeric_only=True)
    rho, rho_p = spearmanr(summary_df["pilot_n"], summary_df["mean_abs_curve_error"])
    paired = summary_df.pivot(
        index="rep", columns="pilot_n", values="mean_abs_curve_error"
    )
    friedman = friedmanchisquare(*(paired[n] for n in args.pilots))
    pairwise = []
    for smaller, larger in zip(args.pilots[:-1], args.pilots[1:]):
        test = wilcoxon(
            paired[smaller], paired[larger], alternative="greater"
        )
        pairwise.append(
            {
                "smaller_pilot_n": smaller,
                "larger_pilot_n": larger,
                "paired_w": float(test.statistic),
                "one_sided_p": float(test.pvalue),
            }
        )
    report = {
        "modality": args.modality,
        "eval_n": args.eval_n,
        "reps_requested": args.reps,
        "reference_omega2_at_80_power": ref_cross,
        "spearman_pilot_n_vs_curve_error": {
            "rho": float(rho),
            "p": float(rho_p),
        },
        "friedman_repeated_pilot_size_test": {
            "chi_square": float(friedman.statistic),
            "p": float(friedman.pvalue),
        },
        "adjacent_paired_wilcoxon_greater": pairwise,
        "median_metrics_by_pilot_n": medians.to_dict(orient="records"),
    }
    (args.out / "pilot_information_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["gene", "protein"], default="gene")
    parser.add_argument("--pilots", default="4,7,10")
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--reference-reps", type=int, default=2)
    parser.add_argument("--eval-n", type=int, default=80)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--boot", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--xmax", type=float, default=0.22)
    parser.add_argument("--bin-width", type=float, default=0.008)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("pilot_information_supplement"))
    args = parser.parse_args()
    args.pilots = [int(x) for x in args.pilots.split(",")]
    return args


def main() -> None:
    args = parse_args()
    raw_path = args.out / "pilot_information_raw.csv"
    if args.plot_only:
        raw = pd.read_csv(raw_path)
    else:
        raw = run_simulation(args)
    summarize_and_plot(raw, args)
    print(args.out / "pilot_information_supplement.png", flush=True)


if __name__ == "__main__":
    main()
