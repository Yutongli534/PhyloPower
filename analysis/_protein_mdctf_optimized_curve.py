#!/usr/bin/env python3
"""MDC-TF balanced with cleaner null, transition-omega-uniform grid, and structure diagnostics.

This is a conservative optimization experiment for the protein synthetic pool:

1. Keep MDC-TF balanced as the generator.
2. Choose strength points from a pilot-only preview, so curve points are uniform
   on omega^2 before saturation and still keep plateau anchors.
3. Evaluate strength=0 by an exchangeable-null bootstrap: samples are drawn from
   the pooled null pool and labels are randomized in each replicate.
4. Report degree-distribution diagnostics, not only mean ratios.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pcam_gen as P
from _protein_mdctf_curve import mdctf_pool, _resolve_auto_edge_fraction
from logistic_fit import fit_logistic, logistic_curve
from phylopower import core
from semisynthetic_power import summarize_distance_metrics_with_replacement


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    vals = np.sort(np.unique(np.concatenate([left, right])))
    lcdf = np.searchsorted(left, vals, side="right") / len(left)
    rcdf = np.searchsorted(right, vals, side="right") / len(right)
    return float(np.max(np.abs(lcdf - rcdf)))


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    lr = pd.Series(left).rank(method="average").to_numpy()
    rr = pd.Series(right).rank(method="average").to_numpy()
    return float(np.corrcoef(lr, rr)[0, 1])


def _structure_diag(d: dict, table: pd.DataFrame, sgm: pd.Series, pilot_n: int, strength: float) -> pd.DataFrame:
    taxon_ids = d["uid"].astype(int)
    n_taxa = int(taxon_ids.max()) + 1
    funcs = d["meta"]["Function"].astype(str).to_numpy()
    _, function_ids = np.unique(funcs, return_inverse=True)
    n_funcs = int(function_ids.max()) + 1
    rows: list[dict] = []
    for g in d["groups"]:
        real = d["abund"][:, d["gs"][g]] > 0
        syn = table.loc[:, sgm[sgm == g].index].to_numpy() > 0

        def summaries(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            edge = mat.sum(axis=0)
            tax_deg = np.array([len(np.unique(taxon_ids[mat[:, i]])) for i in range(mat.shape[1])])
            fun_deg = np.array([len(np.unique(funcs[mat[:, i]])) for i in range(mat.shape[1])])
            tax_prev = np.zeros(n_taxa, dtype=float)
            fun_prev = np.zeros(n_funcs, dtype=float)
            for i in range(mat.shape[1]):
                present = mat[:, i]
                if np.any(present):
                    tax_prev += np.bincount(taxon_ids[present], minlength=n_taxa) > 0
                    fun_prev += np.bincount(function_ids[present], minlength=n_funcs) > 0
            tax_prev /= max(mat.shape[1], 1)
            fun_prev /= max(mat.shape[1], 1)
            return edge, tax_deg, fun_deg, tax_prev, fun_prev

        re, rt, rf, rtp, rfp = summaries(real)
        se, st, sf, stp, sfp = summaries(syn)
        rows.append(
            {
                "pilot_n": int(pilot_n),
                "strength": float(strength),
                "group": str(g),
                "edge_mean_ratio": float(np.mean(se) / max(np.mean(re), 1e-9)),
                "taxon_degree_mean_ratio": float(np.mean(st) / max(np.mean(rt), 1e-9)),
                "function_degree_mean_ratio": float(np.mean(sf) / max(np.mean(rf), 1e-9)),
                "connectance_ratio": float((np.mean(se) / max(real.shape[0], 1)) / max(np.mean(re) / max(real.shape[0], 1), 1e-9)),
                "edge_count_ks": _ks(re, se),
                "taxon_degree_ks": _ks(rt, st),
                "function_degree_ks": _ks(rf, sf),
                "taxon_prevalence_rank_corr": _rank_corr(rtp, stp),
                "function_prevalence_rank_corr": _rank_corr(rfp, sfp),
            }
        )
    return pd.DataFrame(rows)


def _exchangeable_null_power(dm, *, eval_n: int, boot: int, perms: int, seed: int) -> float:
    ids = list(dm.index)
    arr = dm.loc[ids, ids].to_numpy(dtype=float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    rng = np.random.default_rng(int(seed))
    pvals: list[float] = []
    labels = np.array(["A"] * int(eval_n) + ["B"] * int(eval_n), dtype=object)
    for b in range(int(boot)):
        pick = rng.choice(len(ids), size=2 * int(eval_n), replace=True)
        sub = arr[np.ix_(pick, pick)]
        sub = (sub + sub.T) / 2.0
        np.fill_diagonal(sub, 0.0)
        grp = labels.copy()
        rng.shuffle(grp)
        if np.allclose(sub, 0):
            pvals.append(1.0)
            continue
        rep_ids = [f"n{b}_{i}" for i in range(sub.shape[0])]
        try:
            # Route through core so the permutation stream is seeded (skbio's
            # default seed=None is non-deterministic) and so the skbio < 0.7
            # no-seed-keyword fallback is shared with the main workflow.
            p = float(
                core.compute_permanova_p_value(
                    pd.DataFrame(sub, index=rep_ids, columns=rep_ids),
                    pd.Series(grp, index=rep_ids),
                    permutations=perms,
                    seed=core.make_permanova_seed(int(seed), b),
                )
            )
            pvals.append(1.0 if not np.isfinite(p) else p)
        except Exception:
            pvals.append(1.0)
    return float(np.mean(np.asarray(pvals) < 0.05))


def _preview_strengths(
    d: dict,
    *,
    seed: int,
    n_points: int,
    preview_M: int,
    edge_fraction: float,
    coarse_count: int,
    preview_eval_ns: list[int] | tuple[int, ...],
    preview_boot: int,
    preview_perms: int,
    plateau_points: int = 5,
) -> list[float]:
    coarse = np.linspace(0.0, 1.0, int(coarse_count))
    preview_rows: list[dict] = []
    for i, s in enumerate(coarse):
        point_seed = seed + i * 7919
        tab, sgm = mdctf_pool(d, preview_M, point_seed, float(s), edge_fraction=edge_fraction)
        dm = P.recompute_distance(d, tab)
        om = 0.0 if np.isclose(s, 0.0) else max(0.0, float(core.compute_omega2(dm, sgm)))
        powers: list[str] = []
        for en in sorted(set(int(x) for x in preview_eval_ns)):
            if np.isclose(s, 0.0):
                power = _exchangeable_null_power(
                    dm, eval_n=en, boot=preview_boot, perms=preview_perms, seed=point_seed + en + 41
                )
            else:
                metrics = summarize_distance_metrics_with_replacement(
                    dm=dm,
                    group_map=sgm,
                    boot_number=preview_boot,
                    alpha=0.05,
                    n_jobs=1,
                    random_seed=point_seed + en + 31,
                    n_per_group=en,
                    permutations=preview_perms,
                    omega2_floor=0.0,
                )
                power = float(metrics["power"])
            preview_rows.append({"strength": float(s), "eval_n": int(en), "omega2": float(om), "power": float(power)})
            powers.append(f"n{en}={power:.2f}")
        print(f"preview s={s:.3f} omega2={om:.4f} " + " ".join(powers), flush=True)

    preview = pd.DataFrame(preview_rows)
    if preview.empty:
        return [float(x) for x in coarse]
    selected: list[float] = [0.0, 1.0]
    for _, sub in preview.groupby("eval_n"):
        sub = sub.sort_values("strength")
        power = np.maximum.accumulate(sub["power"].to_numpy(dtype=float))
        omega = np.maximum.accumulate(sub["omega2"].to_numpy(dtype=float))
        strength = sub["strength"].to_numpy(dtype=float)
        if np.nanmax(omega) <= 1e-10:
            continue
        plateau_idx = np.flatnonzero(power >= 1.0 - 1e-12)
        transition_hi = float(omega[plateau_idx[0]]) if len(plateau_idx) and plateau_idx[0] > 0 else float(omega[-1])
        omega_targets = np.linspace(0.0, transition_hi, int(n_points))
        unique_omega, idx = np.unique(omega, return_index=True)
        unique_strength = strength[idx]
        selected.extend(
            np.interp(np.clip(omega_targets, unique_omega[0], unique_omega[-1]), unique_omega, unique_strength).tolist()
        )
        if int(plateau_points) > 0 and len(plateau_idx):
            selected.extend(np.linspace(float(strength[plateau_idx[0]]), 1.0, int(plateau_points)).tolist())
    rounded: list[float] = []
    for s in sorted(selected):
        val = float(np.clip(np.round(s, 4), 0.0, 1.0))
        if not rounded or abs(val - rounded[-1]) > 1e-4:
            rounded.append(val)
    return rounded


def _fit_curve(sub: pd.DataFrame, x: np.ndarray) -> np.ndarray | None:
    dat = sub[["true_omega2", "power"]].dropna()
    if len(dat) < 4:
        return None
    fit = fit_logistic(dat, floor=0.05)
    params = fit.get("params")
    if not params:
        return None
    return logistic_curve(x, params["k"], params["x0"], floor=0.05)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilots", default="7,10,19")
    p.add_argument("--eval-n", type=int, default=17)
    p.add_argument("--pool-M", type=int, default=260)
    p.add_argument("--preview-M", type=int, default=90)
    p.add_argument("--n-points", type=int, default=10)
    p.add_argument("--coarse-count", type=int, default=13)
    p.add_argument("--boot", type=int, default=80)
    p.add_argument("--perms", type=int, default=79)
    p.add_argument("--preview-boot", type=int, default=20)
    p.add_argument("--preview-perms", type=int, default=49)
    p.add_argument("--plateau-points", type=int, default=5)
    p.add_argument("--edge-fraction", type=str, default="auto")
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--out", type=Path, default=Path("data/figdata/protein_mdctf_optimized_curve"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.edge_fraction = _resolve_auto_edge_fraction(args.edge_fraction)

    base = P.load_modality("protein")
    pilots = [int(x) for x in args.pilots.split(",") if x.strip()]
    rows: list[dict] = []
    diag_frames: list[pd.DataFrame] = []
    grid_rows: list[dict] = []

    for pn in pilots:
        pilot = P.pilot_view(base, pn, args.seed + pn * 1009)
        print(f"\n[pilot {pn}] transition-omega-uniform preview", flush=True)
        strengths = _preview_strengths(
            pilot,
            seed=args.seed + pn * 10000,
            n_points=args.n_points,
            preview_M=args.preview_M,
            edge_fraction=args.edge_fraction,
            coarse_count=args.coarse_count,
            preview_eval_ns=[args.eval_n],
            preview_boot=args.preview_boot,
            preview_perms=args.preview_perms,
            plateau_points=args.plateau_points,
        )
        print(f"[pilot {pn}] selected strengths: {strengths}", flush=True)
        for s in strengths:
            grid_rows.append({"pilot_n": pn, "strength": s})

        for i, s in enumerate(strengths):
            seed = args.seed + pn * 100000 + i * 1291
            tab, sgm = mdctf_pool(pilot, args.pool_M, seed, s, edge_fraction=args.edge_fraction)
            dm = P.recompute_distance(pilot, tab)
            if np.isclose(s, 0.0):
                omega = 0.0
                power = _exchangeable_null_power(
                    dm, eval_n=args.eval_n, boot=args.boot, perms=args.perms, seed=seed + 41
                )
                eval_mode = "exchangeable_null"
            else:
                omega = max(0.0, float(core.compute_omega2(dm, sgm)))
                metrics = summarize_distance_metrics_with_replacement(
                    dm=dm,
                    group_map=sgm,
                    boot_number=args.boot,
                    alpha=0.05,
                    n_jobs=1,
                    random_seed=seed + 31,
                    n_per_group=args.eval_n,
                    permutations=args.perms,
                    omega2_floor=0.0,
                )
                power = float(metrics["power"])
                eval_mode = "labeled_bootstrap"
            rows.append(
                {
                    "method": "mdc_tf_optimized",
                    "pilot_n": int(pn),
                    "strength": float(s),
                    "eval_n": int(args.eval_n),
                    "true_omega2": float(omega),
                    "power": float(power),
                    "edge_fraction": float(args.edge_fraction),
                    "eval_mode": eval_mode,
                }
            )
            diag_frames.append(_structure_diag(pilot, tab, sgm, pn, s))
            print(
                f"optimized pilot={pn:2d} s={s:.4f} omega2={omega:.4f} "
                f"power={power:.3f} mode={eval_mode}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    pd.DataFrame(grid_rows).to_csv(args.out / "selected_strengths.csv", index=False)
    df.to_csv(args.out / "raw.csv", index=False)
    diag.to_csv(args.out / "structure_diag.csv", index=False)

    colors = {7: "#c2410c", 10: "#2563eb", 19: "#0f766e"}
    xmax = max(0.08, float(df["true_omega2"].max()) * 1.08)
    x = np.linspace(0.0, xmax, 300)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for pn in pilots:
        sub = df[df["pilot_n"] == pn].sort_values("true_omega2")
        axes[0].scatter(sub["true_omega2"], sub["power"], s=34, color=colors.get(pn), alpha=0.85, label=f"pilot {pn}")
        y = _fit_curve(sub, x)
        if y is not None:
            axes[0].plot(x, y, color=colors.get(pn), lw=2.0)
    axes[0].axhline(0.8, color="#666", ls=":", lw=1.0)
    axes[0].axhline(0.05, color="#999", ls="--", lw=0.8)
    axes[0].set_xlim(0.0, xmax)
    axes[0].set_ylim(-0.02, 1.03)
    axes[0].set_xlabel("true omega^2")
    axes[0].set_ylabel("power")
    axes[0].set_title("MDC-TF optimized curve")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)

    b = diag[np.isclose(diag["strength"], 1.0)]
    if not b.empty:
        metrics = ["edge_mean_ratio", "taxon_degree_mean_ratio", "function_degree_mean_ratio", "connectance_ratio"]
        xs = np.arange(len(metrics))
        width = 0.22
        for i, pn in enumerate(pilots):
            vals = b[b["pilot_n"] == pn][metrics].mean(axis=0)
            axes[1].bar(xs + (i - 1) * width, vals, width=width, color=colors.get(pn), alpha=0.86, label=f"pilot {pn}")
        axes[1].axhline(1.0, color="#555", ls=":", lw=1.0)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(["edge", "taxon", "function", "connectance"], rotation=15, ha="right")
        axes[1].set_ylim(0.0, 1.25)
        axes[1].set_ylabel("synthetic / pilot")
        axes[1].set_title("baseline topology ratio")
        axes[1].grid(axis="y", alpha=0.18)
    fig.suptitle(f"Protein MDC-TF optimized @ eval_n={args.eval_n}, M={args.pool_M}, boot={args.boot}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "comparison.png", dpi=180)
    print(f"saved {args.out / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
