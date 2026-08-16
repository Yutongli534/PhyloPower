#!/usr/bin/env python3
"""Producer: MDC-TF-MC power curves with transition-omega-uniform sampling.

Computes fig4_mdctf_mc_power_curves.csv (plotting lives in figures/)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))  # shared figstyle

from phylopower import core  # noqa: E402  (import first: installs the embedded-module finder)
import figstyle  # noqa: E402
import pcam_gen as P  # noqa: E402
from _fig4_curve_plotting import draw_binned_null_hill_group
from _protein_mdctf_mc import mdctf_mc_pool
from _protein_mdctf_optimized_curve import _exchangeable_null_power
from semisynthetic_power import summarize_distance_metrics_with_replacement

core.load_core_runtime()
figstyle.apply_style()


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("fig4_mdctf_mc_full"))
    p.add_argument("--pilots", default="7,10,17")
    p.add_argument("--eval-ns", default="7,10,17,30,50,80")
    p.add_argument("--combine-csv", default=None)
    p.add_argument("--pool-M", type=int, default=180)
    p.add_argument("--preview-M", type=int, default=70)
    p.add_argument("--boot", type=int, default=100)
    p.add_argument("--perms", type=int, default=99)
    p.add_argument("--preview-boot", type=int, default=20)
    p.add_argument("--preview-perms", type=int, default=49)
    p.add_argument("--edge-fraction", type=float, default=1.0)
    p.add_argument("--marginal-strength", default="auto")
    p.add_argument("--eb-k", default="auto")
    p.add_argument("--residual-mode", choices=["random", "template"], default="random")
    p.add_argument("--strength-candidates", type=int, default=21)
    p.add_argument("--n-strengths", type=int, default=15)
    p.add_argument("--plateau-points", type=int, default=5)
    p.add_argument("--max-strengths", type=int, default=28)
    p.add_argument("--seed", type=int, default=20260627)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.combine_csv:
        df = pd.concat([pd.read_csv(x) for x in args.combine_csv.split(",")], ignore_index=True)
    else:
        df = run_rows(args)
    df.to_csv(args.out / "fig4_mdctf_mc_power_curves.csv", index=False)
    print(f"[fig4-mc] saved {args.out / 'fig4_mdctf_mc_power_curves.csv'}", flush=True)


if __name__ == "__main__":
    main()
