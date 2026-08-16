#!/usr/bin/env python3
"""Figure 3 — Predicted vs observed power (MPrESS-style validation), both modalities, PCAM pipeline.

External-truth validation of the simulated power curve:
  observed(n)  = subsample n/group WITHOUT replacement from the FULL real data -> PERMANOVA -> P(reject)
  predicted(n) = subsample n/group WITHOUT replacement from the PCAM pool (pi=1, the data's own
                 effect; feature space -> recomputed REAL distance) -> PERMANOVA -> P(reject)
Both sides drawn WITHOUT replacement so the sampling scheme is identical. Only n < N_full is
validatable. The simulated (predicted) curve is trustworthy iff it tracks the observed curve.

    PATH=/opt/miniconda3/bin:$PATH /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python fig3.py --out fig3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pcam_gen as P  # noqa: E402
from phylopower import core  # noqa: E402
import figstyle  # noqa: E402

core.load_core_runtime(); figstyle.apply_style()
FIGDATA = ROOT / "figdata"


def _power_vs_n(dm_df, members_by_group, ns, *, B, perms, alpha, seed, replace):
    groups = list(members_by_group); out = {}
    for ni, n in enumerate(ns):
        rng = np.random.default_rng(seed + ni * 131); rej, oms = [], []
        for _ in range(B):
            ids, lab = [], []
            for g in groups:
                ch = rng.choice(members_by_group[g], size=n, replace=replace)
                ids.extend(ch.tolist()); lab.extend([str(g)] * n)
            uids = [f"{s}__{j}" for j, s in enumerate(ids)]
            sub = pd.DataFrame(dm_df.loc[ids, ids].to_numpy(), index=uids, columns=uids)
            gm = pd.Series(lab, index=uids, name="group")
            rej.append(core.compute_permanova_p_value(sub, gm, permutations=perms) < alpha)
            oms.append(max(0.0, float(core.compute_omega2(sub, gm))))
        r = float(np.mean(rej))
        out[n] = {"power": r, "se": float(np.sqrt(r * (1 - r) / B)), "omega2": float(np.mean(oms))}
    return out


def _eval_modality(modality, ns, args):
    d = P.load_modality(modality)
    real_dm, rsgm = P.real_distance(d)
    real_members = {str(g): rsgm[rsgm == g].index.to_numpy() for g in sorted(set(rsgm))}
    pool_dm, psgm = P.pool_distance(d, args.pool_M, args.seed + 777, 1.0, 1.0)
    pool_members = {str(g): psgm[psgm == g].index.to_numpy() for g in sorted(set(psgm))}
    obs = _power_vs_n(real_dm, real_members, ns, B=args.true_reps, perms=args.permutations,
                      alpha=args.alpha, seed=args.seed, replace=False)
    pred = _power_vs_n(pool_dm, pool_members, ns, B=args.pred_reps, perms=args.permutations,
                       alpha=args.alpha, seed=args.seed + 999, replace=False)
    nfull = min(len(v) for v in real_members.values())
    return obs, pred, nfull


def _panel(ax, ns, obs, pred, modality, n_full):
    op = [obs[n]["power"] for n in ns]; ose = [obs[n]["se"] for n in ns]; pp = [pred[n]["power"] for n in ns]
    ax.fill_between(ns, np.array(op) - 1.96 * np.array(ose), np.array(op) + 1.96 * np.array(ose),
                    color=figstyle.REAL, alpha=0.15)
    ax.plot(ns, op, "o-", color=figstyle.REAL, lw=2, ms=5, label="observed (real subsample)")
    ax.plot(ns, pp, "s--", color=figstyle.SYN, lw=2, ms=5, label="predicted (PCAM)")
    ax.axhline(0.8, color=figstyle.NEUTRAL, ls=":", lw=1)
    r = pearsonr(op, pp)[0] if len(ns) > 2 else float("nan")
    mae = float(np.mean(np.abs(np.array(op) - np.array(pp))))
    ax.text(0.03, 0.97, f"Pearson r = {r:.3f}\nmean |pred-obs| = {mae:.3f}\n(validatable n < N={n_full})",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    ax.set_xlabel("samples per group (n)"); ax.set_ylabel("power"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{modality}: predicted vs observed power"); ax.legend(loc="lower right")
    return {"pearson_r": r, "mae": mae, "ns": list(ns), "observed": op, "predicted": pp,
            "observed_se": ose, "omega2_obs": [obs[n]["omega2"] for n in ns]}


def main(argv=None):
    p = argparse.ArgumentParser(description="Figure 3: predicted vs observed power (PCAM, MPrESS-style).")
    p.add_argument("--protein-ns", default="4,6,8,10,12,14")
    p.add_argument("--gene-ns", default="3,4,5,6,7,8")
    p.add_argument("--pool-M", type=int, default=200)
    p.add_argument("--true-reps", type=int, default=400)
    p.add_argument("--pred-reps", type=int, default=400)
    p.add_argument("--permutations", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--skip-gene", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True); FIGDATA.mkdir(parents=True, exist_ok=True)
    p_ns = [int(x) for x in args.protein_ns.split(",") if x.strip()]
    g_ns = [int(x) for x in args.gene_ns.split(",") if x.strip()]

    mods = []
    if not args.skip_gene:
        print("[fig3] gene (PCAM)...", flush=True)
        g_obs, g_pred, g_nfull = _eval_modality("gene", g_ns, args)
        for n in g_ns: print(f"   gene n={n}: obs={g_obs[n]['power']:.3f} pred={g_pred[n]['power']:.3f}", flush=True)
        mods.append(("gene", "Gene (Gemelli pRPCA)", g_ns, g_obs, g_pred, g_nfull))
    print("[fig3] protein (PCAM)...", flush=True)
    p_obs, p_pred, p_nfull = _eval_modality("protein", p_ns, args)
    for n in p_ns: print(f"   protein n={n}: obs={p_obs[n]['power']:.3f} pred={p_pred[n]['power']:.3f}", flush=True)
    mods.append(("protein", "Protein (PhyloFunc)", p_ns, p_obs, p_pred, p_nfull))

    ncol = len(mods); fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.8), squeeze=False)
    summ = {}; rows = []
    for i, (key, name, ns, obs, pred, nfull) in enumerate(mods):
        summ[key] = _panel(axes[0, i], ns, obs, pred, name, nfull)
        for n in ns:
            rows.append({"modality": key, "n": n, "observed_power": obs[n]["power"], "observed_se": obs[n]["se"],
                         "predicted_power": pred[n]["power"], "omega2_obs": obs[n]["omega2"]})
    fig.suptitle("Figure 3 — Predicted (PCAM) vs observed power; both WITHOUT replacement", y=1.02, fontweight="bold", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out / "fig3.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(FIGDATA / "fig3_pred_vs_obs.csv", index=False)
    (args.out / "fig3_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(f"[fig3] data table -> {FIGDATA}/fig3_pred_vs_obs.csv", flush=True)
    print(f"[fig3] done -> {args.out}/fig3.png\n"
          f"{json.dumps({k: {'pearson_r': v['pearson_r'], 'mae': v['mae']} for k, v in summ.items()}, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
