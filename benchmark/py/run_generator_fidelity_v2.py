#!/usr/bin/env python3
"""Generator fidelity benchmark v2: fair 3-way comparison using Jaccard.

ALL tools generate synthetic OTU tables from the same 10/group pilot, recompute
Jaccard distance, and compare against full-cohort Jaccard ground truth.

  micropower:  per-group Dirichlet-multinomial (dirmult R package)
  MPrESS:      per-group HMP Dirichlet-multinomial mixture
  PhyloPower:  PCAM phylogenetic clade-block mosaic

Usage:
  python3 benchmark/py/run_generator_fidelity_v2.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "analysis"))

from phylopower import core; core.load_core_runtime()  # noqa: E402
import semisynthetic_power as sp  # noqa: E402

SEED = 20260614
POOL_SIZE = 200
N_PILOT = 10
N_REPEATS = 5
RSCRIPT = "/opt/miniconda3/envs/phylopower-r-benchmark/bin/Rscript"

QINJ_FULL = _REPO / "validation_datasets" / "processed" / "QinJ_2012_full"
OUT = _REPO / "benchmark" / "results"
FIG = _REPO / "benchmark" / "figures"


def write_r_scripts(tmpdir):
    """Write R generator scripts to temp files."""

    # --- micropower (dirmult) ---
    (tmpdir / "gen_micropower.R").write_text("""
suppressMessages({ library(dirmult); library(vegan) })
args <- commandArgs(trailingOnly = TRUE)
# args: pilot_csv group_csv out_csv pool_size seed
pilot <- round(as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE)))
grp <- read.csv(args[2], stringsAsFactors = FALSE)
out_file <- args[3]
pool_size <- as.integer(args[4])
set.seed(as.integer(args[5]))
groups <- unique(grp$group)
gen_all <- list()
for (g in groups) {
  g_samples <- grp$sample_id[grp$group == g]
  g_table <- pilot[, g_samples, drop = FALSE]
  g_table <- g_table[rowSums(g_table) > 0, , drop = FALSE]
  # dirmult expects SAMPLES x FEATURES
  g_table_t <- t(g_table)
  fit <- dirmult(g_table_t)
  K <- nrow(g_table)
  n_reads <- max(1000, round(mean(colSums(g_table))))
  sim <- tryCatch(simPop(J = pool_size, K = K, n = n_reads,
                          pi = fit$pi, theta = fit$theta),
                  error = function(e) NULL)
  if (is.null(sim)) {
    prob <- colSums(g_table_t) / sum(g_table_t)
    sim_data <- t(rmultinom(pool_size, n_reads, prob))  # pool_size x K
  } else {
    sim_data <- sim$data  # SAMPLES x FEATURES
  }
  # Transpose to FEATURES x SAMPLES
  sim_data <- t(sim_data)
  rownames(sim_data) <- colnames(g_table_t)
  colnames(sim_data) <- paste0(g, "_s", 1:pool_size)
  gen_all[[g]] <- sim_data
}
pool_cols <- pool_size * length(groups)
all_taxa <- rownames(pilot)
gen_table <- matrix(0, nrow = length(all_taxa), ncol = pool_cols)
rownames(gen_table) <- all_taxa
colnames(gen_table) <- character(pool_cols)
ci <- 1
for (g in groups) {
  common <- intersect(all_taxa, rownames(gen_all[[g]]))
  gen_table[common, ci:(ci + pool_size - 1)] <- gen_all[[g]][common, ]
  colnames(gen_table)[ci:(ci + pool_size - 1)] <- colnames(gen_all[[g]])
  ci <- ci + pool_size
}
gen_labels <- rep(groups, each = pool_size)
jac <- as.matrix(vegdist(t(gen_table), method = "jaccard", binary = TRUE))
# Vectorized within/between extraction using dist indices
n_tot <- ncol(gen_table)
pairs <- which(upper.tri(jac), arr.ind = TRUE)
same_group <- gen_labels[pairs[, 1]] == gen_labels[pairs[, 2]]
w <- jac[upper.tri(jac)][same_group]
b <- jac[upper.tri(jac)][!same_group]
write.csv(data.frame(group = c(rep("within", length(w)), rep("between", length(b))),
                       distance = c(w, b)),
            out_file, row.names = FALSE)
cat(sprintf("  micropower OK: within=%d between=%d\\n", length(w), length(b)))
""")

    # --- MPrESS (HMP dirmult) ---
    (tmpdir / "gen_mpress.R").write_text("""
suppressMessages({ library(HMP); library(vegan); library(dirmult) })
args <- commandArgs(trailingOnly = TRUE)
pilot <- round(as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE)))
grp <- read.csv(args[2], stringsAsFactors = FALSE)
out_file <- args[3]
pool_size <- as.integer(args[4])
set.seed(as.integer(args[5]))
groups <- unique(grp$group)
gen_all <- list()
for (g in groups) {
  g_samples <- grp$sample_id[grp$group == g]
  g_table <- pilot[, g_samples, drop = FALSE]
  g_table <- g_table[rowSums(g_table) > 0, , drop = FALSE]
  # HMP::dirmult expects SAMPLES x FEATURES
  g_table_t <- t(g_table)
  fit <- tryCatch(HMP::dirmult(g_table_t), error = function(e) NULL)
  K <- nrow(g_table)
  n_reads <- max(1000, round(mean(colSums(g_table))))
  if (is.null(fit) || inherits(fit, "try-error")) {
    prob <- colSums(g_table_t) / sum(g_table_t)
    sim_data <- t(rmultinom(pool_size, n_reads, prob))
  } else {
    # HMP fit provides pi and theta like dirmult
    sim <- tryCatch(dirmult::simPop(J = pool_size, K = K, n = n_reads,
                       pi = fit$pi, theta = fit$theta),
                    error = function(e) NULL)
    if (is.null(sim)) {
      prob <- colSums(g_table_t) / sum(g_table_t)
      sim_data <- t(rmultinom(pool_size, n_reads, prob))
    } else {
      sim_data <- sim$data
    }
  }
  sim_data <- t(sim_data)
  rownames(sim_data) <- colnames(g_table_t)
  colnames(sim_data) <- paste0(g, "_h", 1:pool_size)
  gen_all[[g]] <- sim_data
}
pool_cols <- pool_size * length(groups)
all_taxa <- rownames(pilot)
gen_table <- matrix(0, nrow = length(all_taxa), ncol = pool_cols)
rownames(gen_table) <- all_taxa
colnames(gen_table) <- character(pool_cols)
ci <- 1
for (g in groups) {
  common <- intersect(all_taxa, rownames(gen_all[[g]]))
  gen_table[common, ci:(ci + pool_size - 1)] <- gen_all[[g]][common, ]
  colnames(gen_table)[ci:(ci + pool_size - 1)] <- colnames(gen_all[[g]])
  ci <- ci + pool_size
}
gen_labels <- rep(groups, each = pool_size)
jac <- as.matrix(vegdist(t(gen_table), method = "jaccard", binary = TRUE))
n_tot <- ncol(gen_table)
pairs <- which(upper.tri(jac), arr.ind = TRUE)
same_group <- gen_labels[pairs[, 1]] == gen_labels[pairs[, 2]]
w <- jac[upper.tri(jac)][same_group]
b <- jac[upper.tri(jac)][!same_group]
write.csv(data.frame(group = c(rep("within", length(w)), rep("between", length(b))),
                       distance = c(w, b)),
            out_file, row.names = FALSE)
cat(sprintf("  MPrESS OK: within=%d between=%d\\n", length(w), length(b)))
""")

    return {
        "micropower": str(tmpdir / "gen_micropower.R"),
        "mpress": str(tmpdir / "gen_mpress.R"),
    }


def load_qinj_full():
    table = pd.read_csv(QINJ_FULL / "table.csv", index_col=0)
    grp = pd.read_csv(QINJ_FULL / "group.csv")
    gmap = pd.Series(grp["group_name"].values, index=grp["sample_id"].astype(str))
    return table, gmap


def compute_jaccard_dm(table):
    from skbio.diversity import beta_diversity
    counts = table.transpose().astype(float)
    pa = (counts > 0).to_numpy().astype(float)
    dm = beta_diversity("jaccard", pa, ids=list(counts.index))
    return dm.to_data_frame()


def within_between_dists(dm_df, gmap):
    common = dm_df.index.intersection(gmap.index)
    dm_df = dm_df.loc[common, common]; gmap = gmap.loc[common]
    arr = dm_df.to_numpy(); ids = list(dm_df.index)
    w, b = [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if gmap[ids[i]] == gmap[ids[j]]:
                w.append(arr[i, j])
            else:
                b.append(arr[i, j])
    return np.array(w), np.array(b)


def ks_eval(w_true, b_true, w_gen, b_gen):
    kw = ks_2samp(w_true, w_gen).statistic
    kb = ks_2samp(b_true, b_gen).statistic
    return float(kw), float(kb), float((kw + kb) / 2)


def run_r(r_script_path, args, tmpdir):
    cmd = [RSCRIPT, "--vanilla", r_script_path] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       cwd=str(tmpdir))
    if r.stdout.strip():
        # Only print non-empty lines
        for line in r.stdout.strip().split("\n"):
            if line.strip():
                print(f"  {line.strip()}", flush=True)
    if r.returncode != 0:
        # Print last few error lines
        err = r.stderr.strip()
        if err:
            err_lines = err.split("\n")
            relevant = [l for l in err_lines[-5:]
                        if "Error" in l or "error" in l or "fatal" in l]
            if relevant:
                print(f"  [R ERROR] {'; '.join(relevant)}", flush=True)


def main():
    table_full, gmap_full = load_qinj_full()
    groups = sorted(gmap_full.unique())
    print(f"Full cohort: {len(table_full.columns)} samples, "
          f"{dict(zip(groups, [(gmap_full == g).sum() for g in groups]))}")

    print("Computing ground-truth Jaccard on full cohort ...", flush=True)
    dm_truth = compute_jaccard_dm(table_full)
    w_true, b_true = within_between_dists(dm_truth, gmap_full)
    print(f"  within={len(w_true)} pairs, between={len(b_true)} pairs", flush=True)

    rng = np.random.default_rng(SEED)
    tmpdir = _REPO / "benchmark" / "data" / "gen_fid_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    r_scripts = write_r_scripts(tmpdir)

    all_results = []
    for rep in range(N_REPEATS):
        pilot_seed = int(rng.integers(2**31 - 1))
        pilot = []
        for g in groups:
            members = gmap_full[gmap_full == g].index.to_numpy()
            prng = np.random.default_rng(pilot_seed + abs(hash(g)) % (2**31))
            pilot.extend(prng.choice(members, size=N_PILOT, replace=False).tolist())
        pilot = sorted(pilot)
        n_per = {g: sum(1 for s in pilot if gmap_full[s] == g) for g in groups}
        print(f"\nPilot {rep+1}/{N_REPEATS} (seed={pilot_seed}): {n_per}", flush=True)

        result = {"pilot_seed": pilot_seed}
        table_pilot = table_full[pilot]
        gmap_pilot = gmap_full.loc[pilot]

        # Export pilot data
        ptab = tmpdir / f"pilot_{pilot_seed}_table.csv"
        pgrp = tmpdir / f"pilot_{pilot_seed}_group.csv"
        table_pilot.to_csv(ptab)
        pd.DataFrame({
            "sample_id": pilot,
            "group": [gmap_pilot[s] for s in pilot],
        }).to_csv(pgrp, index=False)

        # --- PhyloPower PCAM ---
        pool, pool_gmap, _ = sp.generate_taxon_pool(
            table_pilot, gmap_pilot,
            pool_size_per_group=POOL_SIZE,
            random_seed=pilot_seed + 1000,
            between_scale=1.0, residual_scale=1.0, noise_multiplier=0.10,
        )
        dm_pp = compute_jaccard_dm(pool)
        w_pp, b_pp = within_between_dists(dm_pp, pool_gmap)
        kw, kb, km = ks_eval(w_true, b_true, w_pp, b_pp)
        result["phylopower_ks_within"] = kw
        result["phylopower_ks_between"] = kb
        result["phylopower_ks_mean"] = km
        print(f"  PhyloPower     within={kw:.3f} between={kb:.3f} mean={km:.3f}", flush=True)

        # --- micropower (dirmult) ---
        mp_out = tmpdir / f"mp_out_{pilot_seed}.csv"
        run_r(r_scripts["micropower"],
              [str(ptab), str(pgrp), str(mp_out), str(POOL_SIZE),
               str(pilot_seed + 2000)],
              tmpdir)
        if mp_out.exists() and mp_out.stat().st_size > 0:
            try:
                mp_df = pd.read_csv(mp_out)
                w_mp = mp_df[mp_df["group"] == "within"]["distance"].dropna().to_numpy()
                b_mp = mp_df[mp_df["group"] == "between"]["distance"].dropna().to_numpy()
                if len(w_mp) > 0 and len(b_mp) > 0:
                    kw, kb, km = ks_eval(w_true, b_true, w_mp, b_mp)
                    result["micropower_ks_within"] = kw
                    result["micropower_ks_between"] = kb
                    result["micropower_ks_mean"] = km
                    print(f"  micropower     within={kw:.3f} between={kb:.3f} mean={km:.3f}", flush=True)
                else:
                    print(f"  micropower     FAILED (empty output)", flush=True)
            except Exception as e:
                print(f"  micropower     FAILED: {e}", flush=True)
        else:
            print(f"  micropower     FAILED (no output file)", flush=True)

        # --- MPrESS (HMP dirmult) ---
        ms_out = tmpdir / f"ms_out_{pilot_seed}.csv"
        run_r(r_scripts["mpress"],
              [str(ptab), str(pgrp), str(ms_out), str(POOL_SIZE),
               str(pilot_seed + 3000)],
              tmpdir)
        if ms_out.exists() and ms_out.stat().st_size > 0:
            try:
                ms_df = pd.read_csv(ms_out)
                w_ms = ms_df[ms_df["group"] == "within"]["distance"].dropna().to_numpy()
                b_ms = ms_df[ms_df["group"] == "between"]["distance"].dropna().to_numpy()
                if len(w_ms) > 0 and len(b_ms) > 0:
                    kw, kb, km = ks_eval(w_true, b_true, w_ms, b_ms)
                    result["mpress_ks_within"] = kw
                    result["mpress_ks_between"] = kb
                    result["mpress_ks_mean"] = km
                    print(f"  MPrESS         within={kw:.3f} between={kb:.3f} mean={km:.3f}", flush=True)
                else:
                    print(f"  MPrESS         FAILED (empty output)", flush=True)
            except Exception as e:
                print(f"  MPrESS         FAILED: {e}", flush=True)
        else:
            print(f"  MPrESS         FAILED (no output file)", flush=True)

        # --- Observed pilot baseline ---
        dm_pilot = compute_jaccard_dm(table_pilot)
        w_po, b_po = within_between_dists(dm_pilot, gmap_pilot)
        kw, kb, km = ks_eval(w_true, b_true, w_po, b_po)
        result["pilot_ks_within"] = kw
        result["pilot_ks_between"] = kb
        result["pilot_ks_mean"] = km
        print(f"  Pilot baseline within={kw:.3f} between={kb:.3f} mean={km:.3f}", flush=True)

        all_results.append(result)

    # Save and summarize
    df = pd.DataFrame(all_results)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "generator_fidelity_v2.csv", index=False)
    
    print("\n" + "=" * 70)
    print("GENERATOR FIDELITY — Jaccard KS vs full-cohort ground truth")
    print("(lower KS = better match to population Jaccard distribution)")
    print("=" * 70)
    methods = [
        ("pilot", "Observed pilot (baseline)"),
        ("micropower", "micropower (dirmult)"),
        ("mpress", "MPrESS (HMP dirmult)"),
        ("phylopower", "PhyloPower (PCAM)"),
    ]
    for prefix, label in methods:
        vals = {}
        for m in ["within", "between", "mean"]:
            col = f"{prefix}_ks_{m}"
            if col in df.columns and df[col].notna().any():
                vals[m] = df[col].dropna()
        if vals:
            print(f"\n{label}:")
            print(f"  within  KS = {vals['within'].mean():.3f} ± {vals['within'].std():.3f}")
            print(f"  between KS = {vals['between'].mean():.3f} ± {vals['between'].std():.3f}")
            print(f"  mean    KS = {vals['mean'].mean():.3f} ± {vals['mean'].std():.3f}")
    
    print(f"\n→ {OUT / 'generator_fidelity_v2.csv'}", flush=True)


if __name__ == "__main__":
    main()
