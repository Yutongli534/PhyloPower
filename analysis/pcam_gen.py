"""PCAM generator and reusable distance-evaluation helpers.

PCAM preserves pilot sparsity and abundance structure by drawing synthetic
samples from pi-mixed template donors and block-level abundance donors, then
recomputing the modality-specific distance on the generated raw table.
"""
from __future__ import annotations
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from skbio import TreeNode


def clade_assign(tree, taxa, K):
    t = tree.shear(set(taxa)); t.prune(); cnt = {}
    for n in t.postorder(): cnt[id(n)] = 1 if n.is_tip() else sum(cnt[id(c)] for c in n.children)
    cl = [t.root()]
    while len(cl) < K:
        i = max(range(len(cl)), key=lambda j: cnt[id(cl[j])] if not cl[j].is_tip() else 0)
        if cl[i].is_tip(): break
        cl = cl[:i] + list(cl[i].children) + cl[i + 1:]
    a = {}
    for ci, nd in enumerate(cl):
        for tp in ([nd] if nd.is_tip() else nd.tips()): a[tp.name] = ci
    return np.array([a.get(t, 0) for t in taxa])


def _resolve_auto_blocks(value, n_taxa, *, lo=6, hi=24):
    if str(value).strip().lower() == "auto":
        return int(np.clip(round(np.sqrt(max(int(n_taxa), 1))), lo, hi))
    resolved = int(value)
    if resolved < 1:
        raise ValueError("Block count must be a positive integer or 'auto'.")
    return resolved


def load_modality(modality, K_gene="auto", K_prot=16, group_file=None, table_file=None, tree_file=None):
    """Load real pilot data + build clade blocks. Returns a dict reused by pcam_pool + recompute.
    Optional table/group/tree paths override the bundled demo data."""
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from semisynthetic_power import _read_group_map, _read_taxon_feature_table
    from phylopower import core; core.load_core_runtime()
    if modality == "protein":
        group_path = Path(group_file) if group_file is not None else core.DATAPRO_DIR / "group.csv"
        table_path = Path(table_file) if table_file is not None else core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv"
        tree_path_obj = Path(tree_file) if tree_file is not None else core.DATAPRO_DIR / "rooted-tree.nwk"
        gm = _read_group_map(group_path)
        df = pd.read_csv(table_path)
        samples = [c for c in df.columns if c in gm.index]
        df = df[["Taxon", "Function"] + samples]
        df[samples] = df[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        unit = df["Taxon"].astype(str).to_numpy(); abund = df[samples].to_numpy(float)
        taxa = pd.unique(unit); uid = pd.Series(np.arange(len(taxa)), index=taxa).loc[unit].to_numpy()
        tree = TreeNode.read(str(tree_path_obj)); K = _resolve_auto_blocks(K_prot, len(taxa), lo=6, hi=32)
        meta = df[["Taxon", "Function"]].reset_index(drop=True); tree_path = str(tree_path_obj)
        post = None
    else:
        group_path = Path(group_file) if group_file is not None else core.DATAGENE_DIR / "group.csv"
        table_path = Path(table_file) if table_file is not None else core.DATAGENE_DIR / "table.csv"
        tree_path_obj = Path(tree_file) if tree_file is not None else core.DATAGENE_DIR / "rooted-tree.nwk"
        gm = _read_group_map(group_path)
        gt, _ = _read_taxon_feature_table(table_path, gm)
        samples = list(gt.columns); abund = gt.to_numpy(float); unit = gt.index.astype(str).to_numpy()
        taxa = unit; uid = np.arange(len(unit)); tree = TreeNode.read(str(tree_path_obj))
        K = _resolve_auto_blocks(K_gene, len(taxa)); meta = None; tree_path = str(tree_path_obj); post = gt.index
    grp = gm.loc[samples].to_numpy(); groups = list(pd.unique(grp))
    gs = {g: np.where(grp == g)[0] for g in groups}
    # complement donor pool per group (uniform over all other groups; identical
    # to the single opposite group for two-group inputs, safe for k >= 2)
    other = {g: np.concatenate([gs[h] for h in groups if h != g]) for g in groups}
    rows = [np.where(clade_assign(tree, taxa, K)[uid] == c)[0] for c in range(K)]
    L = np.log1p(abund); pall = np.concatenate([gs[g] for g in groups])
    grand = L[:, pall].mean(1); dev = {g: L[:, gs[g]].mean(1) - grand for g in groups}
    return dict(modality=modality, abund=abund, L=L, dev=dev, unit=unit, uid=uid, rows=rows, gs=gs,
                groups=groups, other=other, libs=abund[:, pall].sum(0), meta=meta, tree_path=tree_path, post=post)


def pcam_pool(d, M, seed, pi, scale=1.0, ndon=None):
    """PCAM table at effect (pi, scale). pi mixes BOTH presence template and abundance donor;
    scale amplifies group log-deviation. `ndon` = donor-coherence knob: each sample draws ndon
    anchor donors (presence from anchor[0], each abundance block from one of the ndon anchors) ->
    SMALL ndon preserves real within-group dispersion/substructure; ndon=None = independent donor
    per block (max diversity, smoothest within-group). Returns (table_df, group_series)."""
    abund, L, dev, rows, groups, other, gs, libs = (d["abund"], d["L"], d["dev"], d["rows"],
        d["groups"], d["other"], d["gs"], d["libs"])
    rng = np.random.default_rng(seed); N = M * len(groups); ab = np.zeros((abund.shape[0], N)); lab = []; col = 0
    def _draw_donor(g):
        # own group with probability pi, otherwise a donor from the complement pool
        pool = gs[g] if rng.random() < pi else other[g]
        return pool[rng.integers(len(pool))]
    for g in groups:
        for j in range(M):
            if ndon is None:
                t0 = _draw_donor(g)
                def donor(): return _draw_donor(g)
            else:
                anch = []
                for _ in range(ndon):
                    anch.append(_draw_donor(g))
                t0 = anch[0]
                def donor(): return anch[rng.integers(len(anch))]
            supp = abund[:, t0] > 0; mos = np.zeros(abund.shape[0])
            for ed in rows:
                dd = donor()
                mos[ed] = np.where(abund[ed, dd] > 0, np.clip(np.expm1(L[ed, dd] + (scale - 1) * dev[g][ed]), 0, None), 0.0)
            new = np.where(supp, np.where(mos > 0, mos, abund[:, t0]), 0.0); s = new.sum()
            if s > 0: new *= libs[rng.integers(len(libs))] / s
            ab[:, col] = new; lab.append(g); col += 1
    names = [f"{lab[j][:2]}{j}" for j in range(N)]
    return pd.DataFrame(ab, columns=names), pd.Series(lab, index=names)


def pcam_null_pool(d, M, seed, ndon=1):
    """PCAM null pool: labels are assigned by design, but donors come from the
    pooled pilot distribution and no group log-deviation is applied. This is the
    left-end effect point for omega-power displays; it preserves sparsity,
    library-size, clade blocks, and donor coherence without importing a group
    signal into either label.
    """
    abund, L, rows, groups, gs = d["abund"], d["L"], d["rows"], d["groups"], d["gs"]
    all_idx = np.concatenate([gs[g] for g in groups])
    libs = abund[:, all_idx].sum(0)
    rng = np.random.default_rng(seed)
    N = M * len(groups)
    ab = np.zeros((abund.shape[0], N))
    lab = []
    col = 0
    for g in groups:
        for _ in range(M):
            anchors = list(rng.choice(all_idx, size=max(int(ndon), 1), replace=True))
            t0 = int(anchors[0])
            supp = abund[:, t0] > 0
            mos = np.zeros(abund.shape[0])
            for ed in rows:
                dd = int(anchors[int(rng.integers(len(anchors)))])
                mos[ed] = np.where(abund[ed, dd] > 0, np.expm1(L[ed, dd]), 0.0)
            new = np.where(supp, np.where(mos > 0, mos, abund[:, t0]), 0.0)
            s = new.sum()
            if s > 0:
                new *= libs[int(rng.integers(len(libs)))] / s
            ab[:, col] = new
            lab.append(g)
            col += 1
    names = [f"n{j:04d}" for j in range(N)]
    return pd.DataFrame(ab, columns=names), pd.Series(lab, index=names)


def recompute_distance(d, table):
    """Recompute the REAL distance on a synthetic table. Gene uses gemelli's Python API DIRECTLY
    (phylogenetic_rpca_without_taxonomy) — identical to `qiime gemelli` (max|diff|~1e-15) but ~600x
    faster (skips conda activate + QIIME CLI load + .qza I/O). Protein: PhyloFunc in-process."""
    from phylopower import core; core.load_core_runtime()
    if d["modality"] == "protein":
        from phylofunc_fast import phylofunc_fast      # vectorized PhyloFunc, identical to PhyloFunc_matrix (~22x faster)
        long_df = pd.concat([d["meta"], table.reset_index(drop=True)], axis=1)
        return phylofunc_fast(long_df, d["tree_path"])
    try:
        import psutil
        if psutil.cpu_count() is None:
            _cpu_count = psutil.cpu_count

            def _safe_cpu_count(logical=True):
                return _cpu_count(logical) or _cpu_count(False) or 1

            psutil.cpu_count = _safe_cpu_count
        import biom
        from skbio import DistanceMatrix
        from gemelli.rpca import phylogenetic_rpca_without_taxonomy
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The gene workflow must be launched from a compatible QIIME 2 "
            "environment containing Gemelli, biom-format, scikit-bio, and "
            "psutil (validated environment: qiime2-metagenome-2024.10). "
            "Activate that environment before running PhyloPower."
        ) from exc
    bt = biom.Table(table.to_numpy(), observation_ids=[str(x) for x in d["post"]],
                    sample_ids=[str(c) for c in table.columns])
    res = phylogenetic_rpca_without_taxonomy(table=bt, phylogeny=d["tree_path"])
    return [x for x in res if isinstance(x, DistanceMatrix)][0].to_data_frame()


def real_table(d):
    """The REAL pilot feature table as a DataFrame (features x real samples) + group series."""
    cols = [f"r{i}" for i in range(d["abund"].shape[1])]
    tab = pd.DataFrame(d["abund"], columns=cols)
    grp = np.empty(d["abund"].shape[1], object)
    for g, idx in d["gs"].items(): grp[idx] = g
    return tab, pd.Series(list(grp), index=cols)

def real_distance(d):
    """Recompute the REAL distance on the real pilot data (the reference / anchor)."""
    tab, sgm = real_table(d); return recompute_distance(d, tab), sgm

def pool_distance(d, M, seed, pi, scale=1.0, ndon=None):
    """PCAM pool at (pi, scale, ndon): table -> recomputed REAL distance + group series."""
    tab, sgm = pcam_pool(d, M, seed, pi, scale, ndon=ndon); return recompute_distance(d, tab), sgm


def _within_between(dm, sgm):
    import numpy as np
    a = dm.to_numpy(); lab = sgm.loc[list(dm.index)].to_numpy()
    iu = np.triu_indices(a.shape[0], 1); dd = a[iu]; same = lab[iu[0]] == lab[iu[1]]
    return dd[same], dd[~same]


def calibrate_fidelity(d, real_dm, rsgm, M=70, seed=1000):
    """Pick (ndon, pi) that best reproduces the real WITHIN/BETWEEN distance distributions
    (min within-KS + between-KS). ndon (coherence) matches within-group dispersion/substructure;
    pi matches between-group separation. Cheap grid at small M."""
    from scipy.stats import ks_2samp
    rw, rb = _within_between(real_dm, rsgm)
    best = (1e9, 1, 1.0)
    for ndon in [1, 2, 3]:
        for pi in [0.85, 0.9, 0.95, 1.0]:
            dm, sg = pool_distance(d, M, seed, pi, 1.0, ndon=ndon)
            sw, sb = _within_between(dm, sg)
            score = ks_2samp(rw, sw).statistic + ks_2samp(rb, sb).statistic
            if score < best[0]: best = (score, ndon, pi)
    return best[1], best[2], best[0]   # ndon, pi, score


def calibrate_to_omega(d, target_om, M=80, seed=1000):
    """Pick the effect (pi, then scale) whose pool omega^2 best matches target_om — so the generator
    runs at the DATA'S OWN effect size (not the maximal pi=1). Cheap (small M; distance fast)."""
    from phylopower import core; core.load_core_runtime()
    cands = [(pi, 1.0) for pi in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]] + \
            [(1.0, s) for s in [1.2, 1.4, 1.7, 2.0, 2.5]]
    best = (1e9, 1.0, 1.0, 0.0)
    for pi, sc in cands:
        dm, sgm = pool_distance(d, M, seed, pi, sc)
        om = max(0.0, float(core.compute_omega2(dm, sgm)))
        if abs(om - target_om) < best[0]: best = (abs(om - target_om), pi, sc, om)
    return best[1], best[2], best[3]   # pi, scale, achieved omega2


def pilot_view(d, pn, seed):
    """A PCAM 'pilot': restrict donors (and the group log-means used for the scale effect) to a
    subsample of pn real samples/group — so the generator is built FROM that pilot only."""
    if pn is None: return d
    rng = np.random.default_rng(seed)
    gs2 = {g: np.array(rng.choice(idx, min(pn, len(idx)), replace=False)) for g, idx in d["gs"].items()}
    pall = np.concatenate([gs2[g] for g in d["groups"]])
    grand = d["L"][:, pall].mean(1); dev = {g: d["L"][:, gs2[g]].mean(1) - grand for g in d["groups"]}
    e = dict(d); e["gs"] = gs2; e["dev"] = dev; e["libs"] = d["abund"][:, pall].sum(0)
    return e


# ---- parallel evaluation ----
_D = None
_PILOTS = {}
def _winit(modality): 
    global _D; _D = load_modality(modality)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

def _wtask(args):
    from phylopower import core; core.load_core_runtime()
    from semisynthetic_power import summarize_distance_metrics_with_replacement
    pi, scale, M, seed, eval_n, boot = args
    tab, sgm = pcam_pool(_D, M, seed, pi, scale)
    dm = recompute_distance(_D, tab)
    om = max(0.0, float(core.compute_omega2(dm, sgm)))
    if eval_n is None: return (pi, scale, om, np.nan)
    m = summarize_distance_metrics_with_replacement(dm=dm, group_map=sgm, boot_number=boot, alpha=0.05,
        n_jobs=1, random_seed=7, n_per_group=eval_n, permutations=199, omega2_floor=0.0)
    return (pi, scale, om, float(m["power"]))

def eval_points(modality, jobs, n_workers=6):
    """jobs: list of (pi, scale, M, seed, eval_n, boot). Returns list of (pi,scale,omega2,power)."""
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_winit, initargs=(modality,)) as ex:
        return list(ex.map(_wtask, jobs))


def _wtask_pilot(args):
    """Pilot-aware grid point: (pi, scale, M, gseed, eval_ns(tuple), boot, pilot_pn, pilot_seed).
    Distance computed ONCE; power at each eval_n by the framework bootstrap (WITH replacement);
    pool M is large so duplicate draws are negligible."""
    from phylopower import core; core.load_core_runtime()
    from semisynthetic_power import summarize_distance_metrics_with_replacement
    pi, scale, M, gseed, eval_ns, boot, pn, pseed = args
    key = (pn, pseed)
    if key not in _PILOTS: _PILOTS[key] = pilot_view(_D, pn, pseed)
    dpv = _PILOTS[key]
    tab, sgm = pcam_pool(dpv, M, gseed, pi, scale, ndon=1)   # FINAL method: ndon=1 (coherent/faithful)
    dm = recompute_distance(dpv, tab)
    om = max(0.0, float(core.compute_omega2(dm, sgm)))
    powers = {}
    for en in eval_ns:
        m = summarize_distance_metrics_with_replacement(dm=dm, group_map=sgm, boot_number=boot, alpha=0.05,
            n_jobs=1, random_seed=7, n_per_group=en, permutations=99, omega2_floor=0.0)
        powers[en] = float(m["power"])
    return (pn, pseed, pi, scale, om, powers)


def eval_pilot(modality, jobs, n_workers=6):
    """jobs: (pi, scale, M, gseed, eval_n, boot, pilot_pn, pilot_seed) -> rows with pilot+eval keys."""
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_winit, initargs=(modality,)) as ex:
        return list(ex.map(_wtask_pilot, jobs))


def _wnull(args):
    """ONE null pool, reused for ALL eval_n: n_reps without-replacement subsamples/group -> PERMANOVA p."""
    import numpy as np
    import pandas as pd
    from phylopower import core
    eval_ns, M, seed, n_reps, perms = args
    tab, sgm = pcam_pool(_D, M, seed, 0.5, 1.0)              # pi=0.5 => TRUE null (exchangeable)
    dm = recompute_distance(_D, tab); ids = list(dm.index); arr = dm.loc[ids, ids].to_numpy()
    lab = sgm.loc[ids].to_numpy(); N = len(lab); rng = np.random.default_rng(seed * 99991 + 7)
    res = {}
    for en in eval_ns:
        pvals = []
        for b in range(n_reps):
            # bootstrap WITH replacement, but draw from the WHOLE pool + RANDOM group labels: removes the
            # single pool's fixed realized null effect (which large-n bootstrap would otherwise detect).
            pick = rng.choice(N, 2 * en, replace=True)
            grp = np.array(["A"] * en + ["B"] * en); rng.shuffle(grp)
            sub = arr[np.ix_(pick, pick)]; sub = (sub + sub.T) / 2.0; np.fill_diagonal(sub, 0.0)
            if len(set(grp)) < 2 or np.allclose(sub, 0):
                pvals.append(1.0); continue
            try:
                rep_ids = [f"n{b}_{i}" for i in range(sub.shape[0])]
                # Seeded via core so the permutation stream is reproducible
                # (and skbio < 0.7 falls back to a reseeding shim there).
                p = float(core.compute_permanova_p_value(
                    pd.DataFrame(sub, index=rep_ids, columns=rep_ids),
                    pd.Series(grp, index=rep_ids),
                    permutations=perms,
                    seed=core.make_permanova_seed(seed, b),
                ))
                pvals.append(1.0 if not np.isfinite(p) else p)
            except Exception:
                pvals.append(1.0)
        res[en] = np.array(pvals)
    return res


def null_pvalues(modality, eval_ns, M, n_pool_seeds, n_reps, perms, seed0=1000, n_workers=6):
    """Returns {eval_n: concatenated null p-values} over n_pool_seeds independent pools."""
    jobs = [(tuple(eval_ns), M, seed0 + s, n_reps, perms) for s in range(n_pool_seeds)]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_winit, initargs=(modality,)) as ex:
        res = list(ex.map(_wnull, jobs))
    return {en: np.concatenate([r[en] for r in res]) for en in eval_ns}


def _wnull_chunk(args):
    """A chunk of relabel-null reps on a PRECOMPUTED distance array (no pool regeneration)."""
    import numpy as np
    import pandas as pd
    from phylopower import core
    arr, eval_ns, perms, n_chunk, cseed = args
    N = arr.shape[0]; rng = np.random.default_rng(cseed); out = {en: [] for en in eval_ns}
    for en in eval_ns:
        for b in range(n_chunk):
            pk = rng.choice(N, 2 * en, replace=True)
            gl = np.array(["A"] * en + ["B"] * en); rng.shuffle(gl)
            sub = arr[np.ix_(pk, pk)]; sub = (sub + sub.T) / 2.0; np.fill_diagonal(sub, 0.0)
            try:
                rep_ids = [f"c{cseed}_{en}_{b}_{i}" for i in range(sub.shape[0])]
                # Seeded via core so the permutation stream is reproducible
                # (and skbio < 0.7 falls back to a reseeding shim there).
                p = float(core.compute_permanova_p_value(
                    pd.DataFrame(sub, index=rep_ids, columns=rep_ids),
                    pd.Series(gl, index=rep_ids),
                    permutations=perms,
                    seed=core.make_permanova_seed(cseed, b),
                ))
                out[en].append(1.0 if not np.isfinite(p) else p)
            except Exception:
                out[en].append(1.0)
    return {en: np.array(v) for en, v in out.items()}


def null_pvalues_parallel(modality, eval_ns, M, n_reps, perms, seed=5000, n_workers=6):
    """SINGLE null pool (one generation + one distance), relabel bootstrap, with the `n_reps` reps
    SPLIT ACROSS workers (so big n_reps stays fast). Returns {eval_n: p-values}."""
    d = load_modality(modality)
    dm, sg = pool_distance(d, M, seed, 0.5, 1.0)
    arr = dm.to_numpy(); arr = (arr + arr.T) / 2.0; np.fill_diagonal(arr, 0.0)
    per = max(1, n_reps // n_workers)
    jobs = [(arr, tuple(eval_ns), perms, per, seed * 7 + 13 * k) for k in range(n_workers)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(_wnull_chunk, jobs))
    return {en: np.concatenate([p[en] for p in parts]) for en in eval_ns}
