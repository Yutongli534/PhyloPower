"""Protein taxon-function table preprocessing transforms (dimensionality reduction / denoising),
motivated by metaproteomic data being intrinsically low-rank (functional redundancy), very sparse,
and heavy-tailed. All transforms keep intensities NON-NEGATIVE so PhyloFunc stays valid.

  vst       (D): log1p variance-stabilising transform (tames heavy tails).
  lowrank   (A): log1p -> truncated-SVD rank-r denoise -> expm1 (recovers the low-rank signal,
                 discards high-dim noise that small pilots estimate unstably -- the Gemelli analog).
  aggregate (C): collapse taxa into ~n_clades phylogenetic clades (tree patristic clustering) and
                 sum intensities within (clade-representative, function); reduces the taxon dimension.
"""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

ID_COLS = ["Taxon", "Function"]


def _sample_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ID_COLS]


def transform_table(long_df: pd.DataFrame, kind: str, tree_path: str,
                    rank: int = 10, n_clades: int = 300) -> pd.DataFrame:
    if kind in (None, "none"):
        return long_df
    sc = _sample_cols(long_df)
    M = long_df[sc].to_numpy(dtype=float)            # features x samples
    if kind == "vst":
        out = long_df.copy(); out[sc] = np.log1p(M)
        return out
    if kind == "lowrank":
        L = np.log1p(M)
        # center per sample (column) so SVD captures co-variation, not the mean offset
        mu = L.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(L - mu, full_matrices=False)
        r = int(max(1, min(rank, len(s))))
        Lr = (U[:, :r] * s[:r]) @ Vt[:r] + mu
        rec = np.clip(np.expm1(Lr), 0.0, None)
        out = long_df.copy(); out[sc] = rec
        return out
    if kind == "aggregate":
        return _phylo_aggregate(long_df, tree_path, n_clades)
    raise ValueError(f"unknown protein transform {kind!r}")


def _phylo_aggregate(long_df: pd.DataFrame, tree_path: str, n_clades: int) -> pd.DataFrame:
    from skbio import TreeNode
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    taxa = sorted(long_df["Taxon"].astype(str).unique())
    tree = TreeNode.read(tree_path)
    present = set(t.name for t in tree.tips())
    keep = [t for t in taxa if t in present]
    if len(keep) <= n_clades or len(keep) < 3:
        return long_df
    sub = tree.shear(keep)
    dm = sub.tip_tip_distances()                      # patristic distances among kept taxa
    ids = list(dm.ids)
    Z = linkage(squareform(dm.data, checks=False), method="average")
    labels = fcluster(Z, t=n_clades, criterion="maxclust")
    # representative taxon per clade = first member; map every taxon -> its clade representative
    rep_of = {}
    clade_rep: dict = {}
    for tx, cl in zip(ids, labels):
        clade_rep.setdefault(cl, tx)
        rep_of[tx] = clade_rep[cl]
    mapped = long_df.copy()
    mapped["Taxon"] = mapped["Taxon"].astype(str).map(lambda t: rep_of.get(t, t))
    agg = mapped.groupby(["Taxon", "Function"], as_index=False).sum(numeric_only=True)
    return agg
