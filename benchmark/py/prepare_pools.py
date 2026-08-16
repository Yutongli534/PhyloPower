#!/usr/bin/env python3
"""Generate PhyloPower semi-synthetic pools from the DPRS pilot data and export
everything the R-side benchmark drivers need (pool tables, group maps, distance
matrices, realized omega2 per tier).

Usage:
  python3 benchmark/py/prepare_pools.py --scan          # quick omega2 scan over a between_scale grid
  python3 benchmark/py/prepare_pools.py --export        # export pools/DMs for the chosen tiers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from phylopower import core  # noqa: E402
import semisynthetic_power as sp  # noqa: E402

DATA = REPO / "phylopower" / "datagene"
OUT = REPO / "benchmark" / "data"

SEED = 20260614
POOL_PER_GROUP = 40

# chosen after --scan; dilution (<1) and enhancement (>1) tiers around scale=1
TIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def load_pilot():
    group = pd.read_csv(DATA / "group.csv")
    group_map = pd.Series(group["group_name"].values, index=group["sample_id"].astype(str), name="group")
    raw = pd.read_csv(DATA / "table.csv")
    table = raw.rename(columns={raw.columns[0]: "Taxon"}).set_index("Taxon")
    table = table[[s for s in group_map.index if s in table.columns]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    return table, group_map


def pool_distances(pool_table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Distance matrix (samples x samples) for a feature x sample pool table."""
    from skbio.diversity import beta_diversity

    counts = pool_table.transpose().astype(float)  # samples x features
    if metric == "braycurtis":
        dm = beta_diversity("braycurtis", counts.to_numpy(), ids=list(counts.index))
    elif metric == "wunifrac":
        from skbio import TreeNode

        tree = TreeNode.read(str(DATA / "rooted-tree.nwk"))
        if len(tree.children) != 2:
            tree = tree.root_at_midpoint()
        dm = beta_diversity(
            "weighted_unifrac",
            counts.to_numpy(),
            ids=list(counts.index),
            tree=tree,
            taxa=list(pool_table.index.astype(str)),
            normalized=True,  # match phyloseq::UniFrac default (normalized=TRUE)
        )
    else:
        raise ValueError(metric)
    return dm.to_data_frame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    table, group_map = load_pilot()
    tiers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] if args.scan else TIERS

    rows = []
    for tier in tiers:
        pool_table, pool_groups, _ = sp.generate_taxon_pool(
            table,
            group_map,
            pool_size_per_group=POOL_PER_GROUP,
            random_seed=SEED,
            between_scale=float(tier),
            residual_scale=1.0,
            noise_multiplier=0.10,
        )
        for metric in ["braycurtis", "wunifrac"]:
            dm = pool_distances(pool_table, metric)
            omega2 = core.compute_omega2(dm, pool_groups)
            rows.append(
                {"between_scale": tier, "metric": metric, "omega2_full_pool": omega2}
            )
            if args.export:
                tag = f"scale{str(tier).replace('.', 'p')}"
                dm.to_csv(OUT / f"dm_{metric}_{tag}.csv")
        if args.export:
            tag = f"scale{str(tier).replace('.', 'p')}"
            pool_table.to_csv(OUT / f"pool_{tag}.csv")
            pool_groups.to_frame("group").rename_axis("sample_id").to_csv(
                OUT / f"group_{tag}.csv"
            )
        print(f"tier={tier:<5} done", flush=True)

    summary = pd.DataFrame(rows)
    if args.export:
        summary.to_csv(OUT / "tier_summary.csv", index=False)
        with open(OUT / "prepare_meta.json", "w") as fh:
            json.dump(
                {
                    "seed": SEED,
                    "pool_per_group": POOL_PER_GROUP,
                    "residual_scale": 1.0,
                    "noise_multiplier": 0.10,
                    "wunifrac_normalized": True,
                    "tiers": TIERS,
                },
                fh,
                indent=2,
            )
    print(summary.pivot(index="between_scale", columns="metric", values="omega2_full_pool"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
