"""Multi-group (k >= 2) support tests for the PCAM pool generator."""
from __future__ import annotations

import pandas as pd
import pytest

from phylopower import core  # noqa: F401  (installs embedded-module finder)
import pcam_gen

core.load_core_runtime()
DATA = core.DATAGENE_DIR


@pytest.fixture(scope="module")
def three_group_inputs(tmp_path_factory):
    """Derive a small 3-group gene dataset from the bundled demo data."""
    tmp = tmp_path_factory.mktemp("mg")
    table = pd.read_csv(DATA / "table.csv", index_col=0).iloc[:80, :]
    table = table.iloc[:, :18]  # 18 samples for speed
    samples = list(table.columns)
    groups = (["Cd"] * 6) + (["Ni"] * 6) + (["Cr"] * 6)
    group = pd.DataFrame({"sample_id": samples, "group_name": groups})
    table_path = tmp / "table.csv"
    group_path = tmp / "group.csv"
    table.to_csv(table_path)
    group.to_csv(group_path, index=False)
    return table_path, group_path, DATA / "rooted-tree.nwk"


def test_load_modality_accepts_three_groups(three_group_inputs):
    table_path, group_path, tree_path = three_group_inputs
    d = pcam_gen.load_modality(
        "gene", group_file=str(group_path), table_file=str(table_path), tree_file=str(tree_path)
    )
    assert len(d["groups"]) == 3
    # complement donor pools: each excludes its own group and covers the rest
    for g in d["groups"]:
        own = set(d["gs"][g].tolist())
        comp = set(d["other"][g].tolist())
        assert own.isdisjoint(comp)
        assert len(comp) == 12


@pytest.mark.parametrize("pi", [1.0, 0.7])
def test_pcam_pool_three_groups(three_group_inputs, pi):
    table_path, group_path, tree_path = three_group_inputs
    d = pcam_gen.load_modality(
        "gene", group_file=str(group_path), table_file=str(table_path), tree_file=str(tree_path)
    )
    syn_table, syn_groups = pcam_gen.pcam_pool(d, M=6, seed=123, pi=pi)
    assert syn_table.shape[1] == 18  # 3 groups x 6 per group
    counts = pd.Series(syn_groups).value_counts().sort_index()
    assert counts.tolist() == [6, 6, 6]
    assert syn_table.values.sum() > 0


def test_two_group_behavior_unchanged():
    """Two-group pools must still build with identical output shape and labels."""
    d = pcam_gen.load_modality("gene")
    assert len(d["groups"]) == 2
    syn_table, syn_groups = pcam_gen.pcam_pool(d, M=5, seed=7, pi=1.0)
    assert syn_table.shape[1] == 10
    assert sorted(pd.Series(syn_groups).unique().tolist()) == sorted(d["groups"])
