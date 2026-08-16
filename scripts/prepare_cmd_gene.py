#!/usr/bin/env python3
"""Convert curatedMetagenomicData relative-abundance data to PhyloPower gene inputs.

Reads a per-study ``*.relative_abundance.rda`` matrix (MetaPhlAn-style
hierarchical taxonomy strings), the package ``sampleMetadata.rda`` and the
package MetaPhlAn phylogenetic tree in ``R/sysdata.rda`` (both extracted from
``curatedMetagenomicData_3.20.0.tar.gz``), and writes the PhyloPower gene
four-file input set:

- ``table.csv``       species-level ppm pseudo-counts (fraction x 1e6), Taxon x sample
- ``taxonomy.csv``    QIIME-style ``Feature ID,Taxon,Confidence``
- ``rooted-tree.nwk`` package phylogenetic tree pruned to the retained species
- ``group.csv``       two-group case/control assignment for the selected samples

Sample selection keeps ``--per-group`` samples per group, drawn with
gender-stratified random sampling (proportional allocation, largest
remainder) at a fixed seed. Only species rows that exactly match a tree tip
are retained; the script aborts when the match rate falls below 0.90.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.setrecursionlimit(100000)

RSCRIPT_DEFAULT = "/opt/miniconda3/envs/phylopower-r-benchmark/bin/Rscript"
SELECTION_SEED = 20260614
MIN_MATCH_RATE = 0.90

DATASETS = {
    "QinJ_2012": {
        "case": "T2D",
        "control": "control",
        "title": "type 2 diabetes versus control",
    },
    "YachidaS_2019": {
        "case": "CRC",
        "control": "control",
        "title": "colorectal cancer versus control",
    },
}

R_DUMP = r"""
args <- commandArgs(trailingOnly = TRUE)
abundance_rda <- args[1]
metadata_rda <- args[2]
sysdata_rda <- args[3]
study <- args[4]
out_dir <- args[5]

ea <- new.env()
load(abundance_rda, envir = ea)
mat <- get(ls(ea)[1], envir = ea)
spp <- grepl("[|]s__", rownames(mat)) & !grepl("[|]t__", rownames(mat))
mat <- mat[spp, , drop = FALSE]
write.csv(mat, file.path(out_dir, "abundance_species.csv"), quote = TRUE)

em <- new.env()
load(metadata_rda, envir = em)
md <- get("sampleMetadata", envir = em)
md <- md[md$study_name == study,
         c("sample_id", "subject_id", "study_condition", "disease", "gender", "country")]
write.csv(md, file.path(out_dir, "sample_metadata.csv"), row.names = FALSE)

es <- new.env()
load(sysdata_rda, envir = es)
tr <- get("phylogeneticTree", envir = es)
tips <- data.frame(node = seq_along(tr$tip.label), label = tr$tip.label)
write.csv(tips, file.path(out_dir, "tree_tips.csv"), row.names = FALSE)
edges <- data.frame(parent = tr$edge[, 1], child = tr$edge[, 2], length = tr$edge.length)
write.csv(edges, file.path(out_dir, "tree_edges.csv"), row.names = FALSE)
writeLines(as.character(length(tr$tip.label) + 1L), file.path(out_dir, "tree_root.txt"))
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qiime_taxonomy(feature: str) -> str:
    parts = feature.split("|")
    converted = []
    for part in parts:
        if part.startswith("k__"):
            converted.append("d__" + part[3:])
        else:
            converted.append(part)
    return ";".join(converted)


def _pruned_newick(
    tips: pd.DataFrame,
    edges: pd.DataFrame,
    root: int,
    keep: set[str],
) -> tuple[str, int]:
    """Serialize the package tree restricted to ``keep`` tip labels.

    Unary internal nodes are collapsed (branch lengths summed). Returns the
    Newick string and the number of retained tips.
    """
    label_of = dict(zip(tips["node"].astype(int), tips["label"]))
    children: dict[int, list[tuple[int, float]]] = {}
    for parent, child, length in edges.itertuples(index=False):
        children.setdefault(int(parent), []).append((int(child), float(length)))
    counter = {"tips": 0}

    def serialize(node: int, edge_len: float) -> tuple[str, float] | None:
        if node in label_of:
            label = label_of[node]
            if label in keep:
                counter["tips"] += 1
                # Quote tip labels so Newick readers keep underscores verbatim.
                return f"'{label}'", edge_len
            return None
        parts: list[tuple[str, float]] = []
        for child, child_len in children.get(node, []):
            piece = serialize(child, child_len)
            if piece is not None:
                parts.append(piece)
        if not parts:
            return None
        if len(parts) == 1:
            text, length = parts[0]
            return text, length + edge_len
        inner = "(" + ",".join(f"{text}:{length:.8g}" for text, length in parts) + ")"
        return inner, edge_len

    root_piece = serialize(int(root), 0.0)
    if root_piece is None:
        raise ValueError("No retained features remain in the pruned tree.")
    text, _ = root_piece
    if not text.startswith("("):
        text = f"({text})"
    return text + ";\n", counter["tips"]


def _stratified_sample(
    metadata: pd.DataFrame,
    group_name: str,
    per_group: int,
    seed: int,
) -> list[str]:
    pool = metadata[metadata["group_name"] == group_name]
    if len(pool) < per_group:
        raise ValueError(
            f"Group {group_name!r} has only {len(pool)} samples; cannot draw {per_group}."
        )
    rng = np.random.default_rng(seed)
    strata = pool["gender"].fillna("unknown").astype(str)
    counts = strata.value_counts()
    raw_alloc = counts / counts.sum() * per_group
    alloc = raw_alloc.astype(int)
    remainder = per_group - int(alloc.sum())
    if remainder > 0:
        order = (raw_alloc - alloc).sort_values(ascending=False).index
        for name in list(order[:remainder]):
            alloc[name] += 1
    selected: list[str] = []
    for stratum in sorted(alloc.index):
        members = pool[strata == stratum]["sample_id"].to_numpy()
        take = int(alloc[stratum])
        if take <= 0:
            continue
        selected.extend(rng.choice(members, size=take, replace=False).tolist())
    if len(selected) != per_group:
        raise AssertionError("Stratified allocation did not reach the target size.")
    return selected


def prepare(
    *,
    dataset: str,
    abundance_rda: Path,
    reference_tarball: Path,
    out_dir: Path,
    per_group: int,
    min_prevalence: int,
    seed: int,
    rscript: str,
) -> dict:
    spec = DATASETS[dataset]
    if not abundance_rda.is_file():
        raise FileNotFoundError(abundance_rda)
    if not reference_tarball.is_file():
        raise FileNotFoundError(reference_tarball)

    members = [
        "curatedMetagenomicData/data/sampleMetadata.rda",
        "curatedMetagenomicData/R/sysdata.rda",
    ]
    with tempfile.TemporaryDirectory(prefix=f"cmd_{dataset}_") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(reference_tarball) as tar:
            tar.extractall(tmp_path, members=members)
        metadata_rda = tmp_path / members[0]
        sysdata_rda = tmp_path / members[1]
        subprocess.run(
            [
                rscript,
                "-e",
                R_DUMP,
                str(abundance_rda),
                str(metadata_rda),
                str(sysdata_rda),
                dataset,
                str(tmp_path),
            ],
            check=True,
        )
        abundance = pd.read_csv(tmp_path / "abundance_species.csv", index_col=0)
        metadata = pd.read_csv(tmp_path / "sample_metadata.csv", dtype=str)
        tips = pd.read_csv(tmp_path / "tree_tips.csv")
        edges = pd.read_csv(tmp_path / "tree_edges.csv")
        root = int((tmp_path / "tree_root.txt").read_text().strip())

    abundance.columns = abundance.columns.astype(str)
    features = abundance.index.astype(str)
    tip_labels = set(tips["label"].astype(str))
    matched = features.isin(tip_labels)
    match_rate = float(matched.mean())
    if match_rate < MIN_MATCH_RATE:
        raise ValueError(
            f"Only {match_rate:.3f} of species rows match tree tips "
            f"(< {MIN_MATCH_RATE}); aborting instead of forcing conversion."
        )
    abundance = abundance.loc[matched].copy()

    keep_conditions = {spec["case"], spec["control"]}
    metadata = metadata[metadata["study_condition"].isin(keep_conditions)].copy()
    metadata["group_name"] = metadata["study_condition"]
    available_samples = set(abundance.columns)
    metadata = metadata[metadata["sample_id"].isin(available_samples)].copy()
    missing_meta = available_samples - set(metadata["sample_id"])
    group_sizes_available = metadata["group_name"].value_counts().to_dict()

    selected: list[str] = []
    for group_index, group_name in enumerate(sorted(keep_conditions)):
        selected.extend(
            _stratified_sample(metadata, group_name, per_group, seed + 7919 * (group_index + 1))
        )
    selected = sorted(selected)
    selected_meta = metadata[metadata["sample_id"].isin(selected)].copy()

    table = abundance[selected].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    table = table.clip(lower=0.0)
    # skbio 0.6.0 casts table values to int64 inside gemelli's fast_unifrac
    # path, which zeroes fraction-scale abundances. Store pseudo-counts
    # (fraction x 1e6, i.e. ppm) so the integer cast preserves signal.
    table = (table / 100.0 * 1e6).round()
    prevalence = (table > 0).sum(axis=1)
    table = table.loc[prevalence >= int(min_prevalence)].copy()
    if table.empty:
        raise ValueError("No features remain after prevalence filtering.")

    retained = set(table.index)
    newick, n_tree_tips = _pruned_newick(tips, edges, root, retained)
    if n_tree_tips != len(retained):
        raise AssertionError(
            f"Pruned tree has {n_tree_tips} tips but table has {len(retained)} features."
        )

    taxonomy = pd.DataFrame(
        {
            "Feature ID": table.index,
            "Taxon": [_qiime_taxonomy(feature) for feature in table.index],
            "Confidence": 1.0,
        }
    )
    group = selected_meta[["sample_id", "group_name"]].sort_values("sample_id").reset_index(drop=True)
    if set(group["sample_id"]) != set(table.columns):
        raise AssertionError("Group table and abundance table samples differ.")

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "table.csv"
    taxonomy_path = out_dir / "taxonomy.csv"
    tree_path = out_dir / "rooted-tree.nwk"
    group_path = out_dir / "group.csv"
    summary_path = out_dir / "preparation_summary.json"
    table_out = table.copy()
    table_out.index.name = "Taxon"
    table_out.to_csv(table_path)
    taxonomy.to_csv(taxonomy_path, index=False)
    tree_path.write_text(newick, encoding="utf-8")
    group.to_csv(group_path, index=False)

    outputs = [table_path, taxonomy_path, tree_path, group_path]
    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
    )

    summary = {
        "dataset": dataset,
        "comparison": spec["title"],
        "source_abundance": str(abundance_rda),
        "source_reference_tarball": str(reference_tarball),
        "source_metadata": "curatedMetagenomicData/data/sampleMetadata.rda (from tarball)",
        "source_tree": "curatedMetagenomicData/R/sysdata.rda phylogeneticTree (from tarball)",
        "feature_level": "MetaPhlAn species rows (k__..|s__) matched exactly to package tree tip labels",
        "tree_tip_match_rate": round(match_rate, 4),
        "species_rows_total": int(len(features)),
        "species_rows_matched": int(matched.sum()),
        "abundance_unit": (
            "MetaPhlAn relative abundance fraction x 1e6 (ppm pseudo-counts, rounded); "
            "skbio 0.6.0 casts table values to int64 in gemelli's fast_unifrac path"
        ),
        "min_prevalence_samples": int(min_prevalence),
        "output_features": int(len(table)),
        "samples_per_group": int(per_group),
        "samples_total": int(len(selected)),
        "groups": group["group_name"].value_counts().sort_index().to_dict(),
        "available_group_sizes": {k: int(v) for k, v in group_sizes_available.items()},
        "excluded_samples_without_group": int(len(missing_meta)),
        "selection_rule": (
            f"{per_group} samples per group, gender-stratified random sampling "
            f"proportional to stratum size (largest-remainder allocation), seed {seed}"
        ),
        "tree": (
            "curatedMetagenomicData 3.20.0 MetaPhlAn phylogenetic tree pruned to the "
            "retained species; unary nodes collapsed with branch lengths summed"
        ),
        "outputs": {
            "table": table_path.name,
            "taxonomy": taxonomy_path.name,
            "tree": tree_path.name,
            "group": group_path.name,
            "checksums": sums_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1] / "validation_datasets"
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--abundance",
        type=Path,
        default=None,
        help="Defaults to raw/curatedMetagenomicData/<dataset>/2021-10-14.<dataset>.relative_abundance.rda",
    )
    parser.add_argument(
        "--reference-tarball",
        type=Path,
        default=base
        / "raw"
        / "curatedMetagenomicData"
        / "_reference"
        / "curatedMetagenomicData_3.20.0.tar.gz",
        help="Source package tarball providing sampleMetadata.rda and R/sysdata.rda.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--min-prevalence", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--rscript", type=str, default=RSCRIPT_DEFAULT)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    base = Path(__file__).resolve().parents[1] / "validation_datasets"
    abundance = args.abundance
    if abundance is None:
        abundance = (
            base
            / "raw"
            / "curatedMetagenomicData"
            / args.dataset
            / f"2021-10-14.{args.dataset}.relative_abundance.rda"
        )
    out = args.out if args.out is not None else base / "processed" / args.dataset
    result = prepare(
        dataset=args.dataset,
        abundance_rda=abundance,
        reference_tarball=args.reference_tarball,
        out_dir=out,
        per_group=args.per_group,
        min_prevalence=args.min_prevalence,
        seed=args.seed,
        rscript=args.rscript,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
