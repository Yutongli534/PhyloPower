#!/usr/bin/env python3
"""Convert PRIDE PXD069517 processed peptide data to PhyloPower input files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RANKS = ("superkingdom", "phylum", "class", "order", "family", "genus", "species")
INVALID_TAXA = {"", "nan", "none", "root", "unclassified", "unidentified"}
KO_PATTERN = re.compile(r"\bK\d{5}\b")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _lineage_from_row(row: pd.Series) -> tuple[tuple[str, str], ...]:
    lineage: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rank in RANKS:
        value = _clean_text(row[rank])
        if value.lower() in INVALID_TAXA or value in seen:
            continue
        lineage.append((rank, value))
        seen.add(value)
    return tuple(lineage)


def _taxon_from_row(row: pd.Series) -> str:
    lca = _clean_text(row["lca"])
    if lca.lower() not in INVALID_TAXA:
        return lca
    lineage = _lineage_from_row(row)
    return lineage[-1][1] if lineage else ""


def _tip_id(taxon: str, lineage: tuple[tuple[str, str], ...]) -> str:
    key = taxon + "|" + "|".join(f"{rank}:{value}" for rank, value in lineage)
    return "taxon_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _unique_kos(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return sorted(set(KO_PATTERN.findall(str(value))))


def _serialize_newick(children: dict, *, root: bool = False) -> str:
    parts: list[str] = []
    for key in sorted(children):
        subtree = children[key]
        if key.startswith("tip:"):
            parts.append(f"{key[4:]}:1")
        else:
            if not subtree:
                continue
            parts.append(f"({_serialize_newick(subtree)}):1")
    joined = ",".join(parts)
    return joined if root else joined


def _build_tree(mapping: pd.DataFrame) -> str:
    trie: dict = {}
    for row in mapping.itertuples(index=False):
        lineage = json.loads(row.lineage_json)
        node = trie
        for rank, value in lineage:
            node = node.setdefault(f"{rank}:{value}", {})
        node[f"tip:{row.Taxon}"] = {}
    if not trie:
        raise ValueError("Cannot construct a tree without retained taxa.")
    return f"({_serialize_newick(trie, root=True)})root;\n"


def _read_sample_mapping(legend_path: Path) -> pd.DataFrame:
    legend = pd.read_excel(legend_path, sheet_name=0)
    required = {"Patient code", "Proteome Discoverer code"}
    if not required.issubset(legend.columns):
        raise ValueError(f"{legend_path} must contain {sorted(required)}")
    mapping = legend[["Patient code", "Proteome Discoverer code"]].copy()
    mapping.columns = ["sample_id", "pd_code"]
    mapping["sample_id"] = mapping["sample_id"].astype(str).str.strip()
    mapping["pd_code"] = mapping["pd_code"].astype(str).str.strip()
    if mapping.shape[0] != 28 or mapping["sample_id"].duplicated().any() or mapping["pd_code"].duplicated().any():
        raise ValueError("Expected 28 unique patient-to-Proteome-Discoverer mappings.")
    mapping["group_name"] = np.where(
        mapping["sample_id"].str.startswith("C"),
        "CD_only",
        np.where(
            mapping["sample_id"].str.startswith("M"),
            "PolyAI_CD",
            "",
        ),
    )
    if (mapping["group_name"] == "").any():
        raise ValueError("Patient codes must start with C or M.")
    counts = mapping["group_name"].value_counts().to_dict()
    if counts != {"CD_only": 14, "PolyAI_CD": 14}:
        raise ValueError(f"Expected balanced 14/14 groups, observed {counts}.")
    return mapping


def prepare(
    *,
    annotation_path: Path,
    legend_path: Path,
    out_dir: Path,
    min_prevalence: int,
    max_edges: int | None,
) -> dict:
    sample_mapping = _read_sample_mapping(legend_path)
    abundance_source = [f"Abundance F{i}" for i in range(1, 29)]
    usecols = ["lca", *RANKS, "KEGG_ko", *abundance_source]
    raw = pd.read_excel(annotation_path, sheet_name=0, usecols=usecols)
    raw["_source_row"] = np.arange(len(raw), dtype=int)

    pd_to_patient = dict(zip(sample_mapping["pd_code"], sample_mapping["sample_id"]))
    abundance_rename = {
        f"Abundance {pd_code}": sample_id for pd_code, sample_id in pd_to_patient.items()
    }
    missing_abundance = sorted(set(abundance_rename) - set(raw.columns))
    if missing_abundance:
        raise ValueError(f"Missing abundance columns: {missing_abundance}")

    raw["taxon_name"] = raw.apply(_taxon_from_row, axis=1)
    raw["lineage"] = raw.apply(_lineage_from_row, axis=1)
    raw["functions"] = raw["KEGG_ko"].apply(_unique_kos)
    selected = raw[
        raw["taxon_name"].ne("")
        & raw["lineage"].map(bool)
        & raw["functions"].map(bool)
    ].copy()
    selected["Taxon"] = [
        _tip_id(taxon, lineage)
        for taxon, lineage in zip(selected["taxon_name"], selected["lineage"])
    ]
    selected["function_count"] = selected["functions"].map(len)
    selected[abundance_source] = (
        selected[abundance_source]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    selected[abundance_source] = selected[abundance_source].div(
        selected["function_count"], axis=0
    )
    selected = selected.explode("functions", ignore_index=True).rename(
        columns={"functions": "Function", **abundance_rename}
    )
    sample_columns = sample_mapping["sample_id"].tolist()

    aggregated = (
        selected.groupby(["Taxon", "Function"], as_index=False, sort=True)[sample_columns]
        .sum()
    )
    prevalence = (aggregated[sample_columns] > 0).sum(axis=1)
    aggregated = aggregated.loc[prevalence >= int(min_prevalence)].copy()
    aggregated["_total"] = aggregated[sample_columns].sum(axis=1)
    if max_edges is not None and len(aggregated) > int(max_edges):
        aggregated = aggregated.nlargest(int(max_edges), "_total")
    aggregated = (
        aggregated.drop(columns="_total")
        .sort_values(["Taxon", "Function"])
        .reset_index(drop=True)
    )

    lineage_rows = (
        selected[["Taxon", "taxon_name", "lineage"]]
        .drop_duplicates("Taxon")
        .copy()
    )
    retained_taxa = set(aggregated["Taxon"])
    lineage_rows = lineage_rows[lineage_rows["Taxon"].isin(retained_taxa)].copy()
    lineage_rows["lineage_json"] = lineage_rows["lineage"].map(json.dumps)
    mapping = (
        lineage_rows[["Taxon", "taxon_name", "lineage_json"]]
        .sort_values("Taxon")
        .reset_index(drop=True)
    )
    tree_text = _build_tree(mapping)

    if aggregated.duplicated(["Taxon", "Function"]).any():
        raise AssertionError("Duplicate Taxon–Function edges remain after aggregation.")
    values = aggregated[sample_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise AssertionError("Output abundances must be finite and non-negative.")
    if set(mapping["Taxon"]) != set(aggregated["Taxon"]):
        raise AssertionError("Tree mapping and abundance-table taxa are not identical.")

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "protein_taxon_function.csv"
    group_path = out_dir / "group.csv"
    tree_path = out_dir / "rooted-tree.nwk"
    mapping_path = out_dir / "taxon_mapping.csv"
    summary_path = out_dir / "preparation_summary.json"
    aggregated.to_csv(table_path, index=False)
    sample_mapping[["sample_id", "group_name"]].to_csv(group_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    tree_path.write_text(tree_text, encoding="utf-8")

    summary = {
        "accession": "PXD069517",
        "source_annotation": str(annotation_path),
        "source_legend": str(legend_path),
        "taxon_definition": "LCA label (or deepest available rank), disambiguated by full annotated lineage",
        "function_definition": "KEGG orthology identifier",
        "multiple_function_rule": "Peptide intensity divided equally among unique KEGG KO annotations",
        "min_prevalence_samples": int(min_prevalence),
        "max_taxon_function_edges": max_edges,
        "input_peptides": int(len(raw)),
        "annotated_peptides": int(selected["_source_row"].nunique()),
        "output_taxa": int(aggregated["Taxon"].nunique()),
        "output_functions": int(aggregated["Function"].nunique()),
        "output_taxon_function_edges": int(len(aggregated)),
        "samples": int(len(sample_columns)),
        "groups": sample_mapping["group_name"].value_counts().sort_index().to_dict(),
        "tree": "Rooted taxonomic hierarchy from the supplied superkingdom-to-species annotations; unit branch lengths",
        "outputs": {
            "table": table_path.name,
            "group": group_path.name,
            "tree": tree_path.name,
            "taxon_mapping": mapping_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1] / "validation_datasets"
    raw = base / "raw" / "PRIDE" / "PXD069517"
    parser.add_argument(
        "--annotation",
        type=Path,
        default=raw / "MGDB_peptides_intensities_tax-funct_annotations.xlsx",
    )
    parser.add_argument(
        "--legend",
        type=Path,
        default=raw / "sample_code_legend.xlsx",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=base / "processed" / "PXD069517",
    )
    parser.add_argument("--min-prevalence", type=int, default=2)
    parser.add_argument(
        "--max-edges",
        type=int,
        default=10000,
        help="Keep the most abundant edges after prevalence filtering; use 0 for no cap.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = prepare(
        annotation_path=args.annotation,
        legend_path=args.legend,
        out_dir=args.out,
        min_prevalence=args.min_prevalence,
        max_edges=None if args.max_edges == 0 else args.max_edges,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
