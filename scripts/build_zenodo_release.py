#!/usr/bin/env python3
"""Build a clean, deterministic source archive for manual Zenodo upload."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
ARCHIVE_ROOT = f"phylopower-{VERSION}"
RELEASE_DIR = PROJECT_ROOT / "release"
ARCHIVE_PATH = RELEASE_DIR / f"{ARCHIVE_ROOT}.zip"

# Regenerable run outputs are excluded from the archive: every number in the
# manuscript can be regenerated from the bundled inputs with the bundled
# scripts, and the full evidence tree stays browsable on GitHub. Excluding
# them keeps the citable archive lean (code + inputs + scripts).
EXCLUDE_DIRS = (
    # archived run outputs are regenerable via figures/*.py --compute, EXCEPT
    # the tree_error_fig5_like* curve CSVs, which have no compute path
    # (kept below so Fig 8 stays reproducible from the archive alone)
    "data/archived_runs/_fig5data_core_gene_omega003_power08/",
    "data/archived_runs/_fig5data_core_gene_omega003_power08_max80/",
    "data/archived_runs/_fig5data_core_protein_omega003_power08/",
    "data/archived_runs/fig2_fidelity_audit/",
    "data/archived_runs/fig4_new/",
    "data/archived_runs/fig5_rerun_20260701/",
    "data/gene_min_sample_size_output/",
    "validation_datasets/results/",
    # full-cohort conversions only fed the dropped end-to-end experiment;
    # the subsampling-truth numbers they produced are recorded in
    # figures/output/suppfig2_source_data.csv. Full trees stay on GitHub.
    "validation_datasets/processed/QinJ_2012_full/",
    "validation_datasets/processed/YachidaS_2019_full/",
)

# figures/output/ keeps only hand-made (non-regenerable) assets in the
# archive; every scripted figure can be regenerated with figures/*.py
KEEP_OUTPUT = (
    "figures/output/figure1_workflow.png",
    "figures/output/Figure2_corrected.png",
    "figures/output/Figure5_corrected.png",
    "figures/output/Figure8_corrected.png",
    "figures/output/figure2_manuscript_panels/",
)

# regenerable image outputs that happen to live inside data directories
EXCLUDE_FILES = {
    "data/figdata/protein_mdctf_curve/comparison.png",
    "data/pilot_information_supplement/pilot_information_supplement.png",
    "data/pilot_information_supplement/pilot_information_supplement.pdf",
}


def release_files() -> list[str]:
    """The release archive mirrors the curated public tree (every git-tracked
    file; scratch and local-only material is gitignored), minus submission-only
    TIFF rasters and regenerable run outputs (see EXCLUDE_DIRS)."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return [
        f
        for f in out.stdout.split()
        if not f.lower().endswith((".tiff", ".tif"))
        and not (
            f.startswith("data/archived_runs/")
            and f.lower().endswith((".png", ".pdf"))
        )
        and not any(f.startswith(d) for d in EXCLUDE_DIRS)
        and f not in EXCLUDE_FILES
        and not (
            f.startswith("figures/output/")
            and not any(f.startswith(k) for k in KEEP_OUTPUT)
        )
    ]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(2026, 7, 24, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def main() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/build_standalone.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )

    relative_files = release_files()
    missing = [name for name in relative_files if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing release files: " + ", ".join(missing))

    payloads = {
        name: (PROJECT_ROOT / name).read_bytes()
        for name in sorted(relative_files)
    }
    manifest = "\n".join(
        f"{sha256_bytes(data)}  {name}"
        for name, data in payloads.items()
    ) + "\n"
    payloads["FILE_MANIFEST.sha256"] = manifest.encode("utf-8")

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(archive_info(name), data)

    archive_hash = sha256_bytes(ARCHIVE_PATH.read_bytes())
    checksum_path = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".sha256")
    checksum_path.write_text(
        f"{archive_hash}  {ARCHIVE_PATH.name}\n",
        encoding="utf-8",
    )
    print(f"Wrote {ARCHIVE_PATH}")
    print(f"SHA-256 {archive_hash}")


if __name__ == "__main__":
    main()
