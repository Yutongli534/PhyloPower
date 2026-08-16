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

ROOT_FILES = [
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "REPRODUCIBILITY.md",
    "ZENODO_UPLOAD.md",
    "phylopower_cli.py",
    "pyproject.toml",
    "analysis/_fig4_curve_plotting.py",
    "analysis/_protein_mdctf_curve.py",
    "analysis/_protein_mdctf_mc.py",
    "analysis/_protein_mdctf_optimized_curve.py",
    "analysis/gene_power_workflow.py",
    "analysis/logistic_fit.py",
    "analysis/pcam_gen.py",
    "analysis/phylofunc_fast.py",
    "analysis/protein_power_workflow.py",
    "analysis/protein_transforms.py",
    "analysis/semisynthetic_power.py",
]

PACKAGE_FILES = [
    "phylopower/__init__.py",
    "phylopower/__main__.py",
    "phylopower/_core_source.py",
    "phylopower/_data.py",
    "phylopower/_cli_source.py",
    "phylopower/cli.py",
    "phylopower/datagene/group.csv",
    "phylopower/datagene/rooted-tree.nwk",
    "phylopower/datagene/table.csv",
    "phylopower/datagene/taxonomy.csv",
    "phylopower/datapro/group.csv",
    "phylopower/datapro/rooted-tree.nwk",
    "phylopower/datapro/protein_taxon_function_cleaned.csv",
]

SUPPORT_FILES = [
    "scripts/build_standalone.py",
    "scripts/build_zenodo_release.py",
    "tests/test_cli_release.py",
    "tests/test_reproducibility.py",
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

    relative_files = ROOT_FILES + PACKAGE_FILES + SUPPORT_FILES
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
