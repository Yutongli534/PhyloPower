from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import phylopower
from phylopower import cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_api_uses_cli() -> None:
    assert phylopower.compute_gene_min_sample_size is cli.compute_gene_min_sample_size
    assert phylopower.compute_protein_min_sample_size is cli.compute_protein_min_sample_size
    assert phylopower.compute_taxon is cli.compute_gene_min_sample_size
    assert phylopower.compute_taxon_function is cli.compute_protein_min_sample_size


def test_cli_exposes_manuscript_workflows() -> None:
    parser = cli.create_argument_parser()
    gene = parser.parse_args(["gene", "--target-power", "0.8", "--out", "out"])
    protein = parser.parse_args(["protein", "--target-power", "0.8", "--out", "out"])
    assert gene.workflow == "gene"
    assert gene.pcam_gene_blocks == "auto"
    assert gene.target_omega2 is None
    assert protein.workflow == "protein"
    assert protein.edge_fraction == "auto"
    assert protein.target_omega2 is None


def test_standalone_contains_complete_dependency_closure() -> None:
    cli_runner = PROJECT_ROOT / "phylopower_cli.py"
    package_runner = PROJECT_ROOT / "phylopower" / "cli.py"
    assert cli_runner.read_bytes() == package_runner.read_bytes()
    result = subprocess.run(
        [sys.executable, str(cli_runner), "--standalone-info"],
        check=True,
        capture_output=True,
        text=True,
        cwd=cli_runner.parent,
    )
    info = json.loads(result.stdout)
    assert info["entry_point"] == "phylopower.cli:main"
    assert info["embedded_module_count"] == 14
    assert "phylopower.core" in info["module_sha256"]
    assert "phylopower.cli" in info["module_sha256"]


def test_package_cli_runs_without_other_project_code() -> None:
    package_runner = PROJECT_ROOT / "phylopower" / "cli.py"
    with tempfile.TemporaryDirectory() as tmp:
        copied_runner = Path(tmp) / "phylopower_cli.py"
        copied_runner.write_bytes(package_runner.read_bytes())
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        result = subprocess.run(
            [sys.executable, str(copied_runner), "--help"],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp,
            env=env,
        )
    assert "Minimum sample-size estimator using raw-pool workflows." in result.stdout


def test_release_metadata_is_valid_json() -> None:
    metadata = json.loads((PROJECT_ROOT / ".zenodo.json").read_text())
    assert metadata["upload_type"] == "software"
    assert metadata["version"] == phylopower.__version__
