#!/usr/bin/env python3
"""Build the single-file, manuscript-aligned PhyloPower executable.

The generated file embeds every project-local module needed by
``phylopower.cli`` and loads them directly from memory.  It therefore
needs only its third-party Python environment and the input datasets at run
time; no other PhyloPower ``.py`` files are required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    PROJECT_ROOT / "phylopower_cli.py",
    PROJECT_ROOT / "phylopower" / "cli.py",
]

MODULE_FILES = {
    "phylopower.core": "phylopower/_core_source.py",
    "phylopower.cli": "phylopower/_cli_source.py",
    "_fig4_curve_plotting": "analysis/_fig4_curve_plotting.py",
    "_protein_mdctf_curve": "analysis/_protein_mdctf_curve.py",
    "_protein_mdctf_mc": "analysis/_protein_mdctf_mc.py",
    "_protein_mdctf_optimized_curve": "analysis/_protein_mdctf_optimized_curve.py",
    "gene_power_workflow": "analysis/gene_power_workflow.py",
    "logistic_fit": "analysis/logistic_fit.py",
    "pcam_gen": "analysis/pcam_gen.py",
    "phylofunc_fast": "analysis/phylofunc_fast.py",
    "protein_power_workflow": "analysis/protein_power_workflow.py",
    "protein_transforms": "analysis/protein_transforms.py",
    "semisynthetic_power": "analysis/semisynthetic_power.py",
}

DATA_MODULE_SOURCE = r'''
"""Dataset-path helpers for the single-file PhyloPower distribution."""
from __future__ import annotations

import os
from pathlib import Path

_BUNDLE_DIR = Path(__file__).resolve().parent
if "PHYLOPOWER_DATA_DIR" in os.environ:
    _DATA_ROOT = Path(os.environ["PHYLOPOWER_DATA_DIR"]).expanduser().resolve()
elif (_BUNDLE_DIR / "datagene").is_dir() or (_BUNDLE_DIR / "datapro").is_dir():
    _DATA_ROOT = _BUNDLE_DIR
else:
    # Source/Zenodo layout keeps examples inside the Python package.
    _DATA_ROOT = _BUNDLE_DIR / "phylopower"
DATAGENE_DIR = _DATA_ROOT / "datagene"
DATAPRO_DIR = _DATA_ROOT / "datapro"


def demo_path(dataset: str, filename: str | None = None) -> Path:
    if dataset not in {"datagene", "datapro"}:
        raise ValueError(
            f"dataset must be 'datagene' or 'datapro', got {dataset!r}"
        )
    base = DATAGENE_DIR if dataset == "datagene" else DATAPRO_DIR
    return base if filename is None else base / filename
'''.lstrip()

HEADER = '''\
#!/usr/bin/env python3
"""PhyloPower single-file runner generated from the manuscript-aligned source.

This file embeds all project-local Python modules.  Third-party scientific
packages and input datasets remain external requirements.  Put ``datagene``
and/or ``datapro`` next to this file, set ``PHYLOPOWER_DATA_DIR``, or provide
all input paths explicitly on the command line.

The same generated file can be executed directly or imported as
``phylopower.cli``.

Regenerate with: ``python scripts/build_standalone.py``
"""

from __future__ import annotations

import base64 as _base64
import importlib.abc as _importlib_abc
import importlib.util as _importlib_util
import json as _json
import sys as _sys
import types as _types
import zlib as _zlib
from pathlib import Path as _Path

'''

LOADER = r'''

class _EmbeddedModuleLoader(_importlib_abc.MetaPathFinder, _importlib_abc.Loader):
    """Import the bundled PhyloPower modules without writing temporary files."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _EMBEDDED_SOURCES:
            return _importlib_util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        encoded = _EMBEDDED_SOURCES[module.__name__]
        source = _zlib.decompress(_base64.b85decode(encoded)).decode("utf-8")
        module.__file__ = str(_Path(__file__).resolve())
        module.__package__ = module.__name__.rpartition(".")[0]
        exec(compile(source, f"{__file__}::{module.__name__}", "exec"), module.__dict__)


def _install_embedded_finder():
    if not any(isinstance(item, _EmbeddedModuleLoader) for item in _sys.meta_path):
        _sys.meta_path.insert(0, _EmbeddedModuleLoader())


def _install_embedded_modules():
    package = _types.ModuleType("phylopower")
    package.__file__ = str(_Path(__file__).resolve())
    package.__package__ = "phylopower"
    package.__path__ = []
    _sys.modules["phylopower"] = package
    _install_embedded_finder()


def _exec_embedded_cli_in_current_module():
    encoded = _EMBEDDED_SOURCES["phylopower.cli"]
    source = _zlib.decompress(_base64.b85decode(encoded)).decode("utf-8")
    globals()["__package__"] = "phylopower"
    exec(
        compile(source, f"{__file__}::phylopower.cli", "exec"),
        globals(),
    )


def _print_standalone_info():
    print(_json.dumps(_BUILD_INFO, indent=2, sort_keys=True))


if __name__ == "phylopower.cli":
    # Package import: keep the real parent package, install the embedded
    # dependency finder, and expose the public API from the embedded source
    # through this generated module.
    _install_embedded_finder()
    _exec_embedded_cli_in_current_module()
elif __name__ == "__main__":
    if "--standalone-info" in _sys.argv:
        _print_standalone_info()
    else:
        _install_embedded_modules()
        from phylopower.cli import main as _cli_main

        _cli_main()
else:
    # Importing a copied standalone under another module name remains
    # supported and re-exports the manuscript-aligned public functions.
    _install_embedded_modules()
    from phylopower.cli import main as _cli_main
    from phylopower.cli import (
        compute_gene_min_sample_size,
        compute_protein_min_sample_size,
        create_argument_parser,
        main,
    )
'''


def _encode(source: str) -> str:
    compressed = zlib.compress(source.encode("utf-8"), level=9)
    return base64.b85encode(compressed).decode("ascii")


def main() -> None:
    sources = {"phylopower._data": DATA_MODULE_SOURCE}
    for module_name, relative_path in MODULE_FILES.items():
        sources[module_name] = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

    hashes = {
        module_name: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for module_name, source in sorted(sources.items())
    }
    build_info = {
        "entry_point": "phylopower.cli:main",
        "embedded_module_count": len(sources),
        "module_sha256": hashes,
    }
    encoded_sources = {
        module_name: _encode(source)
        for module_name, source in sorted(sources.items())
    }

    payload = (
        HEADER
        + "_EMBEDDED_SOURCES = "
        + json.dumps(encoded_sources, indent=2, sort_keys=True)
        + "\n\n_BUILD_INFO = "
        + json.dumps(build_info, indent=2, sort_keys=True)
        + "\n"
        + textwrap.dedent(LOADER)
    )
    for output in OUTPUTS:
        output.write_text(payload, encoding="utf-8")
        output.chmod(0o755)
        print(f"Wrote {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
