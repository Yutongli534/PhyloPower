"""
Helpers for locating demo datasets bundled inside the ``phylopower`` package.

The datasets under ``phylopower/datagene/`` and ``phylopower/datapro/`` are
shipped as package-data, so ``pip install phylopower`` makes them available
on the filesystem inside ``site-packages``. Because that path depends on
the user's Python install, we use :mod:`importlib.resources` to resolve it
at runtime.

Example
-------
>>> from phylopower import demo_path
>>> demo_path("datagene", "table.csv")
PosixPath('.../site-packages/phylopower/datagene/table.csv')
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["demo_path", "DATAGENE_DIR", "DATAPRO_DIR"]


def demo_path(dataset: str, filename: str | None = None) -> Path:
    """Return the filesystem path of a bundled demo file or directory.

    Parameters
    ----------
    dataset : {"datagene", "datapro"}
        Which bundled dataset to look up.
    filename : str, optional
        File inside the dataset. If omitted, the dataset directory itself
        is returned.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested resource.
    """
    if dataset not in {"datagene", "datapro"}:
        raise ValueError(
            f"dataset must be 'datagene' or 'datapro', got {dataset!r}")

    base = resources.files(__package__).joinpath(dataset)
    target = base if filename is None else base.joinpath(filename)

    # ``resources.files`` returns a ``Traversable``; for a real on-disk
    # install this is already a concrete path, but we coerce to Path for
    # a predictable return type.
    return Path(str(target))


#: Directory containing the bundled metagenomic demo dataset.
DATAGENE_DIR: Path = demo_path("datagene")

#: Directory containing the bundled metaproteomic demo dataset.
DATAPRO_DIR: Path = demo_path("datapro")
