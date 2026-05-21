"""PySCF runtime bootstrap and shared repository paths.

The repo ships a vendored PySCF in ``tests/pyscf/.venv``. When the current
interpreter cannot import PySCF, :func:`ensure_pyscf_runtime` transparently
re-execs the *calling* script under that venv's interpreter so the tools work
regardless of which ``python`` the user invokes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# rmp2_grad/_runtime.py -> pyscf_reference -> mp2 -> benchmarks -> tests -> repo
REPO_ROOT = Path(__file__).resolve().parents[5]
PYSCF_DIR = REPO_ROOT / "tests" / "pyscf"
PYSCF_REFERENCE_DIR = REPO_ROOT / "tests" / "benchmarks" / "mp2" / "pyscf_reference"
LOCAL_PYSCF_PYTHON = PYSCF_DIR / ".venv" / "bin" / "python"

# ``benchmark`` and ``input_utils`` live alongside the package / under tests/pyscf.
for _path in (PYSCF_REFERENCE_DIR, PYSCF_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def ensure_pyscf_runtime(entry: str | Path | None = None) -> None:
    """Import PySCF, re-execing into the vendored venv if necessary.

    Parameters
    ----------
    entry:
        Script to re-exec. Defaults to the process entry point (``sys.argv[0]``)
        which is what a CLI wrapper wants.
    """

    try:
        import pyscf  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    current = Path(sys.executable).resolve()
    target = Path(entry) if entry is not None else Path(sys.argv[0]).resolve()
    if LOCAL_PYSCF_PYTHON.exists() and current != LOCAL_PYSCF_PYTHON.resolve():
        proc = subprocess.run(
            [str(LOCAL_PYSCF_PYTHON), str(target), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(proc.returncode)
    raise SystemExit(
        "PySCF is not importable and the vendored interpreter was not found at "
        f"{LOCAL_PYSCF_PYTHON}. Set up tests/pyscf/.venv or run under a PySCF env."
    )


def default_executable(build_dir: Path | None = None) -> Path:
    """Path to the ``hartree-fock`` binary under ``build_dir`` (default ``build``)."""

    build = build_dir if build_dir is not None else REPO_ROOT / "build"
    return build / "hartree-fock"
