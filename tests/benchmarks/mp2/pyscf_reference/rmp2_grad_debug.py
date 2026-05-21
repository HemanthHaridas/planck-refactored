#!/usr/bin/env python3
"""Entry point for the RMP2 gradient debugging toolkit.

Thin wrapper so the package can be run without the long ``-m`` path:

    python tests/benchmarks/mp2/pyscf_reference/rmp2_grad_debug.py all
    python tests/benchmarks/mp2/pyscf_reference/rmp2_grad_debug.py cphf --case water_rmp2_gradient_sto3g

All logic lives in the ``rmp2_grad`` package next to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when invoked as a plain script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from rmp2_grad.compare import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
