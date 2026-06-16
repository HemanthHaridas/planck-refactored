#!/usr/bin/env python3
"""Generate Planck-style spin-orbital RCCSD warm-start kernels."""

from __future__ import annotations

import argparse
from pathlib import Path

from ccgen.emit.planck_rccsd_warm_start import (
    emit_planck_spinorbital_rccsd_warm_start,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Planck-style spin-orbital RCCSD warm-start kernels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the generated .inc file.",
    )
    args = parser.parse_args()

    code = emit_planck_spinorbital_rccsd_warm_start()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(code + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
