#!/usr/bin/env python3
"""Extract Planck's imat matrices."""
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]

case_inputs = {
    "water_rmp2_gradient_sto3g": REPO_ROOT / "tests" / "inputs" / "regression" / "post_hf" / "water_rmp2_gradient_sto3g.hfinp"
}

def extract_debug_matrices():
    input_path = case_inputs["water_rmp2_gradient_sto3g"]
    executable = REPO_ROOT / "build" / "hartree-fock"

    env = dict(os.environ)
    env["PLANCK_DEBUG_RMP2_IMAT"] = "1"
    proc = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    IMAT_HEADER_RE = re.compile(r"^PLANCK_DEBUG_IMAT_MO (\d+) (\d+)\s*$")
    IMAT_ELEM_RE = re.compile(r"^PLANCK_DEBUG_IMAT_MO_ELEM (\d+) (\d+) ([-+0-9Ee\.]+)\s*$")
    BLOCK_TR_RE = re.compile(r"^PLANCK_DEBUG_IMAT_TOP_RIGHT (\d+) (\d+)\s*$")
    BLOCK_TR_ELEM_RE = re.compile(r"^PLANCK_DEBUG_IMAT_TOP_RIGHT_ELEM (\d+) (\d+) ([-+0-9Ee\.]+)\s*$")
    BLOCK_BL_RE = re.compile(r"^PLANCK_DEBUG_IMAT_BOTTOM_LEFT (\d+) (\d+)\s*$")
    BLOCK_BL_ELEM_RE = re.compile(r"^PLANCK_DEBUG_IMAT_BOTTOM_LEFT_ELEM (\d+) (\d+) ([-+0-9Ee\.]+)\s*$")

    import numpy as np

    imat_mo = None
    imat_top_right = None
    imat_bottom_left = None

    for line in proc.stdout.splitlines():
        header_match = IMAT_HEADER_RE.match(line.strip())
        if header_match:
            rows, cols = int(header_match.group(1)), int(header_match.group(2))
            imat_mo = np.zeros((rows, cols))
            continue

        elem_match = IMAT_ELEM_RE.match(line.strip())
        if elem_match and imat_mo is not None:
            i, j, val = int(elem_match.group(1)), int(elem_match.group(2)), float(elem_match.group(3))
            imat_mo[i, j] = val
            continue

        block_tr_match = BLOCK_TR_RE.match(line.strip())
        if block_tr_match:
            rows, cols = int(block_tr_match.group(1)), int(block_tr_match.group(2))
            imat_top_right = np.zeros((rows, cols))
            continue

        block_tr_elem = BLOCK_TR_ELEM_RE.match(line.strip())
        if block_tr_elem and imat_top_right is not None:
            i, j, val = int(block_tr_elem.group(1)), int(block_tr_elem.group(2)), float(block_tr_elem.group(3))
            imat_top_right[i, j] = val
            continue

        block_bl_match = BLOCK_BL_RE.match(line.strip())
        if block_bl_match:
            rows, cols = int(block_bl_match.group(1)), int(block_bl_match.group(2))
            imat_bottom_left = np.zeros((rows, cols))
            continue

        block_bl_elem = BLOCK_BL_ELEM_RE.match(line.strip())
        if block_bl_elem and imat_bottom_left is not None:
            i, j, val = int(block_bl_elem.group(1)), int(block_bl_elem.group(2)), float(block_bl_elem.group(3))
            imat_bottom_left[i, j] = val
            continue

    return {
        "imat_mo": imat_mo,
        "imat_top_right": imat_top_right,
        "imat_bottom_left": imat_bottom_left,
    }

if __name__ == "__main__":
    result = extract_debug_matrices()
    print("=" * 80)
    print("Planck imat_mo:")
    if result["imat_mo"] is not None:
        print(f"shape: {result['imat_mo'].shape}")
        print(result["imat_mo"])
    print("\n" + "=" * 80)
    print("Planck imat_mo[:nocc, nocc:] (top right):")
    if result["imat_top_right"] is not None:
        print(f"shape: {result['imat_top_right'].shape}")
        print(result["imat_top_right"])
    print("\n" + "=" * 80)
    print("Planck imat_mo[nocc:, :nocc] (bottom left):")
    if result["imat_bottom_left"] is not None:
        print(f"shape: {result['imat_bottom_left'].shape}")
        print(result["imat_bottom_left"])
