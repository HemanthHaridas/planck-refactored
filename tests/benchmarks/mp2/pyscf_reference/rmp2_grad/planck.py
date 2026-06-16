"""Run the ``hartree-fock`` binary and parse its RMP2-gradient debug dumps.

The gradient code emits machine-parseable blocks under a handful of
``PLANCK_DEBUG_*`` environment switches:

==========================  =======================  =========================
env var                     line prefix              what it carries
==========================  =======================  =========================
PLANCK_DEBUG_RHF_RESPONSE   PLANCK_RHF_RESPONSE      CPHF matrix A, rhs, z
PLANCK_DEBUG_RMP2_MATRICES  PLANCK_RMP2_MATRIX       z, corr_relaxed_mo, P_ao,
                                                     dm1_corr_relaxed_ao, dm1p
PLANCK_DEBUG_RMP2_TERMS     PLANCK_RMP2_TERM_ROW     per-atom gradient terms
PLANCK_DEBUG_RMP2_IMAT      PLANCK_DEBUG_IMAT_MO     imat_mo and its occ-virt
                            (+_TOP_RIGHT/_BOTTOM_LEFT)  / virt-occ blocks
==========================  =======================  =========================

This module turns each block into ``dict[str, np.ndarray]`` keyed by the dump
name, plus a parser for the final ``Nuclear Gradient`` table.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

# 2D matrix dumps: "<PREFIX> <name> <rows> <cols>" then "<PREFIX>_ELEM <name> i j val".
_MATRIX_HEADER_RE = re.compile(r"^(PLANCK_\w+?)\s+(\S+)\s+(\d+)\s+(\d+)\s*$")
_MATRIX_ELEM_RE = re.compile(r"^(PLANCK_\w+?)_ELEM\s+(\S+)\s+(\d+)\s+(\d+)\s+([-+0-9Ee.]+)\s*$")
# Per-atom term rows: "PLANCK_RMP2_TERM_ROW <name> <atom> gx gy gz".
_TERM_ROW_RE = re.compile(
    r"^PLANCK_RMP2_TERM_ROW\s+(\S+)\s+(\d+)\s+([-+0-9Ee.]+)\s+([-+0-9Ee.]+)\s+([-+0-9Ee.]+)\s*$"
)
# Final gradient table: "Atom   1:   gx  gy  gz".
_GRAD_LINE_RE = re.compile(
    r"Atom\s+(\d+)\s*:\s*([-+0-9Ee.]+)\s+([-+0-9Ee.]+)\s+([-+0-9Ee.]+)"
)
# imat dumps each use a distinct prefix (the name *is* the prefix, no name field):
#   "PLANCK_DEBUG_IMAT_MO rows cols" then "PLANCK_DEBUG_IMAT_MO_ELEM i j val".
_IMAT_BLOCKS = {
    "imat_mo": "PLANCK_DEBUG_IMAT_MO",
    "imat_top_right": "PLANCK_DEBUG_IMAT_TOP_RIGHT",
    "imat_bottom_left": "PLANCK_DEBUG_IMAT_BOTTOM_LEFT",
}


def run_planck(executable: Path, input_path: Path, debug_vars: list[str]) -> str:
    """Run ``hartree-fock`` with the given ``PLANCK_DEBUG_*`` flags set to "1"."""

    env = dict(os.environ)
    for var in debug_vars:
        env[var] = "1"
    proc = subprocess.run(
        [str(executable), str(input_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Planck run failed for {input_path} (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def parse_matrices(output: str, prefix: str) -> dict[str, np.ndarray]:
    """Parse ``<prefix> name rows cols`` / ``<prefix>_ELEM name i j val`` blocks."""

    mats: dict[str, np.ndarray] = {}
    for raw in output.splitlines():
        line = raw.strip()
        header = _MATRIX_HEADER_RE.match(line)
        if header and header.group(1) == prefix:
            name = header.group(2)
            mats[name] = np.zeros((int(header.group(3)), int(header.group(4))), dtype=float)
            continue
        elem = _MATRIX_ELEM_RE.match(line)
        if elem and elem.group(1) == prefix:
            name = elem.group(2)
            if name in mats:
                mats[name][int(elem.group(3)), int(elem.group(4))] = float(elem.group(5))
    return mats


def parse_terms(output: str) -> dict[str, np.ndarray]:
    """Parse per-atom ``PLANCK_RMP2_TERM_ROW`` rows into ``name -> (natom, 3)``."""

    rows: dict[str, dict[int, list[float]]] = {}
    for raw in output.splitlines():
        match = _TERM_ROW_RE.match(raw.strip())
        if not match:
            continue
        name = match.group(1)
        atom = int(match.group(2)) - 1  # Planck prints 1-based atom indices.
        rows.setdefault(name, {})[atom] = [float(match.group(i)) for i in (3, 4, 5)]

    terms: dict[str, np.ndarray] = {}
    for name, by_atom in rows.items():
        natom = max(by_atom) + 1
        arr = np.zeros((natom, 3))
        for atom, vec in by_atom.items():
            arr[atom] = vec
        terms[name] = arr
    return terms


def parse_imat_blocks(output: str) -> dict[str, np.ndarray]:
    """Parse the ``PLANCK_DEBUG_IMAT_*`` blocks (imat_mo + occ-virt/virt-occ)."""

    # Match longest prefixes first so IMAT_MO doesn't shadow IMAT_MO_ELEM etc.
    blocks = sorted(_IMAT_BLOCKS.items(), key=lambda kv: len(kv[1]), reverse=True)
    mats: dict[str, np.ndarray] = {}
    for raw in output.splitlines():
        line = raw.strip()
        for name, prefix in blocks:
            elem = re.match(rf"^{prefix}_ELEM\s+(\d+)\s+(\d+)\s+([-+0-9Ee.]+)\s*$", line)
            if elem:
                if name in mats:
                    mats[name][int(elem.group(1)), int(elem.group(2))] = float(elem.group(3))
                break
            header = re.match(rf"^{prefix}\s+(\d+)\s+(\d+)\s*$", line)
            if header:
                mats[name] = np.zeros((int(header.group(1)), int(header.group(2))), dtype=float)
                break
    return mats


def parse_gradient(output: str) -> np.ndarray:
    """Parse the final ``Nuclear Gradient`` table into an ``(natom, 3)`` array."""

    rows = [
        [float(m.group(2)), float(m.group(3)), float(m.group(4))]
        for m in (_GRAD_LINE_RE.search(line) for line in output.splitlines())
        if m
    ]
    if not rows:
        raise RuntimeError("No 'Atom N:' gradient lines found in Planck output.")
    return np.array(rows)


# --- convenience wrappers: one call per debug surface ---------------------

def cphf_matrices(executable: Path, input_path: Path) -> dict[str, np.ndarray]:
    """A, rhs, z from the CPHF solver."""

    out = run_planck(executable, input_path, ["PLANCK_DEBUG_RHF_RESPONSE"])
    mats = parse_matrices(out, "PLANCK_RHF_RESPONSE")
    if not mats:
        raise RuntimeError("No PLANCK_RHF_RESPONSE rows found (is the binary current?).")
    return mats


def response_chain(executable: Path, input_path: Path) -> dict[str, np.ndarray]:
    """z, corr_relaxed_mo, P_ao, dm1_corr_relaxed_ao, dm1p."""

    out = run_planck(executable, input_path, ["PLANCK_DEBUG_RMP2_MATRICES"])
    mats = parse_matrices(out, "PLANCK_RMP2_MATRIX")
    if not mats:
        raise RuntimeError("No PLANCK_RMP2_MATRIX rows found (is the binary current?).")
    return mats


def imat_blocks(executable: Path, input_path: Path) -> dict[str, np.ndarray]:
    """imat_mo and its occ-virt (top-right) / virt-occ (bottom-left) blocks."""

    out = run_planck(executable, input_path, ["PLANCK_DEBUG_RMP2_IMAT"])
    mats = parse_imat_blocks(out)
    if not mats:
        raise RuntimeError("No PLANCK_DEBUG_IMAT_* rows found (is the binary current?).")
    return mats


def gradient_terms(executable: Path, input_path: Path) -> dict[str, np.ndarray]:
    """Per-atom gradient term decomposition (two_e, h1, vhf1, s_*, electronic)."""

    out = run_planck(executable, input_path, ["PLANCK_DEBUG_RMP2_TERMS"])
    terms = parse_terms(out)
    if not terms:
        raise RuntimeError("No PLANCK_RMP2_TERM_ROW rows found (is the binary current?).")
    return terms


def total_gradient(executable: Path, input_path: Path) -> np.ndarray:
    """Final analytic nuclear gradient, ``(natom, 3)``."""

    return parse_gradient(run_planck(executable, input_path, []))
