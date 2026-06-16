"""CLI: diff Planck's RMP2 gradient intermediates against PySCF, stage by stage.

Subcommands map to pipeline stages:

  cphf      A (CPHF matrix), rhs, z          [PLANCK_DEBUG_RHF_RESPONSE]
  imat      imat_mo + occ-virt/virt-occ      [PLANCK_DEBUG_RMP2_IMAT]
  chain     z, corr_relaxed_mo, P_ao, ...    [PLANCK_DEBUG_RMP2_MATRICES]
  terms     per-atom gradient terms          [PLANCK_DEBUG_RMP2_TERMS]
  gradient  final analytic nuclear gradient
  all       run every stage above

Element-wise diffs include both signed and phase-invariant (|.|) max errors:
the MP2 gradient is invariant to MO-orbital sign choices, so a large *signed*
diff with a tiny *abs* diff is just a phase artifact, not a bug. See
``RMP2_GRADIENT_FIX_SUMMARY.md`` for how that distinction drove the fix.

Run via the wrapper (handles the vendored PySCF venv automatically):

    python tests/benchmarks/mp2/pyscf_reference/rmp2_grad_debug.py <stage> [--case ...]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ._runtime import default_executable, ensure_pyscf_runtime

ensure_pyscf_runtime()

import numpy as np  # noqa: E402

from benchmark import CASE_INPUTS  # noqa: E402

from . import planck, reference  # noqa: E402


def summarize_diff(name: str, lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Print signed + phase-invariant diff stats; return the signed max-abs error."""

    if lhs.shape != rhs.shape:
        print(f"{name}: SHAPE MISMATCH planck={lhs.shape} pyscf={rhs.shape}")
        return float("inf")
    diff = lhs - rhs
    max_abs = float(np.max(np.abs(diff)))
    abs_diff = float(np.max(np.abs(np.abs(lhs) - np.abs(rhs))))  # phase-invariant
    rms = math.sqrt(float(np.mean(diff * diff)))
    worst = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
    loc = ", ".join(str(int(i)) for i in np.atleast_1d(worst))
    print(
        f"{name:<26} max|Δ|={max_abs:.3e}  max||a|-|b||={abs_diff:.3e}  "
        f"rms={rms:.3e}  worst[{loc}] planck={lhs[worst]:+.6e} pyscf={rhs[worst]:+.6e}"
    )
    return max_abs


def _diff_dicts(planck_d: dict, pyscf_d: dict, order: list[str]) -> None:
    for name in order:
        if name in planck_d and name in pyscf_d:
            summarize_diff(name, planck_d[name], pyscf_d[name])
        elif name in planck_d or name in pyscf_d:
            who = "planck" if name in planck_d else "pyscf"
            print(f"{name:<26} only present in {who}")


def cmd_cphf(executable: Path, input_path: Path) -> None:
    print("== CPHF (A, rhs, z) ==")
    pl = planck.cphf_matrices(executable, input_path)
    ref = reference.build_intermediates(input_path)
    _diff_dicts(pl, ref, ["A", "rhs", "z"])


def cmd_imat(executable: Path, input_path: Path) -> None:
    print("== imat_mo blocks ==")
    pl = planck.imat_blocks(executable, input_path)
    ref = reference.build_intermediates(input_path)
    _diff_dicts(pl, ref, ["imat_mo", "imat_top_right", "imat_bottom_left"])


def cmd_chain(executable: Path, input_path: Path) -> None:
    print("== Response-density chain ==")
    pl = planck.response_chain(executable, input_path)
    ref = reference.build_intermediates(input_path)
    _diff_dicts(pl, ref, ["z", "corr_relaxed_mo", "P_ao", "dm1_corr_relaxed_ao", "dm1p"])


def cmd_terms(executable: Path, input_path: Path) -> None:
    print("== Per-atom gradient terms ==")
    pl = planck.gradient_terms(executable, input_path)
    ref = reference.build_terms(input_path)
    order = [
        "two_e", "h1", "h1_kinetic", "h1_nuc_a", "h1_nuc_c",
        "s_im1", "s_zeta", "s_vhf",
        "vhf1", "vhf1_rs", "vhf1_rq", "vhf1_pq", "vhf1_ps",
        "electronic",
    ]
    _diff_dicts(pl, ref, order)


def cmd_gradient(executable: Path, input_path: Path) -> None:
    print("== Final analytic nuclear gradient ==")
    pl = planck.total_gradient(executable, input_path)
    ref = reference.total_gradient(input_path)
    summarize_diff("gradient", pl, ref)
    print("\nPlanck:")
    print(pl)
    print("PySCF:")
    print(ref)


_COMMANDS = {
    "cphf": cmd_cphf,
    "imat": cmd_imat,
    "chain": cmd_chain,
    "terms": cmd_terms,
    "gradient": cmd_gradient,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "stage",
        choices=[*_COMMANDS, "all"],
        help="pipeline stage to compare",
    )
    parser.add_argument("--case", default="water_rmp2_gradient_sto3g", choices=sorted(CASE_INPUTS))
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args()

    input_path = CASE_INPUTS[args.case]
    executable = default_executable(args.build_dir)
    if not executable.exists():
        raise SystemExit(f"hartree-fock binary not found at {executable}")

    stages = list(_COMMANDS) if args.stage == "all" else [args.stage]
    for stage in stages:
        _COMMANDS[stage](executable, input_path)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
