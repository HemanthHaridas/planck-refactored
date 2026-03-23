"""
test_parser.py — Unit tests for parser.py against visualizer_test.log.

Run from the repo root:
    python -m pytest visualizer/test_parser.py -v
or directly:
    python visualizer/test_parser.py
"""

import sys
from pathlib import Path

# Allow running from repo root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualizer.parser import parse_log, ParsedRun


LOG = Path(__file__).parent.parent / "visualizer_test.log"


def get_run() -> ParsedRun:
    return parse_log(LOG)


# ── Metadata ────────────────────────────────────────────────────────────────

def test_metadata():
    run = get_run()
    assert run.calculation_type == "Geometry Optimization + Frequency"
    assert run.scf_type == "RHF"
    assert run.basis == "6-31g"
    assert run.point_group == "C2v"


# ── Input coordinates ────────────────────────────────────────────────────────

def test_input_coords_count():
    run = get_run()
    assert len(run.input_atoms) == 3, f"Expected 3 input atoms, got {len(run.input_atoms)}"


def test_input_coords_atomic_numbers():
    run = get_run()
    zs = [a.Z for a in run.input_atoms]
    assert zs == [8, 1, 1], f"Unexpected atomic numbers: {zs}"


def test_input_coords_oxygen():
    run = get_run()
    o = run.input_atoms[0]
    assert abs(o.x - -0.000) < 0.01
    assert abs(o.z -  0.033) < 0.01


# ── Standard (symmetry-reoriented) coordinates ───────────────────────────────

def test_standard_coords_count():
    run = get_run()
    assert len(run.standard_atoms) == 3


def test_standard_coords_atomic_numbers():
    run = get_run()
    zs = [a.Z for a in run.standard_atoms]
    assert zs == [8, 1, 1]


# ── Initial SCF convergence ───────────────────────────────────────────────────

def test_scf_iters_collected():
    run = get_run()
    assert len(run.scf_iters) > 0, "No SCF iterations collected"


def test_scf_first_iter():
    run = get_run()
    it = run.scf_iters[0]
    assert it.iteration == 1
    assert abs(it.energy - -69.564) < 0.01


def test_scf_converged_iter_count():
    # Water/6-31G initial SCF converges in 14 iterations
    run = get_run()
    assert len(run.scf_iters) == 14, f"Expected 14 SCF iters, got {len(run.scf_iters)}"


def test_scf_final_energy():
    run = get_run()
    last = run.scf_iters[-1]
    assert abs(last.energy - -75.9797467005) < 1e-6


# ── Energies ─────────────────────────────────────────────────────────────────

def test_total_energy():
    run = get_run()
    assert run.total_energy is not None
    assert abs(run.total_energy - -75.9797467005) < 1e-6


def test_mp2_corr_energy():
    run = get_run()
    assert run.mp2_corr_energy is not None
    assert abs(run.mp2_corr_energy - -0.1321398754) < 1e-6


def test_mp2_total_energy():
    run = get_run()
    assert run.mp2_total_energy is not None
    assert abs(run.mp2_total_energy - -76.1118865759) < 1e-6


# ── Geometry optimisation ─────────────────────────────────────────────────────

def test_opt_converged():
    run = get_run()
    assert run.opt_converged


def test_opt_step_count():
    run = get_run()
    # Water optfreq: 5 opt steps (0..4)
    assert len(run.opt_steps) == 5, f"Expected 5 opt steps, got {len(run.opt_steps)}"


def test_opt_step_numbers():
    run = get_run()
    steps = [s.step for s in run.opt_steps]
    assert steps == [0, 1, 2, 3, 4]


def test_opt_step0_energy():
    run = get_run()
    s0 = run.opt_steps[0]
    assert abs(s0.energy - -76.1118865759) < 1e-6


def test_opt_step_final_energy():
    run = get_run()
    sf = run.opt_steps[-1]
    assert abs(sf.energy - -76.1142119360) < 1e-6


def test_opt_step_grad_max():
    run = get_run()
    s0 = run.opt_steps[0]
    assert abs(s0.grad_max - 2.947e-2) < 1e-4


# ── Optimised geometry ────────────────────────────────────────────────────────

def test_opt_atoms_count():
    run = get_run()
    assert len(run.opt_atoms) == 3


def test_opt_atoms_atomic_numbers():
    run = get_run()
    zs = [a.Z for a in run.opt_atoms]
    assert zs == [8, 1, 1]


def test_opt_oxygen_position():
    run = get_run()
    o = run.opt_atoms[0]
    # From log: 8  -0.00000000  -0.00000000   0.02326127
    assert abs(o.z - 0.02326127) < 1e-6


# ── Vibrational frequencies ───────────────────────────────────────────────────

def test_freq_count():
    run = get_run()
    assert len(run.freq_modes) == 3, f"Expected 3 freq modes, got {len(run.freq_modes)}"


def test_freq_mode_numbers():
    run = get_run()
    modes = [f.mode for f in run.freq_modes]
    assert modes == [1, 2, 3]


def test_freq_symmetry_labels():
    run = get_run()
    syms = [f.symmetry for f in run.freq_modes]
    assert syms == ["A1", "A1", "B2"]


def test_freq_values():
    run = get_run()
    freqs = [f.frequency for f in run.freq_modes]
    assert abs(freqs[0] - 1815.24) < 0.1
    assert abs(freqs[1] - 3659.32) < 0.1
    assert abs(freqs[2] - 3792.65) < 0.1


def test_no_imaginary_freqs():
    run = get_run()
    assert run.n_imaginary == 0


# ── ZPE ──────────────────────────────────────────────────────────────────────

def test_zpe_hartree():
    run = get_run()
    assert run.zpe_hartree is not None
    assert abs(run.zpe_hartree - 0.021112) < 1e-5


def test_zpe_kcal():
    run = get_run()
    assert run.zpe_kcal is not None
    assert abs(run.zpe_kcal - 13.25) < 0.01


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
