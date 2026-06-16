"""
app.py — Planck visualizer Flask app (log-file viewer mode).

Serves a Three.js / Chart.js single-page application and exposes one
GET endpoint:

  GET /          → renders templates/index.html
  GET /api/data  → returns pre-parsed log data as JSON
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template

from .parser import ParsedRun
from .molecule3d import ELEMENT_SYMBOLS, COVALENT_RADII

# ---------------------------------------------------------------------------
# Element tables (Z-keyed)
# ---------------------------------------------------------------------------

_CPK_Z: dict[int, str] = {
    1:  "#d4d4d4",  2:  "#d9ffff",  3:  "#cc80ff",  4:  "#c2ff00",
    5:  "#ffb5b5",  6:  "#606060",  7:  "#3b6fe0",  8:  "#e03030",
    9:  "#90e050",  10: "#b3e3f5",  11: "#ab5cf2",  12: "#8aff00",
    13: "#bfa6a6",  14: "#f0c8a0",  15: "#ff8000",  16: "#e8d800",
    17: "#1fd01f",  18: "#80d1e3",  19: "#8f40d4",  20: "#3dff00",
    26: "#e06633",  27: "#f090a0",  28: "#50d050",  29: "#c88033",
    30: "#7d80b0",  35: "#a62929",  53: "#940094",
}

_SPH_R_Z: dict[int, float] = {
    1:  0.27,  2:  0.25,  3:  0.70,  4:  0.40,  5:  0.72,
    6:  0.65,  7:  0.60,  8:  0.55,  9:  0.48,  10: 0.36,
    11: 0.90,  12: 0.70,  13: 0.58,  14: 0.53,  15: 0.50,
    16: 0.50,  17: 0.72,  18: 0.68,  19: 1.00,  20: 0.90,
    26: 0.70,  27: 0.70,  28: 0.70,  29: 0.70,  30: 0.70,
    35: 0.80,  53: 0.90,
}


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

def _detect_bonds(atoms: list, tol: float = 0.45) -> list[list[int]]:
    bonds: list[list[int]] = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_RADII.get(atoms[i].Z, 0.75)
            rj = COVALENT_RADII.get(atoms[j].Z, 0.75)
            dx = atoms[i].x - atoms[j].x
            dy = atoms[i].y - atoms[j].y
            dz = atoms[i].z - atoms[j].z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist <= ri + rj + tol:
                bonds.append([i, j])
    return bonds


# ---------------------------------------------------------------------------
# ParsedRun → JSON
# ---------------------------------------------------------------------------

def _run_to_json(run: ParsedRun, log_path: str) -> dict:
    atoms = run.opt_atoms or run.standard_atoms or run.input_atoms

    symbols = [ELEMENT_SYMBOLS.get(a.Z, f"X{a.Z}") for a in atoms]
    coords_ang = [[a.x, a.y, a.z] for a in atoms]
    bonds = _detect_bonds(atoms)
    n_elec = sum(a.Z for a in atoms)  # neutral-molecule approximation

    mol_data = {
        "symbols": symbols,
        "coords_ang": coords_ang,
        "bonds": bonds,
        "hbonds": [],
        "cpk_colors": [_CPK_Z.get(a.Z, "#ff69b4") for a in atoms],
        "sphere_radii": [_SPH_R_Z.get(a.Z, 0.55) for a in atoms],
        "num_atoms": len(atoms),
        "num_electrons": n_elec,
        "charge": run.charge if run.charge is not None else 0,
        "multiplicity": run.multiplicity if run.multiplicity is not None else 1,
    }

    # SCF
    scf_data = None
    if run.scf_iters:
        energy_history = [it.energy for it in run.scf_iters]
        final_energy = (
            run.total_energy
            if run.total_energy is not None
            else energy_history[-1]
        )
        scf_data = {
            "energy": final_energy,
            "e_nuclear": run.nuclear_repulsion_energy,
            "e_electronic": run.electronic_energy,
            "converged": run.total_energy is not None,
            "iterations": len(run.scf_iters),
            "energy_history": energy_history,
            "n_occ": None,
            "orbital_energies": [],
            "orbital_energies_ev": [],
        }

    # Geopt
    geopt_data = None
    if run.opt_steps:
        last = run.opt_steps[-1]
        geopt_data = {
            "converged": run.opt_converged,
            "iterations": len(run.opt_steps),
            "n_energy_evals": len(run.opt_steps),
            "gradient_rms": float(last.grad_rms),
            "gradient_max": float(last.grad_max),
            "message": (
                "Converged" if run.opt_converged else "Not converged"
            ),
            "trajectory": [
                {
                    "step": s.step,
                    "energy": float(s.energy),
                    "gradient_max": float(s.grad_max),
                    "gradient_rms": float(s.grad_rms),
                }
                for s in run.opt_steps
            ],
        }

    # MP2
    mp2_data = None
    if run.mp2_total_energy is not None:
        mp2_data = {
            "energy": float(run.mp2_total_energy),
            "e_corr": (
                float(run.mp2_corr_energy)
                if run.mp2_corr_energy else None
            ),
        }

    # CASSCF / SA-CASSCF
    casscf_data = None
    if run.casscf_iters or run.casscf_total_energy is not None:
        casscf_data = {
            "active_space": run.casscf_active_space,
            "n_active_electrons": run.casscf_n_active_electrons,
            "n_active_orbitals": run.casscf_n_active_orbitals,
            "n_core": run.casscf_n_core,
            "n_roots": run.casscf_n_roots,
            "converged": run.casscf_converged,
            "iterations": len(run.casscf_iters),
            "energy": (
                float(run.casscf_total_energy)
                if run.casscf_total_energy is not None else None
            ),
            "e_corr": (
                float(run.casscf_corr_energy)
                if run.casscf_corr_energy is not None else None
            ),
            "natural_occs": run.casscf_natural_occs,
            "sa_roots": [
                {
                    "root": r.root,
                    "energy": float(r.energy),
                    "weight": float(r.weight),
                }
                for r in run.casscf_sa_roots
            ],
            "energy_history": [it.energy for it in run.casscf_iters],
            "sa_grad_history": [it.sa_grad for it in run.casscf_iters],
        }

    # Freq
    freq_data = None
    if run.freq_modes:
        freq_data = {
            "frequencies": [m.frequency for m in run.freq_modes],
            "normal_modes": run.normal_modes,
            "n_imaginary": run.n_imaginary,
            "zpe": (
                float(run.zpe_hartree)
                if run.zpe_hartree is not None else None
            ),
        }

    # TDDFT / UV-Vis
    tddft_data = None
    if run.tddft_roots or run.uvvis_points or run.uvvis_peaks:
        tddft_data = {
            "roots": [
                {
                    "root": root.root,
                    "omega_eh": float(root.omega_eh),
                    "omega_ev": float(root.omega_ev),
                    "wavelength_nm": float(root.wavelength_nm),
                    "oscillator_strength": float(root.oscillator_strength),
                }
                for root in run.tddft_roots
            ],
            "uvvis_points": [
                {
                    "energy_ev": float(point.energy_ev),
                    "wavelength_nm": float(point.wavelength_nm),
                    "intensity": float(point.intensity),
                }
                for point in run.uvvis_points
            ],
            "uvvis_peaks": [
                {
                    "energy_ev": float(point.energy_ev),
                    "wavelength_nm": float(point.wavelength_nm),
                    "intensity": float(point.intensity),
                }
                for point in run.uvvis_peaks
            ],
            "uvvis_sigma_ev": (
                float(run.uvvis_sigma_ev)
                if run.uvvis_sigma_ev is not None else None
            ),
            "uvvis_spectrum_path": run.uvvis_spectrum_path,
        }

    return {
        "status": "done",
        "level": run.scf_type or "RHF",
        "basis": run.basis or "unknown",
        "log_file": str(Path(log_path).name) if log_path else "",
        "calculation_type": run.calculation_type,
        "point_group": run.point_group,
        "charge": run.charge,
        "multiplicity": run.multiplicity,
        "molecule": mol_data,
        "scf": scf_data,
        "mp2": mp2_data,
        "casscf": casscf_data,
        "geopt": geopt_data,
        "freq": freq_data,
        "tddft": tddft_data,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(run: ParsedRun, log_path: str = "") -> Flask:
    """Create and return the Flask application (does NOT call app.run())."""
    app = Flask(__name__, template_folder="templates")
    data = _run_to_json(run, log_path)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/data")
    def get_data():
        return jsonify(data)

    return app
