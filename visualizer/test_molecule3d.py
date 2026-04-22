"""
test_molecule3d.py — unit tests for molecule3d.py.
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualizer.molecule3d import (  # noqa: E402
    ELEMENT_COLORS,
    ELEMENT_SYMBOLS,
    COVALENT_RADII,
    atoms_to_xyz,
    build_bonds,
    trajectory_html,
    viewer_html,
)
from visualizer.parser import AtomCoord  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WATER_ATOMS = [
    AtomCoord(Z=8, x=0.000, y=0.000, z=0.119),
    AtomCoord(Z=1, x=0.000, y=0.757, z=-0.476),
    AtomCoord(Z=1, x=0.000, y=-0.757, z=-0.476),
]

WATER_Z = [8, 1, 1]
WATER_COORDS = np.array([
    [0.000,  0.000,  0.119],
    [0.000,  0.757, -0.476],
    [0.000, -0.757, -0.476],
])

TRAJ_STEPS = {
    0: WATER_ATOMS,
    1: [
        AtomCoord(Z=8, x=0.000, y=0.000, z=0.115),
        AtomCoord(Z=1, x=0.000, y=0.760, z=-0.470),
        AtomCoord(Z=1, x=0.000, y=-0.760, z=-0.470),
    ],
    2: WATER_ATOMS,
}


# ---------------------------------------------------------------------------
# Element data tests
# ---------------------------------------------------------------------------

def test_element_data_keys_consistent():
    assert (
        set(ELEMENT_COLORS.keys())
        == set(ELEMENT_SYMBOLS.keys())
        == set(COVALENT_RADII.keys())
    )


def test_common_elements_present():
    for Z in (1, 6, 7, 8):
        assert Z in ELEMENT_COLORS
        assert Z in ELEMENT_SYMBOLS
        assert Z in COVALENT_RADII


def test_hydrogen_properties():
    assert ELEMENT_SYMBOLS[1] == "H"
    assert ELEMENT_COLORS[1] == "#FFFFFF"
    assert abs(COVALENT_RADII[1] - 0.31) < 1e-6


# ---------------------------------------------------------------------------
# build_bonds tests
# ---------------------------------------------------------------------------

def test_water_has_two_bonds():
    assert len(build_bonds(WATER_COORDS, WATER_Z)) == 2


def test_bond_indices_valid():
    bonds = build_bonds(WATER_COORDS, WATER_Z)
    n = len(WATER_Z)
    for i, j in bonds:
        assert 0 <= i < n
        assert 0 <= j < n
        assert i < j


def test_no_bonds_far_apart():
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    assert build_bonds(coords, [6, 6]) == []


def test_bond_tolerance():
    r_c = COVALENT_RADII[6]
    d = 1.25 * (r_c + r_c)        # < 1.3 × 2r_c → bonds
    coords = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    assert len(build_bonds(coords, [6, 6])) == 1

    d2 = 1.4 * (r_c + r_c)        # > 1.3 × 2r_c → no bond
    coords2 = np.array([[0.0, 0.0, 0.0], [d2, 0.0, 0.0]])
    assert len(build_bonds(coords2, [6, 6])) == 0


# ---------------------------------------------------------------------------
# atoms_to_xyz tests
# ---------------------------------------------------------------------------

def test_xyz_line_count():
    lines = atoms_to_xyz(WATER_ATOMS).strip().splitlines()
    assert lines[0] == "3"
    assert len(lines) == 2 + 3      # count + comment + 3 atoms


def test_xyz_element_symbols():
    lines = atoms_to_xyz(WATER_ATOMS).strip().splitlines()
    assert lines[2].startswith("O")
    assert lines[3].startswith("H")
    assert lines[4].startswith("H")


def test_xyz_comment():
    lines = atoms_to_xyz(WATER_ATOMS, comment="Step 5").strip().splitlines()
    assert lines[1] == "Step 5"


def test_xyz_coordinates_present():
    lines = atoms_to_xyz(WATER_ATOMS).strip().splitlines()
    assert "0.119000" in lines[2]   # oxygen z-coord


def test_xyz_multi_model_concatenation():
    """Two concatenated frames → 2 × (2 + natoms) lines."""
    multi = (
        atoms_to_xyz(WATER_ATOMS, comment="Frame 0")
        + atoms_to_xyz(WATER_ATOMS, comment="Frame 1")
    )
    assert len(multi.strip().splitlines()) == 2 * (2 + 3)


# ---------------------------------------------------------------------------
# viewer_html tests
# ---------------------------------------------------------------------------

def test_viewer_html_is_string():
    assert isinstance(viewer_html(WATER_ATOMS), str)


def test_viewer_html_contains_3dmol_cdn():
    html = viewer_html(WATER_ATOMS)
    assert "3dmol" in html.lower()
    script_srcs = re.findall(r'<script[^>]+src=["\\\']([^"\\\']+)["\\\']', html, flags=re.IGNORECASE)
    hosts = [urlparse(src).hostname for src in script_srcs]
    assert "cdn.jsdelivr.net" in hosts


def test_viewer_html_contains_xyz_data():
    html = viewer_html(WATER_ATOMS)
    assert '"O' in html or "\\nO" in html or "O  " in html


def test_viewer_html_is_valid_html():
    html = viewer_html(WATER_ATOMS)
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_viewer_html_calls_addmodel():
    assert "addModel" in viewer_html(WATER_ATOMS)


def test_viewer_html_calls_render():
    assert "render()" in viewer_html(WATER_ATOMS)


# ---------------------------------------------------------------------------
# trajectory_html tests
# ---------------------------------------------------------------------------

def test_trajectory_html_is_string():
    assert isinstance(trajectory_html(TRAJ_STEPS), str)


def test_trajectory_html_contains_3dmol():
    assert "3dmol" in trajectory_html(TRAJ_STEPS).lower()


def test_trajectory_html_contains_all_steps():
    html = trajectory_html(TRAJ_STEPS)
    for step in (0, 1, 2):
        assert f"Step {step}" in html


def test_trajectory_html_contains_animation_controls():
    html = trajectory_html(TRAJ_STEPS)
    assert "setFrame" in html
    assert "Play" in html


def test_trajectory_html_contains_slider():
    assert 'type="range"' in trajectory_html(TRAJ_STEPS)


def test_trajectory_html_uses_add_models_as_frames():
    assert "addModelsAsFrames" in trajectory_html(TRAJ_STEPS)


def test_trajectory_html_empty_returns_fallback():
    html = trajectory_html({})
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "No per-step" in html
