"""
molecule3d.py — 3Dmol.js HTML molecule builder.

No Dash or Plotly dependency; only numpy and the standard library are required.
Public API produces self-contained HTML strings that can be embedded in a
Dash html.Iframe(srcDoc=...) without any server-side rendering.
"""

from __future__ import annotations

import json

import numpy as np


# ---------------------------------------------------------------------------
# Element data (Z → property)  — CPK colors, Alvarez (2008) covalent radii
# ---------------------------------------------------------------------------

ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H",   2: "He",
    3: "Li",  4: "Be",  5: "B",   6: "C",   7: "N",   8: "O",   9: "F",  10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",  16: "S",  17: "Cl", 18: "Ar",
    19: "K",  20: "Ca",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    35: "Br", 53: "I",
}

ELEMENT_COLORS: dict[int, str] = {
    1:  "#FFFFFF",  # H
    2:  "#D9FFFF",  # He
    3:  "#CC80FF",  # Li
    4:  "#C2FF00",  # Be
    5:  "#FFB5B5",  # B
    6:  "#404040",  # C
    7:  "#3050F8",  # N
    8:  "#FF0D0D",  # O
    9:  "#90E050",  # F
    10: "#B3E3F5",  # Ne
    11: "#AB5CF2",  # Na
    12: "#8AFF00",  # Mg
    13: "#BFA6A6",  # Al
    14: "#F0C8A0",  # Si
    15: "#FF8000",  # P
    16: "#FFFF30",  # S
    17: "#1FF01F",  # Cl
    18: "#80D1E3",  # Ar
    19: "#8F40D4",  # K
    20: "#3DFF00",  # Ca
    26: "#E06633",  # Fe
    27: "#F090A0",  # Co
    28: "#50D050",  # Ni
    29: "#C88033",  # Cu
    30: "#7D80B0",  # Zn
    35: "#A62929",  # Br
    53: "#940094",  # I
}

# Alvarez (2008) covalent radii in Ångström
COVALENT_RADII: dict[int, float] = {
    1:  0.31,
    2:  0.28,
    3:  1.28,
    4:  0.96,
    5:  0.84,
    6:  0.76,
    7:  0.71,
    8:  0.66,
    9:  0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    19: 2.03,
    20: 1.76,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    35: 1.20,
    53: 1.39,
}

_DEFAULT_RADIUS = 1.00


def _radius(Z: int) -> float:
    return COVALENT_RADII.get(Z, _DEFAULT_RADIUS)


def _symbol(Z: int) -> str:
    return ELEMENT_SYMBOLS.get(Z, f"X")


# ---------------------------------------------------------------------------
# Bond detection
# ---------------------------------------------------------------------------

def build_bonds(
    coords: np.ndarray,
    atomic_numbers: list[int],
    tolerance: float = 1.3,
) -> list[tuple[int, int]]:
    """Return list of (i, j) pairs where a covalent bond exists.

    Bond criterion:  dist(i, j) < tolerance * (r_cov_i + r_cov_j)
    """
    n = len(atomic_numbers)
    bonds: list[tuple[int, int]] = []
    radii = [_radius(Z) for Z in atomic_numbers]
    for i in range(n):
        for j in range(i + 1, n):
            threshold = tolerance * (radii[i] + radii[j])
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < threshold:
                bonds.append((i, j))
    return bonds


# ---------------------------------------------------------------------------
# XYZ serialisation
# ---------------------------------------------------------------------------

def atoms_to_xyz(atoms, comment: str = "") -> str:
    """Serialise a list of AtomCoord objects to XYZ format (Angstrom).

    Accepts any objects with .Z, .x, .y, .z attributes (duck-typed).
    """
    lines = [str(len(atoms)), comment]
    for a in atoms:
        sym = _symbol(a.Z)
        lines.append(f"{sym:<2s}  {a.x:12.6f}  {a.y:12.6f}  {a.z:12.6f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 3Dmol.js CDN
# ---------------------------------------------------------------------------

_3DMOL_CDN = "https://cdn.jsdelivr.net/npm/3dmol@2/build/3Dmol-min.js"

_BASE_STYLE_JS = (
    'viewer.getModels().forEach(m => m.setStyle({}, '
    '{stick: {radius: 0.12, colorscheme: "Jmol"}, '
    'sphere: {scale: 0.25, colorscheme: "Jmol"}}));'
)


# ---------------------------------------------------------------------------
# Single-geometry viewer
# ---------------------------------------------------------------------------

def viewer_html(atoms, height: int = 450) -> str:
    """Return a self-contained HTML page with a 3Dmol.js viewer for *atoms*.

    Suitable for use as html.Iframe(srcDoc=viewer_html(...)).
    """
    xyz_js = json.dumps(atoms_to_xyz(atoms))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; overflow: hidden; background: white; }}
  #viewer {{ width: 100%; height: 100vh; position: relative; }}
</style>
<script src="{_3DMOL_CDN}"></script>
</head>
<body>
<div id="viewer"></div>
<script>
  let viewer = $3Dmol.createViewer(
    document.getElementById("viewer"),
    {{backgroundColor: "white"}}
  );
  viewer.addModel({xyz_js}, "xyz");
  {_BASE_STYLE_JS}
  viewer.zoomTo();
  viewer.render();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Trajectory (animated multi-step) viewer
# ---------------------------------------------------------------------------

def trajectory_html(step_geometries: dict, height: int = 500) -> str:
    """Return a self-contained HTML page with an animated 3Dmol.js trajectory.

    *step_geometries* is a dict[int, list[AtomCoord]] keyed by step number.
    The page contains Play/Pause and a slider; no Dash callbacks are needed.
    """
    if not step_geometries:
        return (
            "<!DOCTYPE html><html><body style='font-family:monospace;padding:20px'>"
            "<p>No per-step geometry data. Rebuild the binary to enable trajectory replay.</p>"
            "</body></html>"
        )

    sorted_steps = sorted(step_geometries.keys())

    # Build multi-model XYZ (concatenated; 3Dmol.js addModelsAsFrames parses this)
    multi_xyz = "".join(
        atoms_to_xyz(step_geometries[sn], comment=f"Step {sn}")
        for sn in sorted_steps
    )
    xyz_js          = json.dumps(multi_xyz)
    step_labels_js  = json.dumps([str(s) for s in sorted_steps])
    n_frames        = len(sorted_steps)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: white; font-family: monospace; font-size: 13px; }}
  #viewer {{
    width: 100%;
    height: calc(100vh - 48px);
    position: relative;
  }}
  #controls {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 48px;
    background: rgba(248,249,250,0.95);
    border-top: 1px solid #ddd;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 12px;
  }}
  #controls button {{
    padding: 3px 10px;
    cursor: pointer;
    border: 1px solid #aaa;
    border-radius: 3px;
    background: #fff;
    font-size: 13px;
  }}
  #controls button:hover {{ background: #e8e8e8; }}
  #step-slider {{ flex: 1; }}
  #step-label {{ min-width: 80px; text-align: right; color: #333; }}
</style>
<script src="{_3DMOL_CDN}"></script>
</head>
<body>
<div id="viewer"></div>
<div id="controls">
  <button onclick="prevFrame()">&#9664;</button>
  <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
  <button onclick="nextFrame()">&#9654;</button>
  <input type="range" id="step-slider"
         min="0" max="{n_frames - 1}" value="0"
         oninput="setFrame(parseInt(this.value))">
  <span id="step-label">Step 0</span>
</div>
<script>
  const stepLabels = {step_labels_js};
  const nFrames    = {n_frames};
  let   curFrame   = 0;
  let   playing    = false;
  let   playTimer  = null;

  let viewer = $3Dmol.createViewer(
    document.getElementById("viewer"),
    {{backgroundColor: "white"}}
  );
  viewer.addModelsAsFrames({xyz_js}, "xyz");
  {_BASE_STYLE_JS}
  viewer.zoomTo();
  viewer.setFrame(0);
  viewer.render();

  function updateUI() {{
    document.getElementById("step-slider").value = curFrame;
    document.getElementById("step-label").textContent =
      "Step " + stepLabels[curFrame];
  }}

  function setFrame(n) {{
    curFrame = Math.max(0, Math.min(n, nFrames - 1));
    viewer.setFrame(curFrame);
    viewer.render();
    updateUI();
  }}

  function nextFrame() {{ setFrame(curFrame + 1); }}
  function prevFrame() {{ setFrame(curFrame - 1); }}

  function togglePlay() {{
    if (playing) {{
      clearInterval(playTimer);
      playing = false;
      document.getElementById("play-btn").innerHTML = "&#9654; Play";
    }} else {{
      playing = true;
      document.getElementById("play-btn").innerHTML = "&#9646;&#9646; Pause";
      playTimer = setInterval(function() {{
        setFrame(curFrame < nFrames - 1 ? curFrame + 1 : 0);
      }}, 600);
    }}
  }}
</script>
</body>
</html>"""
