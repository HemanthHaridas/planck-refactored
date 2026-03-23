"""
parser.py — Planck log file parser.

Reads a saved stdout log from the hartree-fock binary and returns a
ParsedRun dataclass containing all visualisable data.

Log line anatomy (INFO level):
    [YYYY-MM-DD HH:MM:SS] [Planck][INF]       <label:30><message>
    |←    22 chars      →| |← 20 chars →||←  30 chars →||← rest →|

SCF iteration rows are raw (no timestamp):
    {iter:6}{energy:20.10f}{dE:15}{rmsD:15.3e}{maxD:15.3e}
    {diis:15.3e}{damp:12.3f}{t:12.3f}

scf_header() emits two 110-dash lines (before and after the column
header row); scf_footer() emits one more.  The parser enters
SCF_TABLE state on the first dash (after the "Begin SCF Cycles :"
sentinel), ignores the second dash (column-header separator), and
exits on the third dash (footer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AtomCoord:
    Z: int
    x: float  # Angstrom
    y: float
    z: float


@dataclass
class SCFIter:
    iteration: int
    energy: float
    delta_e: float
    rms_d: float
    max_d: float
    diis_err: float
    damping: float
    time_s: float


@dataclass
class OptStep:
    step: int
    energy: float       # Eh (MP2 total when post-HF active, else HF)
    grad_max: float     # Ha/Bohr
    grad_rms: float
    grad_rms_ic: Optional[float] = None   # IC optimizer only


@dataclass
class FreqMode:
    mode: int
    frequency: float          # cm⁻¹; negative → imaginary
    symmetry: Optional[str] = None


@dataclass
class GradAtom:
    atom: int
    dx: float  # Ha/Bohr
    dy: float
    dz: float


@dataclass
class ParsedRun:
    # --- metadata ---
    calculation_type: Optional[str] = None
    scf_type: Optional[str] = None
    basis: Optional[str] = None
    point_group: Optional[str] = None

    # --- geometry ---
    input_atoms: list[AtomCoord] = field(default_factory=list)
    standard_atoms: list[AtomCoord] = field(default_factory=list)
    opt_atoms: list[AtomCoord] = field(default_factory=list)

    # --- initial SCF convergence (first SCF table only) ---
    scf_iters: list[SCFIter] = field(default_factory=list)

    # --- geometry optimisation ---
    opt_steps: list[OptStep] = field(default_factory=list)
    opt_converged: bool = False
    opt_step_geometries: dict[int, list[AtomCoord]] = field(default_factory=dict)

    # --- vibrational frequencies ---
    freq_modes: list[FreqMode] = field(default_factory=list)
    zpe_hartree: Optional[float] = None
    zpe_kcal: Optional[float] = None
    n_imaginary: int = 0

    # --- analytic gradient (single-point gradient calc) ---
    gradient_atoms: list[GradAtom] = field(default_factory=list)
    gradient_max: Optional[float] = None
    gradient_rms: Optional[float] = None

    # --- final energies ---
    total_energy: Optional[float] = None
    mp2_corr_energy: Optional[float] = None
    mp2_total_energy: Optional[float] = None


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Matches any INFO/WARN log line.
# Layout after "[Planck][INF]": 7 padding spaces, label (30 chars), message.
_LOG_RE = re.compile(
    r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] '
    r'\[Planck\]\[(?:INF|WARN|ERR|MAT)\]'
    r'.{7}'     # 7 spaces (setw(20) padding after the 13-char prefix)
    r'(.{30})'  # label field — always 30 chars (may overflow into message)
    r'(.*)$'    # message (may be empty)
)

# 110-dash separator (SCF table header/footer); no timestamp.
_DASH110_RE = re.compile(r'^-{110}\s*$')

# SCF iteration row (no timestamp).
_SCF_ITER_RE = re.compile(
    r'^\s*(\d+)'
    r'\s+([-\d.]+)'
    r'\s+([-\d.eE+\-]+)'
    r'\s+([-\d.eE+\-]+)'
    r'\s+([-\d.eE+\-]+)'
    r'\s+([-\d.eE+\-]+)'
    r'\s+([\d.]+)'
    r'\s+([\d.]+)\s*$'
)

# Atom coordinate line: {Z:5d}{x:10.3f}{y:10.3f}{z:10.3f}
_COORD_RE = re.compile(
    r'^\s*(\d+)'
    r'\s+([-\d.]+)'
    r'\s+([-\d.]+)'
    r'\s+([-\d.]+)\s*$'
)

# Optimised geometry atom line:
#   "  Atom {N:3d}:  {Z:14d}  {x:14.8f}  {y:14.8f}  {z:14.8f}"
_OPT_ATOM_RE = re.compile(
    r'Atom\s+(\d+):\s+(\d+)'
    r'\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
)

# Opt step label and message.
_OPT_STEP_LABEL_RE = re.compile(r'^Opt Step\s+(\d+)\s*:')
_OPT_STEP_MSG_RE = re.compile(
    r'E = ([-\d.]+) Eh\s+'
    r'max\|g\| = ([\d.eE+\-]+)\s+'
    r'rms\|g\| = ([\d.eE+\-]+)'
    r'(?:\s+rms\|g_ic\| = ([\d.eE+\-]+))?'
)

# Vibrational frequency lines (message part).
# With symmetry:    "  {mode:6d}  {sym:10s}  {freq:14.2f}[i  (imaginary)]"
# Without symmetry: "  {mode:6d}  {freq:14.2f}[i  (imaginary)]"
_FREQ_SYM_RE = re.compile(
    r'^\s*(\d+)\s+([A-Za-z][A-Za-z0-9\'"]*)\s+'
    r'([\d.]+)(i)?\s*(?:\(imaginary\))?\s*$'
)
_FREQ_NOSYM_RE = re.compile(
    r'^\s*(\d+)\s+([\d.]+)(i)?\s*(?:\(imaginary\))?\s*$'
)

# ZPE message: "X.XXXXXX Eh  (XX.XX kcal/mol)"
_ZPE_RE = re.compile(
    r'([\d.]+)\s+Eh\s+\(([\d.]+)\s+kcal/mol\)'
)

# Gradient atom label and message.
_GRAD_LABEL_RE = re.compile(r'^Atom\s+(\d+)\s*:')
_GRAD_MSG_RE = re.compile(
    r'^\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$'
)

# Raw energy table lines (no timestamp).
_TOTAL_ENERGY_RE = re.compile(r'^\s*Total Energy\s+([-+\d.eE]+)')
_CORR_ENERGY_RE = re.compile(r'^\s*Correlation Energy\s+([-+\d.eE]+)')
_MP2_ENERGY_RE = re.compile(r'^\s*Total MP2 Energy\s+([-+\d.eE]+)')

# Geometry optimisation convergence sentinel in message.
_OPT_CONV_RE = re.compile(r'Converged in \d+ steps')


# ---------------------------------------------------------------------------
# Parser states
# ---------------------------------------------------------------------------

class _State:
    INIT = "INIT"
    INPUT_COORDS = "INPUT_COORDS"
    STANDARD_COORDS = "STANDARD_COORDS"
    OPT_COORDS = "OPT_COORDS"
    SCF_TABLE = "SCF_TABLE"
    FREQ_TABLE = "FREQ_TABLE"
    GRADIENT = "GRADIENT"
    STEP_GEOM = "STEP_GEOM"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_log_line(raw: str):
    """Return (label_stripped, message) for an INFO line, or None."""
    m = _LOG_RE.match(raw)
    if not m:
        return None
    field30 = m.group(1)   # exactly 30 chars (label field, may have trailing spaces)
    message = m.group(2)
    # Handle label overflow: setw(30) is a minimum, not a maximum.
    # "Optimized Geometry (Angstrom) :" is 31 chars — the trailing ':'
    # bleeds into message[0].  Detect: rstripped field doesn't end with ':'
    # but message starts with ':'.
    if not field30.rstrip().endswith(':') and message.startswith(':'):
        label = (field30 + ':').rstrip()
        message = message[1:]
    else:
        label = field30.rstrip()
    return label, message


def _try_coord(message: str) -> Optional[AtomCoord]:
    m = _COORD_RE.match(message)
    if not m:
        return None
    return AtomCoord(
        int(m.group(1)),
        float(m.group(2)),
        float(m.group(3)),
        float(m.group(4)),
    )


def _try_freq(message: str) -> Optional[FreqMode]:
    m = _FREQ_SYM_RE.match(message)
    if m:
        freq = float(m.group(3)) * (-1 if m.group(4) else 1)
        return FreqMode(int(m.group(1)), freq, m.group(2))
    m = _FREQ_NOSYM_RE.match(message)
    if m:
        freq = float(m.group(2)) * (-1 if m.group(3) else 1)
        return FreqMode(int(m.group(1)), freq)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_log(path: str | Path) -> ParsedRun:
    """Parse a Planck log file and return a ParsedRun."""
    run = ParsedRun()
    state = _State.INIT

    # Tracks the most recent opt step number so STEP_GEOM can key into it.
    _current_step_n: int = -1

    # SCF collection gates.
    # scf_armed  — set after "Begin SCF Cycles :"; cleared after first table.
    # scf_done   — set after the first SCF footer; blocks further collection.
    # scf_dashes — counts 110-dash lines seen while inside SCF_TABLE:
    #              1 = column-header separator (stay), 2 = footer (exit).
    scf_armed = False
    scf_done = False
    scf_dashes = 0

    # Pending gradient atom index (label sets it; message provides values).
    pending_grad_atom: Optional[int] = None

    lines = Path(path).read_text(encoding="utf-8").splitlines()

    for raw in lines:

        # ── Raw (non-INFO) lines ───────────────────────────────────────────
        if _DASH110_RE.match(raw):
            if state == _State.SCF_TABLE:
                scf_dashes += 1
                if scf_dashes >= 2:
                    # Footer: stop collecting.
                    state = _State.INIT
                    scf_done = True
                # scf_dashes == 1 is the column-header separator; stay.
            elif not scf_done and scf_armed:
                # First 110-dash after the "Begin SCF Cycles :" sentinel.
                state = _State.SCF_TABLE
                scf_dashes = 0
            continue

        if state == _State.SCF_TABLE:
            m = _SCF_ITER_RE.match(raw)
            if m:
                run.scf_iters.append(SCFIter(
                    iteration=int(m.group(1)),
                    energy=float(m.group(2)),
                    delta_e=float(m.group(3)),
                    rms_d=float(m.group(4)),
                    max_d=float(m.group(5)),
                    diis_err=float(m.group(6)),
                    damping=float(m.group(7)),
                    time_s=float(m.group(8)),
                ))
            continue  # header text row falls through harmlessly

        # Raw energy table lines (after first SCF has completed).
        if scf_done:
            m = _TOTAL_ENERGY_RE.match(raw)
            if m and run.total_energy is None:
                run.total_energy = float(m.group(1))
                continue
            m = _CORR_ENERGY_RE.match(raw)
            if m:
                run.mp2_corr_energy = float(m.group(1))
                continue
            m = _MP2_ENERGY_RE.match(raw)
            if m:
                run.mp2_total_energy = float(m.group(1))
                continue

        # ── INFO log lines ─────────────────────────────────────────────────
        parsed = _parse_log_line(raw)
        if parsed is None:
            continue
        label, message = parsed

        # ── Metadata ───────────────────────────────────────────────────────
        if label == "Calculation Type :":
            run.calculation_type = message.strip()
            continue
        if label == "Theory :":
            run.scf_type = message.strip()
            continue
        if label == "Basis :":
            run.basis = message.strip()
            continue
        if label == "Point Group :" and run.point_group is None:
            run.point_group = message.strip()
            continue

        # Arm the SCF gate on "Begin SCF Cycles :".
        if label == "Begin SCF Cycles :":
            if not scf_done:
                scf_armed = True
            continue

        # Geometry optimisation convergence flag.
        if label == "Geometry Optimization :" and _OPT_CONV_RE.search(message):
            run.opt_converged = True
            continue

        # Gradient summary (message carries the value).
        if label == "Gradient max|g| :":
            m = re.search(r'([\d.eE+\-]+)', message)
            if m:
                run.gradient_max = float(m.group(1))
            continue
        if label == "Gradient rms|g| :":
            m = re.search(r'([\d.eE+\-]+)', message)
            if m:
                run.gradient_rms = float(m.group(1))
            continue

        # ── State transitions ──────────────────────────────────────────────
        if label == "Input Coordinates :":
            state = _State.INPUT_COORDS
            continue
        if label == "Standard Coordinates :":
            state = _State.STANDARD_COORDS
            continue
        if label == "Optimized Geometry (Angstrom) :":
            state = _State.OPT_COORDS
            continue
        if label == "Vibrational Frequencies :":
            state = _State.FREQ_TABLE
            continue
        if label == "Nuclear Gradient (Ha/Bohr) :":
            state = _State.GRADIENT
            pending_grad_atom = None
            continue

        # ── Opt step ───────────────────────────────────────────────────────
        m = _OPT_STEP_LABEL_RE.match(label)
        if m:
            step_n = int(m.group(1))
            _current_step_n = step_n
            mm = _OPT_STEP_MSG_RE.search(message)
            if mm:
                run.opt_steps.append(OptStep(
                    step=step_n,
                    energy=float(mm.group(1)),
                    grad_max=float(mm.group(2)),
                    grad_rms=float(mm.group(3)),
                    grad_rms_ic=(
                        float(mm.group(4)) if mm.group(4) else None
                    ),
                ))
            state = _State.INIT
            continue

        # ── Step Geometry (per-opt-step geometry snapshot) ─────────────────
        if label == "Step Geometry :":
            if _current_step_n >= 0:
                run.opt_step_geometries[_current_step_n] = []
                state = _State.STEP_GEOM
            continue

        # ── ZPE ────────────────────────────────────────────────────────────
        if label == "Zero-point energy :":
            m = _ZPE_RE.search(message)
            if m:
                run.zpe_hartree = float(m.group(1))
                run.zpe_kcal = float(m.group(2))
            continue

        # ── State-dependent line processing ────────────────────────────────
        if state == _State.INPUT_COORDS:
            if label == "":
                atom = _try_coord(message)
                if atom:
                    run.input_atoms.append(atom)
            else:
                state = _State.INIT

        elif state == _State.STANDARD_COORDS:
            if label == "":
                atom = _try_coord(message)
                if atom:
                    run.standard_atoms.append(atom)
            else:
                state = _State.INIT

        elif state == _State.OPT_COORDS:
            if label == "":
                m = _OPT_ATOM_RE.search(message)
                if m:
                    run.opt_atoms.append(AtomCoord(
                        Z=int(m.group(2)),
                        x=float(m.group(3)),
                        y=float(m.group(4)),
                        z=float(m.group(5)),
                    ))
            else:
                state = _State.INIT

        elif state == _State.FREQ_TABLE:
            if label == "":
                mode = _try_freq(message)
                if mode:
                    run.freq_modes.append(mode)
            # "Zero-point energy :" label is handled above; freq table ends
            # naturally when the next named label arrives.

        elif state == _State.STEP_GEOM:
            if label == "":
                atom = _try_coord(message)
                if atom:
                    run.opt_step_geometries[_current_step_n].append(atom)
            else:
                state = _State.INIT

        elif state == _State.GRADIENT:
            mg = _GRAD_LABEL_RE.match(label)
            if mg:
                pending_grad_atom = int(mg.group(1))
            elif label == "" and pending_grad_atom is not None:
                m = _GRAD_MSG_RE.match(message)
                if m:
                    run.gradient_atoms.append(GradAtom(
                        atom=pending_grad_atom,
                        dx=float(m.group(1)),
                        dy=float(m.group(2)),
                        dz=float(m.group(3)),
                    ))
                pending_grad_atom = None
            elif label not in ("Gradient max|g| :", "Gradient rms|g| :"):
                state = _State.INIT
                pending_grad_atom = None

    run.n_imaginary = sum(1 for f in run.freq_modes if f.frequency < 0)
    return run
