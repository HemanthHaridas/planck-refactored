"""
parser.py — Planck log file parser.

Reads a saved stdout log from the hartree-fock binary and returns a
ParsedRun dataclass containing all visualisable data.

Log line anatomy:
    New format:
        <label:30><message>
    Legacy format:
        [YYYY-MM-DD HH:MM:SS] [Planck][INF]       <label:30><message>
        |←    22 chars      →| |← 20 chars →||←  30 chars →||← rest →|

SCF iteration rows are raw:
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
from enum import Enum, auto
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
class CASIter:
    iteration: int
    energy: float
    delta_e: float
    sa_grad: float
    max_root_g: float
    step_err: float
    damping: float
    time_s: float


@dataclass
class CASRoot:
    root: int
    energy: float
    weight: float


@dataclass
class TDDFTRoot:
    root: int
    omega_eh: float
    omega_ev: float
    wavelength_nm: float
    oscillator_strength: float


@dataclass
class UVVisPoint:
    energy_ev: float
    wavelength_nm: float
    intensity: float


@dataclass
class ParsedRun:
    # --- metadata ---
    calculation_type: Optional[str] = None
    scf_type: Optional[str] = None
    basis: Optional[str] = None
    point_group: Optional[str] = None
    charge: Optional[int] = None
    multiplicity: Optional[int] = None

    # --- geometry ---
    input_atoms: list[AtomCoord] = field(default_factory=list)
    standard_atoms: list[AtomCoord] = field(default_factory=list)
    opt_atoms: list[AtomCoord] = field(default_factory=list)

    # --- initial SCF convergence (first SCF table only) ---
    scf_iters: list[SCFIter] = field(default_factory=list)

    # --- geometry optimisation ---
    opt_steps: list[OptStep] = field(default_factory=list)
    opt_converged: bool = False
    opt_step_geometries: dict[int, list[AtomCoord]] = field(
        default_factory=dict
    )

    # --- vibrational frequencies ---
    freq_modes: list[FreqMode] = field(default_factory=list)
    zpe_hartree: Optional[float] = None
    zpe_kcal: Optional[float] = None
    n_imaginary: int = 0

    # --- analytic gradient (single-point gradient calc) ---
    gradient_atoms: list[GradAtom] = field(default_factory=list)
    gradient_max: Optional[float] = None
    gradient_rms: Optional[float] = None

    # --- normal mode displacements ---
    # normal_modes[mode_idx][atom_idx] = [dx, dy, dz]
    normal_modes: list[list[list[float]]] = field(default_factory=list)

    # --- final energies ---
    electronic_energy: Optional[float] = None
    nuclear_repulsion_energy: Optional[float] = None
    total_energy: Optional[float] = None
    mp2_corr_energy: Optional[float] = None
    mp2_total_energy: Optional[float] = None

    # --- CASSCF / SA-CASSCF ---
    casscf_iters: list[CASIter] = field(default_factory=list)
    casscf_active_space: Optional[str] = None          # "(4e, 4o)"
    casscf_n_active_electrons: Optional[int] = None
    casscf_n_active_orbitals: Optional[int] = None
    casscf_n_core: Optional[int] = None
    casscf_n_virt: Optional[int] = None
    casscf_n_roots: Optional[int] = None               # None = single root
    casscf_converged: bool = False
    casscf_natural_occs: list[float] = field(default_factory=list)
    casscf_corr_energy: Optional[float] = None
    casscf_total_energy: Optional[float] = None
    casscf_sa_roots: list[CASRoot] = field(default_factory=list)

    # --- TDDFT / UV-Vis ---
    tddft_roots: list[TDDFTRoot] = field(default_factory=list)
    uvvis_points: list[UVVisPoint] = field(default_factory=list)
    uvvis_peaks: list[UVVisPoint] = field(default_factory=list)
    uvvis_sigma_ev: Optional[float] = None
    uvvis_spectrum_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Matches log lines in three formats:
#   Current:  [INF/WRN/ERR] <label:30><message>
#             └──6 chars──┘ └──30 chars──┘
#   Legacy:   [YYYY-MM-DD HH:MM:SS] [Planck][INF]       <label:30><message>
#   Bare:     <label:30><message>  (no prefix at all)
_LOG_RE = re.compile(
    r'^(?:'
    # Current format: 6-char [INF]/[WRN]/[ERR] prefix.
    r'\[(?:INF|WRN|ERR|MAT)\] '
    r'|'
    # Legacy format: timestamp + [Planck][LEVEL] + 7 spaces.
    r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] '
    r'\[Planck\]\[(?:INF|WARN|ERR|MAT)\]'
    r'.{7}'
    r')?'
    r'(.{30})'  # label field — always 30 chars (may overflow into message)
    r'(.*)$'    # message (may be empty)
)

# 110-dash separator (SCF table header/footer).
_DASH_RE = re.compile(r'^-+\s*$')

# SCF iteration row.
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

# Raw energy table lines.
_ELECTRONIC_ENERGY_RE = re.compile(r'^\s*Electronic Energy\s+([-+\d.eE]+)')
_NUCLEAR_REPULSION_RE = re.compile(r'^\s*Nuclear Repulsion\s+([-+\d.eE]+)')
_TOTAL_ENERGY_RE = re.compile(r'^\s*Total Energy\s+([-+\d.eE]+)')
_CORR_ENERGY_RE = re.compile(r'^\s*Correlation Energy\s+([-+\d.eE]+)')
_MP2_ENERGY_RE = re.compile(r'^\s*Total MP2 Energy\s+([-+\d.eE]+)')

# Normal mode displacement header: "Normal Mode    1 :"
_MODE_LABEL_RE = re.compile(r'^Normal Mode\s+(\d+)\s*:')

# Normal mode displacement row: "  {atom}   {dx}   {dy}   {dz}"
_MODE_DISP_RE = re.compile(
    r'^\s*(\d+)'
    r'\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$'
)

# Geometry optimisation convergence sentinel in message.
_OPT_CONV_RE = re.compile(r'Converged in \d+ steps')

# CASSCF header info in message.
_CASSCF_ACTIVE_RE = re.compile(
    r'Active space:\s+\((\d+)e,\s*(\d+)o\)\s+n_core=(\d+)\s+n_virt=(\d+)'
)
_CASSCF_SA_NROOTS_RE = re.compile(r'State-averaged over (\d+) roots')

# CASSCF raw output section (not [INF] lines).
_CASSCF_NAT_OCC_HEADER_RE = re.compile(r'^\s*CASSCF Natural Occupations\s*:')
_CASSCF_NAT_OCC_ROW_RE = re.compile(r'^\s*MO\s+\d+\s+([\d.]+)\s*$')
_CASSCF_CORR_ENERGY_RE = re.compile(
    r'^\s*CASSCF Correlation Energy\s+([-+\d.eE]+)'
)
_CASSCF_TOTAL_ENERGY_RE = re.compile(
    r'^\s*CASSCF Total Energy\s+([-+\d.eE]+)'
)
_CASSCF_SA_ROOT_RE = re.compile(
    r'^\s*Root\s+(\d+)\s+([-+\d.eE]+)\s+Eh\s+\(weight\s+([\d.eE+\-]+)\)'
)
_TDDFT_ROOT_HEADER_RE = re.compile(
    r'^Root\s+Omega \(Eh\)\s+Omega \(eV\)\s+Lambda \(nm\)\s+f\b'
)
_TDDFT_ROOT_ROW_RE = re.compile(
    r'^\s*(\d+)'
    r'\s+([-+\d.eE]+)'
    r'\s+([-+\d.eE]+)'
    r'\s+([-+\d.eE]+)'
    r'\s+([-+\d.eE]+)'
    r'(?:\s+.*)?$'
)
_UVVIS_WRITE_RE = re.compile(
    r'Wrote\s+\d+\s+Gaussian-broadened points to\s+(\S+)\s+\(sigma =\s+([-+\d.eE]+)\s+eV\)'
)
_UVVIS_PEAK_HEADER_RE = re.compile(
    r'^Peak \(eV\)\s+Lambda \(nm\)\s+Intensity \(arb\)\s*$'
)
_UVVIS_PEAK_ROW_RE = re.compile(
    r'^\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*$'
)


# ---------------------------------------------------------------------------
# Parser states
# ---------------------------------------------------------------------------

class _State(Enum):
    INIT = auto()
    INPUT_COORDS = auto()
    STANDARD_COORDS = auto()
    OPT_COORDS = auto()
    SCF_TABLE = auto()
    CAS_TABLE = auto()
    FREQ_TABLE = auto()
    GRADIENT = auto()
    STEP_GEOM = auto()
    NORMAL_MODE = auto()
    TDDFT_ROOTS = auto()
    UVVIS_PEAKS = auto()


@dataclass(frozen=True)
class _StructuredLine:
    label: str
    message: str


@dataclass
class _ParseContext:
    run: ParsedRun = field(default_factory=ParsedRun)
    state: _State = _State.INIT
    current_step_n: int = -1
    scf_armed: bool = False
    scf_done: bool = False
    scf_dashes: int = 0
    cas_armed: bool = False
    cas_done: bool = False
    cas_dashes: int = 0
    in_cas_nat_occ: bool = False
    pending_grad_atom: Optional[int] = None
    tddft_root_dashes: int = 0
    uvvis_peak_dashes: int = 0


_TEXT_LABEL_FIELDS: dict[str, str] = {
    "Calculation Type :": "calculation_type",
    "Theory :": "scf_type",
    "Basis :": "basis",
}

_INT_LABEL_FIELDS: dict[str, str] = {
    "Charge :": "charge",
    "Multiplicity :": "multiplicity",
}

_SECTION_STATES: dict[str, _State] = {
    "Input Coordinates :": _State.INPUT_COORDS,
    "Standard Coordinates :": _State.STANDARD_COORDS,
    "Optimized Geometry (Angstrom) :": _State.OPT_COORDS,
    "Vibrational Frequencies :": _State.FREQ_TABLE,
    "Nuclear Gradient (Ha/Bohr) :": _State.GRADIENT,
}

_GRADIENT_SUMMARY_FIELDS: dict[str, str] = {
    "Gradient max|g| :": "gradient_max",
    "Gradient rms|g| :": "gradient_rms",
}

_RAW_FLOAT_FIELDS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (_ELECTRONIC_ENERGY_RE, "electronic_energy", True),
    (_NUCLEAR_REPULSION_RE, "nuclear_repulsion_energy", True),
    (_TOTAL_ENERGY_RE, "total_energy", True),
    (_CORR_ENERGY_RE, "mp2_corr_energy", False),
    (_MP2_ENERGY_RE, "mp2_total_energy", False),
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_log_line(raw: str) -> Optional[_StructuredLine]:
    """Return parsed structured line fields, or None for raw-table lines."""
    m = _LOG_RE.match(raw)
    if not m:
        return None
    # exactly 30 chars (label field, may have trailing spaces)
    field30 = m.group(1)
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
    return _StructuredLine(label=label, message=message)


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


def _parse_scf_row(match: re.Match[str]) -> SCFIter:
    return SCFIter(
        iteration=int(match.group(1)),
        energy=float(match.group(2)),
        delta_e=float(match.group(3)),
        rms_d=float(match.group(4)),
        max_d=float(match.group(5)),
        diis_err=float(match.group(6)),
        damping=float(match.group(7)),
        time_s=float(match.group(8)),
    )


def _parse_casscf_row(match: re.Match[str]) -> CASIter:
    return CASIter(
        iteration=int(match.group(1)),
        energy=float(match.group(2)),
        delta_e=float(match.group(3)),
        sa_grad=float(match.group(4)),
        max_root_g=float(match.group(5)),
        step_err=float(match.group(6)),
        damping=float(match.group(7)),
        time_s=float(match.group(8)),
    )


def _set_int_field(run: ParsedRun, field_name: str, value: str) -> None:
    try:
        setattr(run, field_name, int(value.strip()))
    except ValueError:
        pass


def _set_gradient_summary(run: ParsedRun, field_name: str, message: str) -> None:
    match = re.search(r"([\d.eE+\-]+)", message)
    if match:
        setattr(run, field_name, float(match.group(1)))


def _consume_separator_line(ctx: _ParseContext, raw: str) -> bool:
    if not _DASH_RE.match(raw):
        return False

    if ctx.state == _State.SCF_TABLE:
        ctx.scf_dashes += 1
        if ctx.scf_dashes >= 2:
            ctx.state = _State.INIT
            ctx.scf_done = True
        return True

    if ctx.state == _State.CAS_TABLE:
        ctx.cas_dashes += 1
        if ctx.cas_dashes >= 2:
            ctx.state = _State.INIT
            ctx.cas_done = True
        return True

    if ctx.state == _State.TDDFT_ROOTS:
        ctx.tddft_root_dashes += 1
        if ctx.tddft_root_dashes >= 2:
            ctx.state = _State.INIT
        return True

    if ctx.state == _State.UVVIS_PEAKS:
        ctx.uvvis_peak_dashes += 1
        if ctx.uvvis_peak_dashes >= 2:
            ctx.state = _State.INIT
        return True

    if not ctx.scf_done and ctx.scf_armed:
        ctx.state = _State.SCF_TABLE
        ctx.scf_dashes = 0
        return True

    if ctx.scf_done and ctx.cas_armed and not ctx.cas_done:
        ctx.state = _State.CAS_TABLE
        ctx.cas_dashes = 0
        return True

    return True


def _consume_table_row(ctx: _ParseContext, raw: str) -> bool:
    if ctx.state == _State.SCF_TABLE:
        match = _SCF_ITER_RE.match(raw)
        if match:
            ctx.run.scf_iters.append(_parse_scf_row(match))
        return True

    if ctx.state == _State.CAS_TABLE:
        match = _SCF_ITER_RE.match(raw)
        if match:
            ctx.run.casscf_iters.append(_parse_casscf_row(match))
            return True
        return False

    if ctx.state == _State.TDDFT_ROOTS:
        match = _TDDFT_ROOT_ROW_RE.match(raw)
        if match:
            ctx.run.tddft_roots.append(
                TDDFTRoot(
                    root=int(match.group(1)),
                    omega_eh=float(match.group(2)),
                    omega_ev=float(match.group(3)),
                    wavelength_nm=float(match.group(4)),
                    oscillator_strength=float(match.group(5)),
                )
            )
            return True
        return False

    if ctx.state == _State.UVVIS_PEAKS:
        match = _UVVIS_PEAK_ROW_RE.match(raw)
        if match:
            ctx.run.uvvis_peaks.append(
                UVVisPoint(
                    energy_ev=float(match.group(1)),
                    wavelength_nm=float(match.group(2)),
                    intensity=float(match.group(3)),
                )
            )
            return True
        return False

    return False


def _consume_post_scf_raw_line(ctx: _ParseContext, raw: str) -> bool:
    if not ctx.scf_done:
        return False

    for pattern, field_name, first_only in _RAW_FLOAT_FIELDS:
        match = pattern.match(raw)
        if not match:
            continue
        if not first_only or getattr(ctx.run, field_name) is None:
            setattr(ctx.run, field_name, float(match.group(1)))
        return True

    if _CASSCF_NAT_OCC_HEADER_RE.match(raw):
        ctx.in_cas_nat_occ = True
        return True

    if ctx.in_cas_nat_occ:
        match = _CASSCF_NAT_OCC_ROW_RE.match(raw)
        if match:
            ctx.run.casscf_natural_occs.append(float(match.group(1)))
            return True
        if raw.strip() and not raw.strip().startswith("-"):
            ctx.in_cas_nat_occ = False

    match = _CASSCF_CORR_ENERGY_RE.match(raw)
    if match:
        ctx.run.casscf_corr_energy = float(match.group(1))
        return True

    match = _CASSCF_TOTAL_ENERGY_RE.match(raw)
    if match:
        ctx.run.casscf_total_energy = float(match.group(1))
        return True

    match = _CASSCF_SA_ROOT_RE.match(raw)
    if match:
        ctx.run.casscf_sa_roots.append(
            CASRoot(
                root=int(match.group(1)),
                energy=float(match.group(2)),
                weight=float(match.group(3)),
            )
        )
        return True

    return False


def _consume_named_line(ctx: _ParseContext, line: _StructuredLine) -> bool:
    field_name = _TEXT_LABEL_FIELDS.get(line.label)
    if field_name is not None:
        setattr(ctx.run, field_name, line.message.strip())
        return True

    if line.label == "Point Group :" and ctx.run.point_group is None:
        ctx.run.point_group = line.message.strip()
        return True

    field_name = _INT_LABEL_FIELDS.get(line.label)
    if field_name is not None:
        _set_int_field(ctx.run, field_name, line.message)
        return True

    if line.label == "Begin SCF Cycles :":
        if not ctx.scf_done:
            ctx.scf_armed = True
        return True

    if line.label == "CASSCF :":
        message = line.message.strip()
        ctx.cas_armed = True

        match = _CASSCF_ACTIVE_RE.search(message)
        if match:
            ctx.run.casscf_n_active_electrons = int(match.group(1))
            ctx.run.casscf_n_active_orbitals = int(match.group(2))
            ctx.run.casscf_active_space = (
                f"({match.group(1)}e, {match.group(2)}o)"
            )
            ctx.run.casscf_n_core = int(match.group(3))
            ctx.run.casscf_n_virt = int(match.group(4))

        match = _CASSCF_SA_NROOTS_RE.search(message)
        if match:
            ctx.run.casscf_n_roots = int(match.group(1))

        if message == "Converged.":
            ctx.run.casscf_converged = True
            ctx.state = _State.INIT
            ctx.cas_done = True
        return True

    if line.label == "Geometry Optimization :" and _OPT_CONV_RE.search(line.message):
        ctx.run.opt_converged = True
        return True

    field_name = _GRADIENT_SUMMARY_FIELDS.get(line.label)
    if field_name is not None:
        _set_gradient_summary(ctx.run, field_name, line.message)
        return True

    next_state = _SECTION_STATES.get(line.label)
    if next_state is not None:
        ctx.state = next_state
        if next_state == _State.GRADIENT:
            ctx.pending_grad_atom = None
        return True

    match = _OPT_STEP_LABEL_RE.match(line.label)
    if match:
        ctx.current_step_n = int(match.group(1))
        step_match = _OPT_STEP_MSG_RE.search(line.message)
        if step_match:
            ctx.run.opt_steps.append(
                OptStep(
                    step=ctx.current_step_n,
                    energy=float(step_match.group(1)),
                    grad_max=float(step_match.group(2)),
                    grad_rms=float(step_match.group(3)),
                    grad_rms_ic=(
                        float(step_match.group(4))
                        if step_match.group(4)
                        else None
                    ),
                )
            )
        ctx.state = _State.INIT
        return True

    if line.label == "Step Geometry :":
        if ctx.current_step_n >= 0:
            ctx.run.opt_step_geometries[ctx.current_step_n] = []
            ctx.state = _State.STEP_GEOM
        return True

    if line.label == "Zero-point energy :":
        match = _ZPE_RE.search(line.message)
        if match:
            ctx.run.zpe_hartree = float(match.group(1))
            ctx.run.zpe_kcal = float(match.group(2))
        return True

    if line.label == "UV-Vis Spectrum :":
        match = _UVVIS_WRITE_RE.search(line.message)
        if match:
            ctx.run.uvvis_spectrum_path = match.group(1)
            ctx.run.uvvis_sigma_ev = float(match.group(2))
        return True

    if line.label == "Normal Mode Displacements :":
        ctx.state = _State.INIT
        return True

    match = _MODE_LABEL_RE.match(line.label)
    if match:
        ctx.run.normal_modes.append([])
        ctx.state = _State.NORMAL_MODE
        return True

    if ctx.state == _State.GRADIENT:
        match = _GRAD_LABEL_RE.match(line.label)
        if match:
            ctx.pending_grad_atom = int(match.group(1))
            return True

    return False


def _consume_blank_line(ctx: _ParseContext, message: str) -> None:
    stripped = message.strip()

    if _TDDFT_ROOT_HEADER_RE.match(stripped):
        ctx.state = _State.TDDFT_ROOTS
        ctx.tddft_root_dashes = 0
        return

    if _UVVIS_PEAK_HEADER_RE.match(stripped):
        ctx.state = _State.UVVIS_PEAKS
        ctx.uvvis_peak_dashes = 0
        return

    if ctx.state == _State.INPUT_COORDS:
        atom = _try_coord(message)
        if atom:
            ctx.run.input_atoms.append(atom)
        return

    if ctx.state == _State.STANDARD_COORDS:
        atom = _try_coord(message)
        if atom:
            ctx.run.standard_atoms.append(atom)
        return

    if ctx.state == _State.OPT_COORDS:
        match = _OPT_ATOM_RE.search(message)
        if match:
            ctx.run.opt_atoms.append(
                AtomCoord(
                    Z=int(match.group(2)),
                    x=float(match.group(3)),
                    y=float(match.group(4)),
                    z=float(match.group(5)),
                )
            )
        return

    if ctx.state == _State.FREQ_TABLE:
        mode = _try_freq(message)
        if mode:
            ctx.run.freq_modes.append(mode)
        return

    if ctx.state == _State.STEP_GEOM:
        atom = _try_coord(message)
        if atom:
            ctx.run.opt_step_geometries[ctx.current_step_n].append(atom)
        return

    if ctx.state == _State.NORMAL_MODE:
        match = _MODE_DISP_RE.match(message)
        if match and ctx.run.normal_modes:
            ctx.run.normal_modes[-1].append(
                [
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                ]
            )
        return

    if ctx.state == _State.GRADIENT and ctx.pending_grad_atom is not None:
        match = _GRAD_MSG_RE.match(message)
        if match:
            ctx.run.gradient_atoms.append(
                GradAtom(
                    atom=ctx.pending_grad_atom,
                    dx=float(match.group(1)),
                    dy=float(match.group(2)),
                    dz=float(match.group(3)),
                )
            )
        ctx.pending_grad_atom = None


def _reset_section_state_on_unhandled_label(ctx: _ParseContext) -> None:
    if ctx.state in {
        _State.INPUT_COORDS,
        _State.STANDARD_COORDS,
        _State.OPT_COORDS,
        _State.STEP_GEOM,
        _State.NORMAL_MODE,
    }:
        ctx.state = _State.INIT
        return

    if ctx.state == _State.GRADIENT:
        ctx.state = _State.INIT
        ctx.pending_grad_atom = None


def _consume_message_line(ctx: _ParseContext, text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if _TDDFT_ROOT_HEADER_RE.match(stripped):
        ctx.state = _State.TDDFT_ROOTS
        ctx.tddft_root_dashes = 0
        return True

    if _UVVIS_PEAK_HEADER_RE.match(stripped):
        ctx.state = _State.UVVIS_PEAKS
        ctx.uvvis_peak_dashes = 0
        return True

    if ctx.state == _State.TDDFT_ROOTS:
        match = _TDDFT_ROOT_ROW_RE.match(stripped)
        if match:
            ctx.run.tddft_roots.append(
                TDDFTRoot(
                    root=int(match.group(1)),
                    omega_eh=float(match.group(2)),
                    omega_ev=float(match.group(3)),
                    wavelength_nm=float(match.group(4)),
                    oscillator_strength=float(match.group(5)),
                )
            )
            return True

    if ctx.state == _State.UVVIS_PEAKS:
        match = _UVVIS_PEAK_ROW_RE.match(stripped)
        if match:
            ctx.run.uvvis_peaks.append(
                UVVisPoint(
                    energy_ev=float(match.group(1)),
                    wavelength_nm=float(match.group(2)),
                    intensity=float(match.group(3)),
                )
            )
            return True

    return False


def _resolve_uvvis_path(log_path: Path, spectrum_path: str) -> Optional[Path]:
    candidate = Path(spectrum_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidates = [
        log_path.parent / spectrum_path,
        candidate,
    ]
    for resolved in candidates:
        if resolved.exists():
            return resolved.resolve()
    return None


def _load_uvvis_points(log_path: Path, run: ParsedRun) -> None:
    if not run.uvvis_spectrum_path:
        return

    spectrum_path = _resolve_uvvis_path(log_path, run.uvvis_spectrum_path)
    if spectrum_path is None:
        return

    points: list[UVVisPoint] = []
    try:
        for raw in spectrum_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _UVVIS_PEAK_ROW_RE.match(line)
            if not match:
                continue
            points.append(
                UVVisPoint(
                    energy_ev=float(match.group(1)),
                    wavelength_nm=float(match.group(2)),
                    intensity=float(match.group(3)),
                )
            )
    except OSError:
        return

    run.uvvis_points = points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_log(path: str | Path) -> ParsedRun:
    """Parse a Planck log file and return a ParsedRun.

    The parser keeps regex definitions separate from the control flow:
    raw-table rows, structured labels, and section-local blank rows each
    have dedicated handlers so state transitions are easier to follow.
    """
    log_path = Path(path)
    ctx = _ParseContext()

    for raw in log_path.read_text(encoding="utf-8").splitlines():
        if _consume_separator_line(ctx, raw):
            continue

        if _consume_table_row(ctx, raw):
            continue

        if _consume_post_scf_raw_line(ctx, raw):
            continue

        line = _parse_log_line(raw)
        if line is None:
            continue

        if _consume_message_line(ctx, raw):
            continue

        if line.label:
            if _consume_named_line(ctx, line):
                continue
            _reset_section_state_on_unhandled_label(ctx)
            continue

        _consume_blank_line(ctx, line.message)

    ctx.run.n_imaginary = sum(
        1 for mode in ctx.run.freq_modes if mode.frequency < 0
    )
    _load_uvvis_points(log_path, ctx.run)
    return ctx.run
