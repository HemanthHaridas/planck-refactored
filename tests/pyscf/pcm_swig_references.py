"""References for the SWIG C-PCM gates (docs/PCM_GEOMETRY_SCOPE.md steps 1, 6, 4d).

PySCF's default C-PCM (SWIG, lebedev_order 29 = 302 points, vdw_scale 1.2,
modified Bondi, eps 78.3553) on each tests/inputs/regression/*/*pcm_swig*
input, cart following its basis_type (step 6 added the spherical cases), Planck's GBS and its CODATA 2018 Bohr; DFT on Planck's
unpruned 150 x 590 grid. For a `calculation gradient` input it also prints the
analytic gradient (with grid response for DFT) and, for RHF, the three PCM
terms grad_nuc / grad_qv / grad_solver separately. For geomopt inputs it
optimizes with geometric_solver (gau_tight) and reports the optimum; for
geomoptfreq it adds harmonic frequencies from PySCF's analytic PCM Hessian.
  tests/pyscf/.venv/bin/python tests/pyscf/pcm_swig_references.py [name-filter]
"""
import pathlib
import re
import sys

from pyscf import dft, gto, scf
from pyscf.solvent.grad import pcm as pcm_grad
from pyscf.gto.basis import parse_gaussian

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOHR_ANGSTROM = 0.529177210903  # Planck's CODATA 2018 (src/base/tables.h)


def field(text, key):
    m = re.search(rf"^\s*{key}\s+(\S+)", text, re.M)
    return m.group(1) if m else None


def run(path):
    text = path.read_text()
    basis, scf_type, exchange = field(text, "basis"), field(text, "scf_type"), field(text, "exchange")
    coords = text.split("%begin_coords")[1].split("%end_coords")[0].strip().splitlines()
    charge, mult = map(int, coords[1].split())
    gbs = str(ROOT / "basis-sets" / basis)
    elements = {line.split()[0] for line in coords[2:]}
    cart = field(text, "basis_type") != "spherical"
    mol = gto.M(atom="\n".join(coords[2:]), unit=BOHR_ANGSTROM, cart=cart, verbose=0,
                charge=charge, spin=mult - 1,
                basis={e: parse_gaussian.load(gbs, e) for e in elements})
    def make(m):
        if exchange:
            mf = (dft.RKS if mult == 1 else dft.UKS)(m)
            mf.xc = exchange
            mf.grids.atom_grid = (150, 590)
            mf.grids.prune = None
        else:
            mf = (scf.RHF if scf_type == "rhf" else scf.UHF)(m)
        mf = mf.PCM()
        mf.with_solvent.method = "C-PCM"
        mf.with_solvent.eps = 78.3553
        mf.with_solvent.lebedev_order = 29
        # Follow the input's guess so both codes start in the same basin:
        # hcore -> 1e, sad -> minao (PySCF's atomic-density guess).
        mf.init_guess = "minao" if field(text, "guess") == "sad" else "1e"
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9  # PySCF's default sqrt(conv_tol) leaves ~1e-8 in gradients
        return mf

    calculation = field(text, "calculation")
    if calculation in ("geomopt", "geomoptfreq"):
        from pyscf.geomopt.geometric_solver import optimize
        mol = optimize(make(mol), convergence_set="gau_tight", maxsteps=100)
    mf = make(mol)
    e = mf.kernel()
    assert mf.converged, path.stem
    print(f"{path.stem:42s} E = {e:.10f}  G_pcm = {mf.with_solvent.e:.10f}  "
          f"npts = {len(mf.with_solvent.surface['grid_coords'])}")
    if calculation in ("geomopt", "geomoptfreq"):
        print("  optimized geometry (Angstrom): " + "  ".join(
            f"{sym} {x:.6f} {y:.6f} {z:.6f}" for sym, (x, y, z) in
            zip([mol.atom_symbol(i) for i in range(mol.natm)], mol.atom_coords(unit="Angstrom"))))
    if calculation == "geomoptfreq":
        from pyscf.hessian import thermo
        freqs = thermo.harmonic_analysis(mol, mf.Hessian().kernel())["freq_wavenumber"]
        print("  freqs (cm^-1): " + "  ".join(f"{f:.2f}" for f in sorted(freqs.real)))
    if calculation != "gradient":
        return

    def show(label, g):
        print(f"  {label}")
        for row in g:
            print("    [" + ", ".join(f"{x:.12f}" for x in row) + "],")

    g = mf.nuc_grad_method()
    if exchange:
        g.grid_response = True
    show("gradient", g.kernel())
    if not exchange and scf_type == "rhf":
        dm = mf.make_rdm1()
        show("grad_nuc", pcm_grad.grad_nuc(mf.with_solvent, dm))
        show("grad_qv", pcm_grad.grad_qv(mf.with_solvent, dm))
        show("grad_solver", pcm_grad.grad_solver(mf.with_solvent, dm))


name_filter = sys.argv[1] if len(sys.argv) > 1 else ""
for p in sorted(ROOT.glob("tests/inputs/regression/*/*pcm_swig*.hfinp")):
    if name_filter in p.stem:
        run(p)
