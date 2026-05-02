"""
Diagnostic for the water SA-CASSCF(4,4)/STO-3G two-basin problem (P3).

Demonstrates that:
1. PySCF reaches the same SA minimum (-74.7877865 Eh) regardless of init_guess.
2. Both PySCF and Planck land at the same RHF stationary point.
3. Both pick the same active space (HOMO-1, HOMO, LUMO, LUMO+1).
4. PySCF's converged active orbitals contain a large (~80%) admixture of a
   deeper occupied RHF orbital (RHF MO 2, σ_OH bonding) — i.e. PySCF performed
   a large core-active rotation that Planck's trust-region-capped optimizer
   does not take from the same starting orbitals.
5. Per-root orbital gradients are large at PySCF's convergence too (|g_max| ~
   3.7e-2), but cancel pairwise; this matches Planck's behaviour, so the
   difference is in the basin reached, not in the SA convergence criterion.

Re-run this whenever the Planck SA optimizer changes to check whether the
deeper basin becomes reachable.

Reference: docs/CASSCF_STATUS.md item P3.
"""

import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.fci import direct_spin1
from pyscf.mcscf.mc1step import gen_g_hop


GEOM = """
O   0.000000   0.000000   0.117176
H   0.000000   0.757005  -0.468704
H   0.000000  -0.757005  -0.468704
"""

PLANCK_SA_LOCAL = -74.7751377977
PYSCF_SA_GLOBAL = -74.7877865139


def build_mol():
    mol = gto.Mole()
    mol.atom = GEOM
    mol.basis = "sto-3g"
    mol.cart = True
    mol.symmetry = False
    mol.verbose = 0
    mol.build()
    return mol


def run_pyscf(init_guess: str):
    mol = build_mol()
    mf = scf.RHF(mol)
    mf.init_guess = init_guess
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, 4, 4)
    mc = mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1e-9
    mc.conv_tol_grad = 1e-6
    mc.kernel()
    return mol, mf, mc


def report_basin_independence():
    print("=" * 72)
    print("(1) PySCF guess-independence")
    print("=" * 72)
    print(f"{'Guess':12s} {'RHF':>16s} {'CAS root 0':>16s} {'CAS root 1':>16s} {'SA':>16s}")
    for guess in ["hcore", "1e", "minao", "atom", "huckel"]:
        try:
            _, mf, mc = run_pyscf(guess)
            sa = sum(0.5 * e for e in mc.e_states)
            print(f"{guess:12s} {mf.e_tot:>16.10f} {mc.e_states[0]:>16.10f} {mc.e_states[1]:>16.10f} {sa:>16.10f}")
        except Exception as exc:
            print(f"{guess:12s} FAILED: {exc}")


def report_orbital_decomposition():
    print()
    print("=" * 72)
    print("(2-4) PySCF converged active orbital decomposition")
    print("=" * 72)
    mol, mf, mc = run_pyscf("atom")
    S = mol.intor("int1e_ovlp")
    ncore, ncas = mc.ncore, mc.ncas
    mo_act = mc.mo_coeff[:, ncore:ncore + ncas]
    proj = mf.mo_coeff.T @ S @ mo_act
    print(f"  Active space: ncore={ncore}, ncas={ncas}")
    print(f"  Active MO indices in PySCF MO order: [{ncore}..{ncore + ncas - 1}]")
    from pyscf.mcscf import addons
    _, _, occ_nat = addons.cas_natorb(mc)
    print(f"  Natural occupations: {occ_nat[ncore:ncore + ncas]}")
    print()
    print("  Decomposition of converged active CAS orbitals onto canonical RHF:")
    for i in range(ncas):
        print(f"    CAS active MO {i}:")
        for j in range(mf.mo_coeff.shape[1]):
            c = proj[j, i]
            if abs(c) > 0.05:
                print(f"      RHF MO {j} (E={mf.mo_energy[j]:+.4f}):  {c:+.4f}  ({c**2 * 100:5.1f}%)")
    return mol, mf, mc


def report_per_root_gradients(mol, mf, mc):
    print()
    print("=" * 72)
    print("(5) Per-root vs SA orbital gradient at PySCF convergence")
    print("=" * 72)
    ncas = mc.ncas
    nelec = mc.nelecas
    weights = mc.weights
    eris = mc.ao2mo(mc.mo_coeff)
    per_state_dm1, per_state_dm2 = [], []
    for civec in mc.ci:
        dm1, dm2 = direct_spin1.make_rdm12(civec, ncas, nelec)
        per_state_dm1.append(dm1)
        per_state_dm2.append(dm2)
    sa_dm1 = sum(w * d for w, d in zip(weights, per_state_dm1))
    sa_dm2 = sum(w * d for w, d in zip(weights, per_state_dm2))
    for i, (dm1, dm2) in enumerate(zip(per_state_dm1, per_state_dm2)):
        g, _, _, _ = gen_g_hop(mc, mc.mo_coeff, 0, dm1, dm2, eris=eris)
        print(f"  Per-root  (state {i}): |g_orb|_max = {np.max(np.abs(g)):.3e}    ||g_orb|| = {np.linalg.norm(g):.3e}")
    g_sa, _, _, _ = gen_g_hop(mc, mc.mo_coeff, 0, sa_dm1, sa_dm2, eris=eris)
    print(f"  SA-weighted        : |g_orb|_max = {np.max(np.abs(g_sa)):.3e}    ||g_orb|| = {np.linalg.norm(g_sa):.3e}")


def report_basin_summary(mc):
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    sa = sum(0.5 * e for e in mc.e_states)
    print(f"  PySCF SA energy (this run):    {sa:.10f} Eh")
    print(f"  PySCF SA energy (P3 reference):{PYSCF_SA_GLOBAL:+.10f} Eh   delta {abs(sa - PYSCF_SA_GLOBAL):.2e}")
    print(f"  Planck SA energy (P3 gated):   {PLANCK_SA_LOCAL:.10f} Eh")
    print(f"  Basin gap (Planck - PySCF):    {(PLANCK_SA_LOCAL - PYSCF_SA_GLOBAL) * 1000:+.3f} mEh")


def main():
    report_basin_independence()
    mol, mf, mc = report_orbital_decomposition()
    report_per_root_gradients(mol, mf, mc)
    report_basin_summary(mc)


if __name__ == "__main__":
    main()