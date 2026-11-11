"""References for the DFT counterpoise gates (docs/DFT_COUNTERPOISE.md §1).

The five Boys-Bernardi energies for tests/inputs/regression/dft/*_cp_*.hfinp
(D3 inputs excluded: their gate is that BSSE does not change), each a PySCF KS
single point from an hcore guess, as Planck's sub-calculations run. Ghost grids
at the element's Z (see ghost_dft_references.py).
  tests/pyscf/.venv/bin/python tests/pyscf/cp_dft_references.py
"""
import re

from ghost_dft_references import CASES, energy, parse

for p in sorted(CASES.glob("*_cp_*.hfinp")):
    s = parse(p)
    if re.search(r"^\s*dispersion", s["text"], re.M):
        continue
    frags = [[int(i) - 1 for i in f.split()] for f in re.findall(r"^\s*fragment\s+(.+)$", s["text"], re.M)]
    A, B = frags

    def e(real, ghost):
        atoms = [s["atoms"][i][:2] + (False,) for i in real] + [s["atoms"][i][:2] + (True,) for i in ghost]
        return energy(atoms, 0, 1, s["xc"], s["basis"], s["cart"])

    e_ab, e_a, e_b, e_a_cp, e_b_cp = e(A + B, []), e(A, []), e(B, []), e(A, B), e(B, A)
    bsse = (e_a_cp - e_a) + (e_b_cp - e_b)
    print(f"{p.stem}: E(AB) {e_ab:.10f}  BSSE {bsse:.10f}  "
          f"interaction CP {e_ab - e_a_cp - e_b_cp:.10f}")
