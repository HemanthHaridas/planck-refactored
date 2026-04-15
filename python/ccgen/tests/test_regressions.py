"""Regression tests for the ccgen symbolic pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional in the base interpreter
    np = None


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccgen.generate import generate_cc_equations
from ccgen.indices import make_occ, make_vir, relabel_dummies
from ccgen.tensor_ir import lower_term


def _global_slot(idx, nocc: int) -> int:
    if idx.space == "occ":
        return idx_value(idx)
    if idx.space == "vir":
        return nocc + idx_value(idx)
    raise ValueError(f"Unsupported index space {idx.space!r}")


def idx_value(idx) -> int:
    pools = {
        "occ": "ijklmnopqrstuvwxyz",
        "vir": "abcdefghijklmnopqrstuvwxyz",
    }
    names = pools[idx.space]
    if idx.name not in names:
        raise ValueError(f"Unsupported test index name {idx.name!r}")
    return names.index(idx.name)


def evaluate_scalar_term(
    term,
    tensors: dict[str, np.ndarray],
    nocc: int,
) -> float:
    spaces = {
        idx.name: idx.space for idx in term.summed_indices
    }
    ordered = list(term.summed_indices)

    def recurse(level: int, env: dict[str, int]) -> float:
        if level == len(ordered):
            value = float(term.coeff)
            for factor in term.factors:
                indices = []
                for idx in factor.indices:
                    slot = env[idx.name]
                    if factor.name in ("f", "v"):
                        if idx.space == "occ":
                            indices.append(slot)
                        elif idx.space == "vir":
                            indices.append(nocc + slot)
                        else:
                            raise ValueError(
                                f"Unexpected index space"
                                f" {idx.space!r}"
                            )
                    else:
                        indices.append(slot)
                value *= float(tensors[factor.name][tuple(indices)])
            return value

        idx = ordered[level]
        bound = nocc if spaces[idx.name] == "occ" else tensors["t1"].shape[0]
        total = 0.0
        for val in range(bound):
            env[idx.name] = val
            total += recurse(level + 1, env)
        env.pop(idx.name, None)
        return total

    return recurse(0, {})


def build_test_tensors(
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if np is None:
        raise unittest.SkipTest(
            "numpy is required for tensor-valued regression tests"
        )
    nocc = 2
    nvir = 3
    nso = nocc + nvir
    rng = np.random.default_rng(seed)

    t1_ai = rng.normal(size=(nvir, nocc))
    t2_abij = rng.normal(size=(nvir, nvir, nocc, nocc))
    fock = rng.normal(size=(nso, nso))
    v_full = np.zeros((nso, nso, nso, nso))
    oovv = rng.normal(size=(nocc, nocc, nvir, nvir))

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    v_full[nocc + a, nocc + b, i, j] = oovv[i, j, a, b]
                    v_full[i, j, nocc + a, nocc + b] = oovv[i, j, a, b]

    tensors = {
        "f": fock,
        "v": v_full,
        "t1": t1_ai,
        "t2": t2_abij,
    }
    return tensors, fock, oovv


def has_repeated_antisym_slot(term) -> bool:
    for factor in term.factors:
        for group in factor.antisym_groups:
            slots = [
                (factor.indices[pos].space,
                 factor.indices[pos].name)
                for pos in group
            ]
            if len(slots) != len(set(slots)):
                return True
    return False


class RelabelingTests(unittest.TestCase):
    def test_relabel_dummies_avoids_free_name_collisions(self) -> None:
        free = frozenset({
            make_occ("i", dummy=False),
            make_vir("a", dummy=False),
        })
        mapping = relabel_dummies(
            [make_occ("m", dummy=True), make_vir("e", dummy=True)],
            free=free,
        )

        self.assertEqual(mapping[make_occ("m", dummy=True)].name, "j")
        self.assertEqual(mapping[make_vir("e", dummy=True)].name, "b")


class EquationRegressionTests(unittest.TestCase):
    def test_contraction_lowering_tracks_tensor_slots(
        self,
    ) -> None:
        eqs = generate_cc_equations(
            "ccsd", targets=["energy"],
        )
        energy_terms = eqs["energy"]
        ft1_term = next(
            term for term in energy_terms
            if tuple(
                f.name for f in term.factors
            ) == ("f", "t1")
        )

        lowered = lower_term(ft1_term, target="energy")

        self.assertEqual(lowered.lhs_name, "E_CC")
        self.assertEqual(lowered.lhs_indices, ())
        self.assertEqual(
            lowered.summed_indices,
            ft1_term.summed_indices,
        )

        cmap = {
            item.index.name: item
            for item in lowered.contractions
        }
        self.assertEqual(sorted(cmap), ["a", "i"])
        self.assertEqual(len(cmap["i"].slots), 2)
        self.assertEqual(len(cmap["a"].slots), 2)
        self.assertTrue(
            all(not it.is_free for it in lowered.contractions),
        )

    def test_ccsd_energy_matches_reference_formula(
        self,
    ) -> None:
        tensors, fock, oovv = build_test_tensors()
        eqs = generate_cc_equations(
            "ccsd", targets=["energy"],
        )
        energy_terms = eqs["energy"]

        generated = sum(
            evaluate_scalar_term(term, tensors, nocc=2)
            for term in energy_terms
        )

        t1_ia = tensors["t1"].T
        t2_ijab = np.transpose(tensors["t2"], (2, 3, 0, 1))
        expected = np.einsum(
            "ia,ia", fock[:2, 2:], t1_ia,
        )
        expected += 0.25 * np.einsum(
            "ijab,ijab", t2_ijab, oovv,
        )
        expected += 0.5 * np.einsum(
            "ia,jb,ijab", t1_ia, t1_ia, oovv,
        )

        self.assertAlmostEqual(
            generated, float(expected), places=12,
        )

        try:
            from pyscf.cc import gccsd
        except ImportError:
            return

        class Eris:
            pass

        eris = Eris()
        eris.fock = fock
        eris.oovv = oovv
        pyscf_energy = gccsd.energy(
            object(), t1_ia, t2_ijab, eris,
        )
        self.assertAlmostEqual(
            generated, float(pyscf_energy), places=12,
        )

    def test_ccsd_generation_is_stable(self) -> None:
        counts: list[dict[str, int]] = []
        for _ in range(3):
            eqs = generate_cc_equations("ccsd")
            counts.append(
                {k: len(v) for k, v in eqs.items()},
            )

        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[1], counts[2])

    def test_no_repeated_indices_in_antisymmetric_slots(self) -> None:
        eqs = generate_cc_equations("ccsd")
        for manifold, terms in eqs.items():
            for term in terms:
                self.assertFalse(
                    has_repeated_antisym_slot(term),
                    msg=(
                        f"{manifold} contains a zero"
                        f" antisymmetric term: {term}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
