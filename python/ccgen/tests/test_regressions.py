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

from ccgen.generate import generate_cc_equations, generate_cc_equations_lowered
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
                if factor.name == "delta":
                    lhs, rhs = factor.indices
                    value *= 1.0 if env[lhs.name] == env[rhs.name] else 0.0
                    continue

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


def evaluate_residual_term(
    term,
    tensors: dict[str, np.ndarray],
    nocc: int,
    nvir: int,
) -> np.ndarray:
    shape = tuple(nocc if idx.space == "occ" else nvir for idx in term.free_indices)
    result = np.zeros(shape, dtype=float)
    spaces = {
        idx.name: idx.space
        for idx in list(term.free_indices) + list(term.summed_indices)
    }

    def tensor_value(factor, env: dict[str, int]) -> float:
        if factor.name == "delta":
            lhs, rhs = factor.indices
            return 1.0 if env[lhs.name] == env[rhs.name] else 0.0

        indices = []
        for idx in factor.indices:
            slot = env[idx.name]
            if factor.name in ("f", "v"):
                indices.append(slot if idx.space == "occ" else nocc + slot)
            else:
                indices.append(slot)
        return float(tensors[factor.name][tuple(indices)])

    def recurse_summed(level: int, env: dict[str, int]) -> None:
        if level == len(term.summed_indices):
            value = float(term.coeff)
            for factor in term.factors:
                value *= tensor_value(factor, env)
            free_idx = tuple(env[idx.name] for idx in term.free_indices)
            result[free_idx] += value
            return

        idx = term.summed_indices[level]
        bound = nocc if spaces[idx.name] == "occ" else nvir
        for val in range(bound):
            env[idx.name] = val
            recurse_summed(level + 1, env)
        env.pop(idx.name, None)

    def recurse_free(level: int, env: dict[str, int]) -> None:
        if level == len(term.free_indices):
            recurse_summed(0, env)
            return

        idx = term.free_indices[level]
        bound = nocc if spaces[idx.name] == "occ" else nvir
        for val in range(bound):
            env[idx.name] = val
            recurse_free(level + 1, env)
        env.pop(idx.name, None)

    recurse_free(0, {})
    return result


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

    def test_apply_deltas_does_not_collapse_dummies_onto_like_named_externals(
        self,
    ) -> None:
        """A summation dummy must not be rewritten into a same-named external.

        Cluster amplitudes use dummy index names (a, b, i, j) that collide with
        the projector's external names.  ``apply_deltas`` used to key its
        protected-external lookup on ``(space, name)`` alone, so ANY dummy
        sharing a name with an external was silently rewritten into it -- even
        with no delta relating them.  That collapsed summations into degenerate
        terms (f(a,a), t2(b,b,i,j), t2(a,b,i,j)v(i,j,i,j)) and deleted the
        genuine ladder contractions from every projected manifold.  Fixed by
        matching on union-find COMPONENT rather than name.
        """
        from ccgen.wick import apply_deltas
        from ccgen.tensors import f as fock, t2
        from ccgen.indices import Index

        ext_a = Index("a", "vir", False)
        ext_b = Index("b", "vir", False)
        dum_a = Index("a", "vir", True)
        dum_b = Index("b", "vir", True)
        i = Index("i", "occ", True)
        j = Index("j", "occ", True)

        tensors = (fock(ext_a, dum_a), t2(dum_a, dum_b, i, j))
        out = apply_deltas(tensors, deltas=(), protected=(ext_a, ext_b))

        self.assertIsNotNone(out)
        # No delta relates them, so nothing may change at all.
        self.assertEqual(out, tensors)
        # In particular the dummies stay dummies (no f(a,a) / t2(a,b,..)).
        fock_out = out[0]
        self.assertTrue(fock_out.indices[1].is_dummy)
        self.assertFalse(fock_out.indices[0].is_dummy)
        for idx in out[1].indices:
            self.assertTrue(idx.is_dummy)


class EquationRegressionTests(unittest.TestCase):
    def test_restricted_closed_shell_lowering_preserves_ccsd_manifold_layout(
        self,
    ) -> None:
        lowered = generate_cc_equations_lowered("ccsd")

        self.assertEqual(lowered["energy"][0].orbital_model, "restricted_closed_shell")
        self.assertTrue(
            all(
                tuple(idx.space for idx in term.canonical_free_indices) == ("occ", "vir")
                for term in lowered["singles"]
            )
        )
        self.assertTrue(
            all(
                tuple(idx.space for idx in term.canonical_free_indices)
                == ("occ", "occ", "vir", "vir")
                for term in lowered["doubles"]
            )
        )

        doubles_with_t2 = [
            factor
            for term in lowered["doubles"]
            for factor in term.factors
            if factor.name == "t2"
        ]
        self.assertTrue(doubles_with_t2)
        self.assertTrue(
            all(factor.spatial_signature == ("occ", "occ", "vir", "vir")
                for factor in doubles_with_t2)
        )
        self.assertTrue(
            all(factor.spatial_permutation == (2, 3, 0, 1)
                for factor in doubles_with_t2)
        )

    def test_restricted_closed_shell_lowering_tracks_ccsdt_triples_layout(
        self,
    ) -> None:
        lowered = generate_cc_equations_lowered("ccsdt", targets=["triples"])
        triples = lowered["triples"]

        self.assertTrue(triples)
        self.assertTrue(
            all(
                tuple(idx.space for idx in term.canonical_free_indices)
                == ("occ", "occ", "occ", "vir", "vir", "vir")
                for term in triples
            )
        )

        t3_factors = [
            factor
            for term in triples
            for factor in term.factors
            if factor.name == "t3"
        ]
        self.assertTrue(t3_factors)
        self.assertTrue(
            all(factor.spatial_signature == ("occ", "occ", "occ", "vir", "vir", "vir")
                for factor in t3_factors)
        )
        self.assertTrue(
            all(factor.spatial_permutation == (3, 4, 5, 0, 1, 2)
                for factor in t3_factors)
        )

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

    def test_residual_manifolds_keep_fixed_free_rank(self) -> None:
        eqs = generate_cc_equations("ccsd")

        self.assertTrue(all(len(term.free_indices) == 2 for term in eqs["singles"]))
        self.assertTrue(all(len(term.free_indices) == 4 for term in eqs["doubles"]))

    def test_ccsdt_reduces_to_ccsd_when_t3_dropped(self) -> None:
        # AR3.1 cross-rank reduction: the CCSDT singles/doubles residuals, with
        # every T3-containing term removed, must equal the CCSD singles/doubles
        # residuals EXACTLY (canonical coefficient multiset). This is the strong,
        # oracle-free way to validate higher-rank generation: it chains the
        # PySCF-validated CCSD residual (test_reference_vs_pyscf) up to CCSDT
        # without any per-term CCSDT oracle (PySCF ships no spin-orbital gccsdt).
        # A wrong CCSDT singles/doubles term that survives T3=0 would break this.
        from collections import Counter
        from ccgen.canonicalize import (
            canonicalize_term_to_fixed_point, merge_like_terms,
        )

        def multiset(terms):
            c: Counter = Counter()
            merged = merge_like_terms(
                [canonicalize_term_to_fixed_point(t) for t in terms]
            )
            for t in merged:
                names = tuple(sorted(f.name for f in t.factors))
                c[(names, t.coeff)] += 1
            return c

        def has_t3(term):
            return any(f.name == "t3" for f in term.factors)

        ccsdt = generate_cc_equations("ccsdt")
        ccsd = generate_cc_equations("ccsd")
        for manifold in ("singles", "doubles"):
            reduced = [t for t in ccsdt[manifold] if not has_t3(t)]
            self.assertEqual(
                multiset(reduced), multiset(ccsd[manifold]),
                f"CCSDT {manifold} with T3=0 != CCSD {manifold}",
            )

    def test_diagram_engine_matches_wick_residual(self) -> None:
        # D4.2 -- the load-bearing gate: generate_cc_equations(engine="diagram")
        # produces equations whose per-manifold RESIDUAL equals the wick engine's,
        # for ccsd and ccsdt. Residual equality (not canonical-multiset equality)
        # is the correct gate: the two paths split repeated-factor terms
        # differently (an exchange-symmetry representational choice), so the term
        # multisets differ but the tensors are identical -- and it is the tensor
        # the emitter accumulates. The diagram path is the solve-free front end,
        # no BCH/Wick. Default engine="wick" is unchanged (asserted separately).
        if np is None:
            self.skipTest("numpy required")
        from ccgen.tests.residual_eval import residual_einsum, random_tensors

        no, nv = 4, 5
        tn = random_tensors(no, nv, seed=7)
        for method in ("ccsd", "ccsdt"):
            wick = generate_cc_equations(method, engine="wick")
            dia = generate_cc_equations(method, engine="diagram")
            self.assertEqual(set(wick), set(dia))
            for manifold in wick:
                rw = sum(
                    residual_einsum(t, no, nv, tensors=tn) for t in wick[manifold]
                )
                rd = sum(
                    residual_einsum(t, no, nv, tensors=tn) for t in dia[manifold]
                )
                self.assertTrue(
                    np.allclose(np.asarray(rw), np.asarray(rd), atol=1e-11),
                    f"{method}/{manifold}: "
                    f"{np.max(np.abs(np.asarray(rw) - np.asarray(rd))):.2e}",
                )

    def test_default_engine_is_wick(self) -> None:
        # The diagram engine must be strictly opt-in: the default output is the
        # wick path, byte-identical. (Guards against a default flip.)
        a = generate_cc_equations("ccsd")
        b = generate_cc_equations("ccsd", engine="wick")
        self.assertEqual(
            {m: [repr(t) for t in ts] for m, ts in a.items()},
            {m: [repr(t) for t in ts] for m, ts in b.items()},
        )

    def test_ccsd_singles_linear_seed_matches_pyscf(self) -> None:
        if np is None:
            self.skipTest("numpy is required for PySCF-backed residual tests")
        try:
            from pyscf.cc import gccsd
        except ImportError:
            self.skipTest("PySCF not available")

        nocc = 2
        nvir = 2
        nso = nocc + nvir
        fock = np.zeros((nso, nso))
        fock[2, 0] = 1.0
        fock[0, 2] = 1.0
        mo_energy = np.array([-2.0, -1.0, 1.0, 2.0])
        eri = np.zeros((nso, nso, nso, nso))
        t1 = np.zeros((nocc, nvir))
        t2 = np.zeros((nocc, nocc, nvir, nvir))

        class FakeCC:
            level_shift = 0.0

        eris = gccsd._PhysicistsERIs()
        eris.fock = fock
        eris.mo_energy = mo_energy
        eris.oovv = eri[:nocc, :nocc, nocc:, nocc:]
        eris.ooov = eri[:nocc, :nocc, :nocc, nocc:]
        eris.ovov = eri[:nocc, nocc:, :nocc, nocc:]
        eris.ovvv = eri[:nocc, nocc:, nocc:, nocc:]
        eris.oooo = eri[:nocc, :nocc, :nocc, :nocc]
        eris.vvvv = eri[nocc:, nocc:, nocc:, nocc:]

        tensors = {
            "f": fock,
            "v": eri,
            "t1": t1.T,
            "t2": np.transpose(t2, (2, 3, 0, 1)),
        }
        eqs = generate_cc_equations("ccsd", targets=["singles"])
        generated = sum(
            evaluate_residual_term(term, tensors, nocc, nvir)
            for term in eqs["singles"]
        )

        ref_t1, _ = gccsd.update_amps(FakeCC(), t1, t2, eris)
        eia = mo_energy[:nocc, None] - mo_energy[nocc:]
        np.testing.assert_allclose(generated, ref_t1 * eia, atol=1e-12)

    def test_ccsd_doubles_linear_seed_matches_pyscf(self) -> None:
        if np is None:
            self.skipTest("numpy is required for PySCF-backed residual tests")
        try:
            from pyscf.cc import gccsd
        except ImportError:
            self.skipTest("PySCF not available")

        nocc = 2
        nvir = 2
        nso = nocc + nvir
        fock = np.zeros((nso, nso))
        mo_energy = np.array([-2.0, -1.0, 1.0, 2.0])
        eri = np.zeros((nso, nso, nso, nso))
        eri[0, 1, 2, 3] = 2.0
        eri[1, 0, 2, 3] = -2.0
        eri[0, 1, 3, 2] = -2.0
        eri[1, 0, 3, 2] = 2.0
        eri[2, 3, 0, 1] = 2.0
        eri[3, 2, 0, 1] = -2.0
        eri[2, 3, 1, 0] = -2.0
        eri[3, 2, 1, 0] = 2.0
        t1 = np.zeros((nocc, nvir))
        t2 = np.zeros((nocc, nocc, nvir, nvir))

        class FakeCC:
            level_shift = 0.0

        eris = gccsd._PhysicistsERIs()
        eris.fock = fock
        eris.mo_energy = mo_energy
        eris.oovv = eri[:nocc, :nocc, nocc:, nocc:]
        eris.ooov = eri[:nocc, :nocc, :nocc, nocc:]
        eris.ovov = eri[:nocc, nocc:, :nocc, nocc:]
        eris.ovvv = eri[:nocc, nocc:, nocc:, nocc:]
        eris.oooo = eri[:nocc, :nocc, :nocc, :nocc]
        eris.vvvv = eri[nocc:, nocc:, nocc:, nocc:]

        tensors = {
            "f": fock,
            "v": eri,
            "t1": t1.T,
            "t2": np.transpose(t2, (2, 3, 0, 1)),
        }
        eqs = generate_cc_equations("ccsd", targets=["doubles"])
        generated = sum(
            evaluate_residual_term(term, tensors, nocc, nvir)
            for term in eqs["doubles"]
        )

        _, ref_t2 = gccsd.update_amps(FakeCC(), t1, t2, eris)
        eia = mo_energy[:nocc, None] - mo_energy[nocc:]
        eijab = eia[:, None, :, None] + eia[None, :, None, :]
        np.testing.assert_allclose(generated, ref_t2 * eijab, atol=1e-12)

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
