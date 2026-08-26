"""R4.2b: dump converged CC amplitudes for the C++ fixture probe.

Solves the GENERATED equations in Python (ccgen_iterate_amps), then writes one
file per rank in the layout the C++ side expects.

TWO layout facts, both load-bearing:

1. **Index order.** ccgen amplitudes are ``(vir...,occ...)``; the C++
   ``rank_dims`` (amplitudes.cpp:54) is ``(occ...,virt...)``. So every tensor is
   transposed on the way out. A mismatch is caught loudly by
   ``seed_arbitrary_order_amplitudes``' exact-dims check -- but only when the
   two counts differ, so DO NOT rely on that for a square case.

2. **Spin-orbital vs spatial.** The Python solve is spin-orbital (GHF); the C++
   RCC state is spatial. ``--spin-blocks`` writes the alpha-alpha spatial block
   (even spin-orbital indices), which is the closed-shell correspondence.

Usage (from repo root):
    tests/pyscf/.venv/bin/python -m ccgen.tests.dump_cc_fixture \\
        --atom "Li 0 0 0; H 0 0 1.6" --basis sto-3g --rank 3 --out <dir>
"""
import argparse
import pathlib
import sys

import numpy as np

import ccgen.tests.test_reference_vs_pyscf as T

_MANIFOLDS = ["singles", "doubles", "triples", "quadruples"]
_TNAME = {1: "t1", 2: "t2", 3: "t3", 4: "t4"}


def solve(atom, basis, rank, spin=0):
    """Solve, returning amplitudes AND the integrals that solve actually used.

    The integrals must be captured, not rebuilt: on a system with a degenerate
    shell (Be's 2p) GHF picks an arbitrary rotation and a second
    _spinorbital_integrals call returns a DIFFERENT orbital basis. See
    test_iterate_amps_fixed_point.
    """
    captured = {}
    orig = T._spinorbital_integrals

    def spy(*args, **kwargs):
        captured["r"] = orig(*args, **kwargs)
        return captured["r"]

    method = {2: "ccsd", 3: "ccsdt", 4: "ccsdtq"}[rank]
    T._spinorbital_integrals = spy
    try:
        e_corr, amps, _mf, _no, _nv = T.ccgen_iterate_amps(
            method, atom, basis, _MANIFOLDS[:rank], spin=spin)
    finally:
        T._spinorbital_integrals = orig
    return e_corr, amps, captured["r"]


def check_fixed_point(amps, integrals, rank, tol=1e-10):
    """Refuse to dump a fixture that is not a fixed point (the vacuity check)."""
    from ccgen.generate import generate_cc_equations
    from ccgen.tests.residual_eval import residual_einsum

    _mf, fock, v, nocc, _nmo, nvir = integrals
    method = {2: "ccsd", 3: "ccsdt", 4: "ccsdtq"}[rank]
    eqs = generate_cc_equations(method)
    tensors = {"v": v, "f": fock, **amps}
    worst = {}
    for m in _MANIFOLDS[:rank]:
        r = sum(residual_einsum(t, nocc, nvir, tensors=tensors) for t in eqs[m])
        worst[m] = float(np.max(np.abs(r)))
    bad = {m: r for m, r in worst.items() if r > tol}
    return worst, bad


def to_cpp_layout(t, rank):
    """ccgen (vir...,occ...) -> C++ (occ...,virt...)."""
    return np.transpose(t, tuple(range(rank, 2 * rank)) + tuple(range(rank)))


def spatial_block(t, rank):
    """Spin-orbital -> the SPATIAL representative block the C++ RCC state holds.

    NOT the all-alpha block. ccgen's spatial representative (spin.py:577) puts a
    single beta on the LAST bra slot and the LAST ket slot: rank 2 -> "abab",
    rank 3 -> "aabaab". Measured on LiH/STO-3G, the all-alpha rank-3 block is
    4.7e-19 -- empty -- while every mixed block carries 8.231e-04, so extracting
    "aaaaaa" would hand the probe a fixture whose t3 is numerically zero and
    reintroduce exactly the inert-manifold vacuity R4.2a was fixed to avoid.

    Alpha spin-orbitals are even indices, beta odd.
    """
    spins = tuple("a" if k != rank - 1 else "b" for k in range(rank)) * 2
    return t[tuple(slice(0 if s == "a" else 1, None, 2) for s in spins)]


def write_tensor(path, t, rank):
    with open(path, "w") as fh:
        fh.write(f"{rank} {t.ndim}\n")
        fh.write(" ".join(str(d) for d in t.shape) + "\n")
        for value in np.ascontiguousarray(t).reshape(-1):
            # repr() on a numpy scalar emits "np.float64(...)"; %.17g is
            # round-trip exact for float64 and parses as a plain C++ double.
            fh.write("%.17g\n" % float(value))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--atom", required=True)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--rank", type=int, default=3)
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spin-blocks", action="store_true",
                    help="write the spatial representative block (C++ RCC is spatial)")
    args = ap.parse_args(argv)

    e_corr, amps, integrals = solve(args.atom, args.basis, args.rank, args.spin)
    print(f"E_corr = {e_corr:.12f}")

    worst, bad = check_fixed_point(amps, integrals, args.rank)
    for m, r in worst.items():
        print(f"  {m:11} max|R| = {r:.3e}")
    if bad:
        print(f"REFUSING to dump: not a fixed point -> {bad}", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in range(1, args.rank + 1):
        t = amps[_TNAME[r]]
        if args.spin_blocks:
            t = spatial_block(t, r)
        t = to_cpp_layout(t, r)
        write_tensor(out / f"t{r}.txt", t, r)
        print(f"  wrote t{r}.txt  shape={t.shape}  max|t|={np.max(np.abs(t)):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ── MO phase alignment (R4.2c) ────────────────────────────────────────────
#
# A converged SCF fixes each MO only up to an overall SIGN. PySCF (which the
# Python solve uses) and Planck's SCF pick independently, so amplitudes and
# residuals built on one are NOT comparable elementwise with the other:
# <ij|ab> picks up p_i p_j p_a p_b.
#
# Measured on LiH/STO-3G: this alone accounted for the ENTIRE apparent
# residual disagreement -- the bare doubles driver differed from PySCF's
# <ij|ab> on exactly the elements where (i==j) != (a==b), and one phase
# choice (1,-1,1,1,1,-1) reconciled them to 1.2e-08, the SCF convergence floor.
# Do NOT compare a dumped fixture against C++ output without this.
#
# ponytail: brute force over 2^nmo. Fine to ~20 orbitals; if a bigger fixture
# is ever needed, align by maximizing overlap with the C++ MO coefficients
# instead of searching.

def solve_phases(reference_block, cpp_block, nocc, nmo):
    """Find per-MO signs making `reference_block` match `cpp_block`.

    Both are (o,o,v,v). Returns (phases, residual) with phases a length-nmo
    array of +/-1, or (None, inf) if no choice reconciles them.
    """
    import itertools

    best = (None, float("inf"))
    for bits in itertools.product((1, -1), repeat=nmo):
        po = np.array(bits[:nocc])
        pv = np.array(bits[nocc:])
        cand = (reference_block
                * po[:, None, None, None] * po[None, :, None, None]
                * pv[None, None, :, None] * pv[None, None, None, :])
        err = float(np.max(np.abs(cand - cpp_block)))
        if err < best[1]:
            best = (np.array(bits), err)
    return best


def apply_phases(t, phases, nocc, rank):
    """Apply per-MO signs to an amplitude in C++ layout (occ...,virt...)."""
    po, pv = phases[:nocc], phases[nocc:]
    out = t.copy()
    for d in range(rank):                      # occupied axes
        shape = [1] * (2 * rank)
        shape[d] = po.shape[0]
        out = out * po.reshape(shape)
    for d in range(rank, 2 * rank):            # virtual axes
        shape = [1] * (2 * rank)
        shape[d] = pv.shape[0]
        out = out * pv.reshape(shape)
    return out
