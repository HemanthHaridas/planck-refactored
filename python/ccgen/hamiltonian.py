"""Builders for the normal-ordered Hamiltonian (F_N + V_N)."""

from __future__ import annotations

from fractions import Fraction

from .indices import make_gen
from .tensors import f, v
from .sqops import create, annihilate
from .expr import OpTerm, Expr


def build_fock_operator() -> Expr:
    """F_N = sum_{pq} f_p^q {a+_p a_q}."""
    p = make_gen("p", dummy=True)
    q = make_gen("q", dummy=True)
    term = OpTerm(
        coeff=Fraction(1),
        tensors=(f(p, q),),
        sqops=(create(p), annihilate(q)),
        origin=("H", "F_N"),
    )
    return Expr([term])


def build_two_body_operator() -> Expr:
    """V_N = 1/4 sum_{pqrs} <pq||rs> {a+_r a+_s a_q a_p}."""
    p = make_gen("p", dummy=True)
    q = make_gen("q", dummy=True)
    r = make_gen("r", dummy=True)
    s = make_gen("s", dummy=True)
    term = OpTerm(
        coeff=Fraction(1, 4),
        tensors=(v(p, q, r, s),),
        sqops=(create(r), create(s), annihilate(q), annihilate(p)),
        origin=("H", "V_N"),
    )
    return Expr([term])


def build_hamiltonian() -> Expr:
    """Return F_N + V_N as a symbolic Expr."""
    return build_fock_operator() + build_two_body_operator()
