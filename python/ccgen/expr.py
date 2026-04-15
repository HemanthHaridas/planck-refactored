"""Symbolic operator expressions.

Terms with coefficients, tensors, and second-quantized operators.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .indices import Index
from .tensors import Tensor
from .sqops import SQOp


@dataclass(frozen=True)
class OpTerm:
    """A single term in an operator expression.

    coeff * (tensor_0 tensor_1 ...) * (sqop_0 sqop_1 ...)
    """

    coeff: Fraction
    tensors: tuple[Tensor, ...]
    sqops: tuple[SQOp, ...]
    origin: tuple[object, ...] = ()

    @property
    def n_create(self) -> int:
        return sum(1 for op in self.sqops if op.is_creator)

    @property
    def n_annihilate(self) -> int:
        return sum(1 for op in self.sqops if op.is_annihilator)

    @property
    def rank_hint(self) -> int:
        """Net excitation rank (creators minus annihilators)."""
        return self.n_create - self.n_annihilate

    def scaled(self, factor: Fraction | int) -> OpTerm:
        return OpTerm(
            self.coeff * Fraction(factor),
            self.tensors,
            self.sqops,
            self.origin,
        )

    def reindexed(self, mapping: dict[Index, Index]) -> OpTerm:
        new_tensors = tuple(t.reindexed(mapping) for t in self.tensors)
        new_sqops = tuple(op.reindexed(mapping) for op in self.sqops)
        return OpTerm(self.coeff, new_tensors, new_sqops, self.origin)

    def with_origin(self, origin: tuple[object, ...]) -> OpTerm:
        return OpTerm(self.coeff, self.tensors, self.sqops, origin)

    def __repr__(self) -> str:
        parts = []
        if self.coeff != 1:
            parts.append(str(self.coeff))
        parts.extend(repr(t) for t in self.tensors)
        parts.extend(repr(op) for op in self.sqops)
        return " ".join(parts) if parts else "1"


@dataclass
class Expr:
    """A sum of operator monomials."""

    terms: list[OpTerm] = field(default_factory=list)

    def __add__(self, other: Expr) -> Expr:
        return Expr(self.terms + other.terms)

    def __iadd__(self, other: Expr) -> Expr:
        self.terms.extend(other.terms)
        return self

    def __mul__(self, scalar: Fraction | int) -> Expr:
        s = Fraction(scalar)
        return Expr([t.scaled(s) for t in self.terms])

    def __rmul__(self, scalar: Fraction | int) -> Expr:
        return self.__mul__(scalar)

    def __neg__(self) -> Expr:
        return self * Fraction(-1)

    def __sub__(self, other: Expr) -> Expr:
        return self + (-other)

    def copy(self) -> Expr:
        return Expr(list(self.terms))

    def drop_zeros(self) -> Expr:
        return Expr([t for t in self.terms if t.coeff != 0])

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        return " + ".join(repr(t) for t in self.terms)
