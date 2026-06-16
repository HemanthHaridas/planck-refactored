"""Second-quantized creation and annihilation operators."""

from __future__ import annotations

from dataclasses import dataclass

from .indices import Index


@dataclass(frozen=True, order=True)
class SQOp:
    """A single fermionic ladder operator."""

    kind: str
    index: Index

    def __post_init__(self) -> None:
        if self.kind not in ("create", "annihilate"):
            raise ValueError(f"Invalid SQOp kind '{self.kind}'")

    @property
    def is_creator(self) -> bool:
        return self.kind == "create"

    @property
    def is_annihilator(self) -> bool:
        return self.kind == "annihilate"

    def reindexed(self, mapping: dict[Index, Index]) -> SQOp:
        return SQOp(self.kind, mapping.get(self.index, self.index))

    def __repr__(self) -> str:
        dag = "\u2020" if self.is_creator else ""
        return f"a{dag}_{self.index}"


def create(idx: Index) -> SQOp:
    return SQOp("create", idx)


def annihilate(idx: Index) -> SQOp:
    return SQOp("annihilate", idx)
