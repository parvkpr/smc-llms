"""
PCTL query representation and predicate evaluation.

Supports the queries used in the LLMCHECKER paper:
  P(F  φ)   —  eventually  φ holds
  P(G  φ)   —  always      φ holds

where φ is an atomic proposition over the node's quantification dict,
e.g.  gender > 0,  polarity <= -30,  readability > 5997,  step == 5 ∧ similarity > 90.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .dtmc import DTMCNode

_OPS: Dict[str, Callable] = {
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class AtomicProp:
    """A single comparison: feature OP value  (e.g. gender > 0)."""

    feature: str
    comparator: str
    value: int

    def evaluate(self, quant: Dict[str, int]) -> bool:
        if self.feature not in quant:
            return False
        return _OPS[self.comparator](quant[self.feature], self.value)

    def __str__(self) -> str:
        return f"{self.feature} {self.comparator} {self.value}"


@dataclass
class ConjunctiveProp:
    """Conjunction of multiple AtomicProps (∧ operator)."""

    props: List[AtomicProp]

    def evaluate(self, quant: Dict[str, int]) -> bool:
        return all(p.evaluate(quant) for p in self.props)

    def __str__(self) -> str:
        return " ∧ ".join(str(p) for p in self.props)


@dataclass
class PCTLQuery:
    """
    A PCTL path formula  P∼p(op φ).

    operator : 'F' (eventually) | 'G' (always)
    formula  : AtomicProp or ConjunctiveProp
    threshold: optional probability threshold for comparison (not used in
               exact verification — the checker returns the raw probability)
    """

    operator: str
    formula: AtomicProp | ConjunctiveProp
    threshold: Optional[float] = None

    def predicate(self, node: DTMCNode) -> bool:
        """Return True if the node satisfies the atomic formula φ."""
        if node.is_terminal or node.quantification is None:
            return False
        return self.formula.evaluate(node.quantification)

    def __str__(self) -> str:
        thr = f" ~ {self.threshold}" if self.threshold is not None else ""
        return f"P{thr}({self.operator} {self.formula})"


# ── Convenience constructors ──────────────────────────────────────────────────

def eventually(feature: str, comparator: str, value: int,
               threshold: Optional[float] = None) -> PCTLQuery:
    return PCTLQuery(
        operator="F",
        formula=AtomicProp(feature, comparator, value),
        threshold=threshold,
    )


def always(feature: str, comparator: str, value: int,
           threshold: Optional[float] = None) -> PCTLQuery:
    return PCTLQuery(
        operator="G",
        formula=AtomicProp(feature, comparator, value),
        threshold=threshold,
    )


def eventually_conj(props: List[tuple], threshold: Optional[float] = None) -> PCTLQuery:
    """
    Convenience for conjunctive queries like  P(F step==5 ∧ similarity>90).
    props: list of (feature, comparator, value) tuples.
    """
    return PCTLQuery(
        operator="F",
        formula=ConjunctiveProp([AtomicProp(f, c, v) for f, c, v in props]),
        threshold=threshold,
    )
