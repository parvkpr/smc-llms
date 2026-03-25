"""
DTMC tree node and transition matrix utilities.

The DTMC built by LLMChecker is a tree (no back-edges), which means:
  - every state is a unique string prefix
  - all nodes at the same depth are mutually independent
  - backward induction (not general linear system solving) suffices

This structure is the key that makes GPU parallelisation tractable.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REST_PLACEHOLDER = "<REST>"


@dataclass
class DTMCNode:
    """A single state in the α-k-bounded LLM text generation DTMC."""

    node_id: int
    string: str
    depth: int
    transition_prob: float            # P(parent → this node)
    quantification: Optional[Dict[str, int]] = None
    is_terminal: bool = False          # True for REST_PLACEHOLDER absorbing states
    children: List[Tuple[DTMCNode, float]] = field(default_factory=list)
    level_index: int = 0               # index within its depth level (for matrix ops)

    def add_child(self, child: DTMCNode, prob: float) -> None:
        self.children.append((child, prob))

    @property
    def is_rest(self) -> bool:
        return self.string == REST_PLACEHOLDER

    def __repr__(self) -> str:
        q = self.quantification or {}
        short = self.string[-40:] if len(self.string) > 40 else self.string
        return (f"DTMCNode(id={self.node_id}, depth={self.depth}, "
                f"p={self.transition_prob:.4f}, q={q}, "
                f"terminal={self.is_terminal}, str='{short}')")


def build_sparse_transition_matrix(
    level: List[DTMCNode],
    next_level_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build a sparse COO transition matrix T ∈ R^{|level| × next_level_size}.
    T[i, j] = P(level[i] → next_level[j]).

    The backward induction step then reduces to:
        values_d = T @ values_{d+1}
    followed by overriding absorbing / satisfying states.
    """
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for i, node in enumerate(level):
        for child, prob in node.children:
            rows.append(i)
            cols.append(child.level_index)
            vals.append(prob)

    if not rows:
        return torch.zeros(len(level), next_level_size, device=device, dtype=torch.float64)

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float64, device=device)
    return torch.sparse_coo_tensor(
        indices, values, (len(level), next_level_size), device=device
    ).coalesce()
