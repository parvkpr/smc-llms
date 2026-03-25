"""
BFS-based DTMC construction — the parallelisation-first replacement for
the recursive DFS in Algorithm 1 of the LLMCHECKER paper.

Why BFS beats DFS here
----------------------
Algorithm 1 calls f(ω) once per node in DFS order.  Each call is a serial
LLM forward pass.  With BFS we collect ALL nodes at depth d, submit them
as a single batch to the inference backend, and receive all their top-k
distributions in one GPU call.

Complexity comparison (ignoring α savings):
  DFS : k^0 + k^1 + ... + k^L  sequential LLM calls  →  O(k^L) serial steps
  BFS : L+1  batched LLM calls                        →  O(L)   serial steps
                                                          O(k^L) parallel work

With vLLM's prefix caching (enable_prefix_caching=True), nodes that share
a common string prefix further reduce the KV-cache computation for the
shared portion to a one-time cost, exactly mirroring the recursion's implicit
reuse but across an entire GPU batch.

Tree structure
--------------
levels[d] contains ALL DTMCNode objects at depth d (including REST terminals).
active_level[d] contains only the non-terminal nodes that need expansion.
Transition probabilities and level_index values are assigned during construction
so that build_sparse_transition_matrix() in dtmc.py can build T[d] without
any additional passes.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tqdm import tqdm  # type: ignore

from .dtmc import DTMCNode, REST_PLACEHOLDER

_TINY_PROB = 1e-12   # ignore REST nodes with negligibly small probability


def build_dtmc_bfs(
    initial_string: str,
    L: int,
    alpha: float,
    k: int,
    quantification_fn: Callable[[str, int], Dict[str, int]],
    llm_backend: Any,
    temperature: float = 1.0,
    verbose: bool = True,
) -> Tuple[List[List[DTMCNode]], Dict[str, float]]:
    """
    Build the α-k-bounded LLM text generation DTMC via BFS.

    Parameters
    ----------
    initial_string   : the prompt / start string ω₀
    L                : maximum generation length in tokens
    alpha            : cumulative probability threshold ∈ (0, 1]
    k                : hard cap on branching factor per node
    quantification_fn: callable(text, depth) → Dict[str, int]
    llm_backend      : VLLMBackend or HFBackend instance
    temperature      : LLM temperature (1.0 = unmodified distribution)
    verbose          : show tqdm progress bar per level

    Returns
    -------
    levels : list of L+1 lists; levels[d] = all nodes at depth d
    stats  : timing and size statistics
    """
    stats: Dict[str, float] = {
        "total_nodes": 0,
        "total_transitions": 0,
        "encoding_time_s": 0.0,
    }
    t_start = time.perf_counter()
    node_counter = [0]

    def _new_node(
        string: str,
        depth: int,
        transition_prob: float,
        is_terminal: bool,
        level_index: int,
    ) -> DTMCNode:
        quant = None if is_terminal else quantification_fn(string, depth)
        n = DTMCNode(
            node_id=node_counter[0],
            string=string,
            depth=depth,
            transition_prob=transition_prob,
            quantification=quant,
            is_terminal=is_terminal,
            level_index=level_index,
        )
        node_counter[0] += 1
        return n

    # ── Level 0: root ─────────────────────────────────────────────────────
    root = _new_node(
        string=initial_string,
        depth=0,
        transition_prob=1.0,
        is_terminal=False,
        level_index=0,
    )
    levels: List[List[DTMCNode]] = [[root]]
    active: List[DTMCNode] = [root]   # non-terminal nodes awaiting expansion
    stats["total_nodes"] += 1

    # ── BFS expansion ─────────────────────────────────────────────────────
    depth_iter = range(L)
    if verbose:
        depth_iter = tqdm(depth_iter, desc="BFS depth", unit="level")

    for depth in depth_iter:
        if not active:
            break

        # ── Single batched LLM call for the entire level ──────────────────
        strings = [node.string for node in active]
        top_k_results = llm_backend.get_top_k_batch(
            strings, alpha=alpha, k=k, temperature=temperature
        )

        next_level: List[DTMCNode] = []
        next_active: List[DTMCNode] = []

        for node, (tokens, probs) in zip(active, top_k_results):
            prob_sum = 0.0

            for token, prob in zip(tokens, probs):
                child_string = node.string + token
                child = _new_node(
                    string=child_string,
                    depth=depth + 1,
                    transition_prob=prob,
                    is_terminal=False,
                    level_index=len(next_level),
                )
                node.add_child(child, prob)
                next_level.append(child)
                next_active.append(child)
                prob_sum += prob
                stats["total_transitions"] += 1

            # REST absorbing state for the remaining probability mass
            rest_prob = max(0.0, 1.0 - prob_sum)
            if rest_prob > _TINY_PROB:
                rest = _new_node(
                    string=REST_PLACEHOLDER,
                    depth=depth + 1,
                    transition_prob=rest_prob,
                    is_terminal=True,
                    level_index=len(next_level),
                )
                node.add_child(rest, rest_prob)
                next_level.append(rest)
                stats["total_transitions"] += 1

        levels.append(next_level)
        stats["total_nodes"] += len(next_level)
        active = next_active

        if verbose:
            depth_iter.set_postfix(  # type: ignore[union-attr]
                nodes=stats["total_nodes"],
                active=len(active),
            )

    stats["encoding_time_s"] = time.perf_counter() - t_start
    return levels, stats
