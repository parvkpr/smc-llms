"""
GPU-accelerated PCTL verification on tree-structured DTMCs.

Three complementary methods are provided:

1. exact_backward_induction
   ─────────────────────────
   Exploits the tree structure of the DTMC to replace the general linear
   system solver (Storm's value iteration) with a simple backward pass.

   For P(F φ):
     base  : prob(s) = 1  if φ(s), 0 if s is terminal
     step  : prob(s) = Σ P(s→s') · prob(s')

   For P(G φ):
     base  : prob(s) = 1  if φ(s) ∧ leaf,  0  if ¬φ(s) or terminal
     step  : prob(s) = 0  if ¬φ(s)
             prob(s) = Σ P(s→s') · prob(s')  otherwise

   Each depth level is processed as a single GPU sparse matrix–vector
   product:
       values_d = T_d  @  values_{d+1}    (torch.sparse_coo)
   followed by a vectorised base-case mask.  This replaces O(|S|) Python
   loops with O(L) GPU kernel launches.

2. statistical_model_check (SMC, DTMC-based)
   ──────────────────────────────────────────
   For very large k or L where even BFS is too memory-hungry, SMC samples
   M independent paths through the already-built DTMC tree by following
   transition probabilities stochastically, then estimates the PCTL
   probability empirically.  Confidence bounds use the Chernoff–Hoeffding
   inequality:
       ε = sqrt(ln(2/δ) / (2M))   (additive error with confidence 1-δ)

   SMC is trivially parallelisable: all M paths are independent and can
   be batched on GPU.

3. direct_smc  ← NOVEL: bypasses DTMC construction entirely
   ──────────────────────────────────────────────────────────
   For long lookaheads (L >> 10) where even building the α-k-bounded DTMC
   is intractable (k_eff^L nodes), direct_smc skips the tree entirely.

   It calls the LLM backend once with n=M to sample M independent complete
   trajectories of length L from the initial string (using vLLM's native
   multi-sequence generation, which batches all M runs on the GPU in one
   call).  The PCTL property is then evaluated on each trajectory by
   applying the quantification function token-by-token.

   Complexity: O(M * L) LLM calls (all parallel on GPU), versus O(k_eff^L)
   for exact verification.  The Chernoff–Hoeffding bound still applies.

   This method is not in the original LLMCHECKER paper (Gross et al., 2025),
   which only performs exact PCTL verification via Storm, limiting the
   approach to L ≤ ~10 tokens.  Direct SMC scales to hundreds of tokens.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch

from .dtmc import DTMCNode, build_sparse_transition_matrix
from .pctl import PCTLQuery


# ── Exact backward induction ──────────────────────────────────────────────────

def exact_backward_induction(
    levels: List[List[DTMCNode]],
    query: PCTLQuery,
    device: str = "cuda",
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the exact PCTL probability for P(F φ) or P(G φ) via GPU-
    accelerated backward induction on the tree DTMC.

    Parameters
    ----------
    levels : output of build_dtmc_bfs() — list of node lists, one per depth
    query  : PCTLQuery with operator ∈ {'F', 'G'}
    device : 'cuda' | 'cpu'

    Returns
    -------
    probability : P(F/G φ) at the root node
    stats       : wall-clock breakdown
    """
    if query.operator not in ("F", "G"):
        raise ValueError(f"Unsupported PCTL operator: {query.operator}")

    stats: Dict[str, float] = {}
    t0 = time.perf_counter()

    L = len(levels) - 1

    # ── Initialise leaf values (depth L) ──────────────────────────────────
    leaf_level = levels[L]
    values = _init_leaf_values(leaf_level, query, device)

    stats["init_ms"] = (time.perf_counter() - t0) * 1000

    # ── Backward pass: depth L-1 → 0 ─────────────────────────────────────
    t_bwd = time.perf_counter()
    for d in range(L - 1, -1, -1):
        level = levels[d]
        next_level_size = len(levels[d + 1])

        # Sparse transition matrix T ∈ R^{|level| × next_level_size}
        T = build_sparse_transition_matrix(level, next_level_size, device)

        # Weighted sum over children: values_d[i] = Σ_j T[i,j] · values[j]
        new_values = torch.sparse.mm(T, values.unsqueeze(1)).squeeze(1)

        # Override base cases
        new_values = _apply_base_cases(new_values, level, query, device)
        values = new_values

    stats["backward_ms"] = (time.perf_counter() - t_bwd) * 1000
    stats["total_ms"] = (time.perf_counter() - t0) * 1000

    probability = values[0].item()
    return probability, stats


def _init_leaf_values(
    leaf_level: List[DTMCNode],
    query: PCTLQuery,
    device: str,
) -> torch.Tensor:
    vals = torch.zeros(len(leaf_level), dtype=torch.float64, device=device)
    for i, node in enumerate(leaf_level):
        if node.is_terminal:
            vals[i] = 0.0
        elif query.predicate(node):
            vals[i] = 1.0
        else:
            vals[i] = 0.0
    return vals


def _apply_base_cases(
    values: torch.Tensor,
    level: List[DTMCNode],
    query: PCTLQuery,
    device: str,
) -> torch.Tensor:
    """
    Vectorised override of base cases after the matrix–vector product.

    For P(F φ):  if φ(s) → 1.0   (absorbing satisfaction)
    For P(G φ):  if ¬φ(s) → 0.0  (absorbing violation)
    Both:        if terminal → 0.0
    """
    sat_mask = torch.zeros(len(level), dtype=torch.bool, device=device)
    term_mask = torch.zeros(len(level), dtype=torch.bool, device=device)

    for i, node in enumerate(level):
        if node.is_terminal:
            term_mask[i] = True
        elif query.predicate(node):
            sat_mask[i] = True

    values = values.clone()
    values[term_mask] = 0.0

    if query.operator == "F":
        values[sat_mask] = 1.0
    elif query.operator == "G":
        # ¬φ(s) means guaranteed violation
        viol_mask = ~sat_mask & ~term_mask
        values[viol_mask] = 0.0

    return values


# ── Statistical model checking ────────────────────────────────────────────────

def statistical_model_check(
    levels: List[List[DTMCNode]],
    query: PCTLQuery,
    num_samples: int = 1000,
    confidence: float = 0.95,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Estimate P(F/G φ) via Monte-Carlo sampling over the tree DTMC.

    Each sample follows a single path root → leaf by drawing child
    transitions according to their probabilities.  All M samples can
    be tracked in parallel as GPU tensors.

    Parameters
    ----------
    levels      : tree levels from build_dtmc_bfs
    query       : PCTLQuery
    num_samples : M  (higher → tighter confidence bound)
    confidence  : 1 - δ for Chernoff–Hoeffding bound
    device      : 'cuda' | 'cpu'
    seed        : optional RNG seed for reproducibility

    Returns
    -------
    estimate    : empirical probability
    lower_bound : Chernoff lower bound at given confidence
    upper_bound : Chernoff upper bound at given confidence
    stats       : timing info
    """
    if seed is not None:
        torch.manual_seed(seed)

    t0 = time.perf_counter()
    L = len(levels) - 1

    # Track M paths simultaneously
    # current_node_indices[m] = index of path m's current node in its level
    current_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    satisfied = torch.zeros(num_samples, dtype=torch.bool, device=device)
    violated = torch.zeros(num_samples, dtype=torch.bool, device=device)

    for d in range(L):
        level = levels[d]
        if not level:
            break

        # Build per-node transition tables.
        # Terminal nodes have no children; give them a self-loop column so
        # multinomial never sees an all-zero row.
        max_children = max((len(n.children) for n in level), default=0)
        max_children = max(max_children, 1)   # always at least one column

        n_nodes = len(level)
        prob_table = torch.zeros(n_nodes, max_children, dtype=torch.float64, device=device)
        idx_table  = torch.zeros(n_nodes, max_children, dtype=torch.long,    device=device)

        for i, node in enumerate(level):
            if node.children:
                for j, (child, prob) in enumerate(node.children):
                    prob_table[i, j] = prob
                    idx_table[i, j]  = child.level_index
            # Terminal node: self-loop at index 0 with prob 1 (absorbed)
            else:
                prob_table[i, 0] = 1.0
                idx_table[i, 0]  = node.level_index

        # Normalise rows to guard against floating-point imprecision
        row_sums   = prob_table.sum(dim=1, keepdim=True).clamp(min=1e-12)
        prob_table = prob_table / row_sums

        # Only sample for paths not yet decided (reduces wasted work)
        active_mask = ~(satisfied | violated)

        # Sample child for each active path (float32 required by multinomial)
        node_probs        = prob_table[current_indices].float()   # [M, max_children]
        sampled_child_col = torch.multinomial(node_probs, num_samples=1).squeeze(1)  # [M]

        # Resolve sampled column to next-level index
        next_indices = idx_table[current_indices, sampled_child_col]  # [M]

        # Check satisfaction / violation in next level
        next_level = levels[d + 1]
        for m_idx in range(num_samples):
            if not active_mask[m_idx]:
                current_indices[m_idx] = next_indices[m_idx]
                continue
            ni   = next_indices[m_idx].item()
            node = next_level[ni]
            if node.is_terminal:
                if query.operator == "G":
                    violated[m_idx] = True
                # For F: terminal = never reached φ; path stays "open"
            elif query.predicate(node):
                if query.operator == "F":
                    satisfied[m_idx] = True
            else:
                if query.operator == "G":
                    violated[m_idx] = True

        current_indices = next_indices

    estimate = satisfied.float().mean().item() if query.operator == "F" else (
        (~violated).float().mean().item()
    )

    # Chernoff–Hoeffding additive bound
    delta = 1.0 - confidence
    eps = math.sqrt(math.log(2.0 / delta) / (2.0 * num_samples))
    lower_bound = max(0.0, estimate - eps)
    upper_bound = min(1.0, estimate + eps)

    stats = {
        "smc_samples": float(num_samples),
        "total_ms": (time.perf_counter() - t0) * 1000,
        "epsilon": eps,
    }

    return estimate, lower_bound, upper_bound, stats


# ── Direct SMC (no DTMC construction) ────────────────────────────────────────

def direct_smc(
    initial_string: str,
    L: int,
    query: PCTLQuery,
    quantification_fn,            # callable(text: str, depth: int) -> Dict[str,int]
    llm_backend,                  # VLLMBackend or HFBackend
    num_samples: int = 2000,
    confidence: float = 0.95,
    temperature: float = 1.0,
    chunk_size: int = 256,        # samples per vLLM call (tune to VRAM)
    verbose: bool = True,
) -> Tuple[float, float, float, Dict]:
    """
    Estimate P(F φ) or P(G φ) for arbitrary L without building a DTMC.

    Instead of constructing the α-k-bounded tree (intractable for large L),
    we sample M complete trajectories of length L from the LLM autoregressively
    and evaluate the PCTL property on each one.

    Each trajectory is a sequence of (text, quant_dict) pairs for depths
    0, 1, ..., L.  The PCTL property is evaluated as:
      P(F φ) : True if ANY depth d has φ(quant_d) = True
      P(G φ) : True if ALL depths d have φ(quant_d) = True

    Parameters
    ----------
    initial_string   : starting prompt ω₀
    L                : number of tokens to generate per trajectory
    query            : PCTLQuery with operator F or G
    quantification_fn: callable(text, depth) → Dict[str,int]
    llm_backend      : VLLMBackend (strongly preferred) or HFBackend
    num_samples      : M — total trajectories to sample
    confidence       : 1-δ for Chernoff–Hoeffding bound
    temperature      : sampling temperature (1.0 = unmodified LLM distribution)
    chunk_size       : trajectories per batch (reduce if OOM)
    verbose          : print progress

    Returns
    -------
    estimate, lower_bound, upper_bound, stats
    """
    from .quantification import Quantifier  # avoid circular imports

    t0 = time.perf_counter()

    # Check whether backend supports native multi-sequence generation
    _has_vllm = hasattr(llm_backend, 'llm')

    satisfied_count = 0
    total_done = 0

    remaining = num_samples
    if verbose:
        from tqdm import tqdm  # type: ignore
        pbar = tqdm(total=num_samples, desc=f"Direct SMC  L={L}", unit="traj")

    while remaining > 0:
        batch = min(chunk_size, remaining)

        if _has_vllm:
            # vLLM: generate `batch` independent sequences in a single call
            trajectories = _vllm_generate_batch(
                llm_backend, initial_string, L, batch, temperature
            )
        else:
            # HF fallback: generate one at a time (slow but correct)
            trajectories = _hf_generate_batch(
                llm_backend, initial_string, L, batch, temperature
            )

        for token_sequence in trajectories:
            # token_sequence: list of token strings of length L
            sat = _evaluate_trajectory(
                initial_string, token_sequence, query, quantification_fn
            )
            if sat:
                satisfied_count += 1

        total_done += batch
        remaining  -= batch
        if verbose:
            pbar.update(batch)
            pbar.set_postfix(p_hat=f"{satisfied_count/total_done:.3f}")

    if verbose:
        pbar.close()

    estimate = satisfied_count / total_done
    delta = 1.0 - confidence
    eps = math.sqrt(math.log(2.0 / delta) / (2.0 * total_done))
    lower_bound = max(0.0, estimate - eps)
    upper_bound = min(1.0, estimate + eps)

    stats = {
        "method":        "direct_smc",
        "smc_samples":   float(num_samples),
        "L":             float(L),
        "satisfied":     float(satisfied_count),
        "total_ms":      (time.perf_counter() - t0) * 1000,
        "epsilon":       eps,
        # State space that exact verification would have needed:
        "exact_states_would_need": float("inf"),
    }
    return estimate, lower_bound, upper_bound, stats


# ── Helpers for direct_smc ────────────────────────────────────────────────────

def _vllm_generate_batch(
    backend,
    initial_string: str,
    L: int,
    n: int,
    temperature: float,
) -> List[List[str]]:
    """
    Use vLLM's native n>1 sampling to generate n independent completions of
    length L in a single forward-pass call.  Returns a list of n token-string
    lists.
    """
    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError:
        raise ImportError("vLLM is required for _vllm_generate_batch")

    params = SamplingParams(
        n=n,
        max_tokens=L,
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        # Request per-token logprobs=0 (we only need the sampled tokens)
    )
    outputs = backend.llm.generate([initial_string], params)
    # outputs[0].outputs is a list of n CompletionOutput objects
    result = []
    for completion in outputs[0].outputs:
        # Split generated text into individual tokens using the tokenizer
        token_ids = completion.token_ids
        tokens = [
            backend.llm.get_tokenizer().decode([tid])
            for tid in token_ids
        ]
        result.append(tokens[:L])
    return result


def _hf_generate_batch(
    backend,
    initial_string: str,
    L: int,
    n: int,
    temperature: float,
) -> List[List[str]]:
    """HF Transformers fallback: generate n sequences one at a time."""
    import torch

    enc = backend.tokenizer(
        initial_string, return_tensors="pt"
    ).to(backend.device)
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        generated = backend.model.generate(
            **enc,
            max_new_tokens=L,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n,
            pad_token_id=backend.tokenizer.eos_token_id,
        )

    result = []
    for seq in generated:
        new_ids = seq[prompt_len:].tolist()
        tokens  = [backend.tokenizer.decode([tid]) for tid in new_ids]
        result.append(tokens[:L])
    return result


def _evaluate_trajectory(
    initial_string: str,
    token_sequence: List[str],
    query: PCTLQuery,
    quantification_fn,
) -> bool:
    """
    Evaluate a PCTL path property on a single sampled trajectory.

    token_sequence[i] is the (i+1)-th generated token (a string).
    We build cumulative prefixes and apply quantification at each step.

    P(F φ) → True if φ holds at ANY prefix
    P(G φ) → True if φ holds at ALL prefixes
    """
    text = initial_string
    results = []

    for depth, token in enumerate(token_sequence, start=1):
        text = text + token
        quant = quantification_fn(text, depth)
        # Evaluate the atomic formula directly (no DTMCNode needed)
        holds = query.formula.evaluate(quant)
        results.append(holds)

    if query.operator == "F":
        return any(results)
    elif query.operator == "G":
        return all(results)
    else:
        raise ValueError(f"Unknown operator: {query.operator}")
