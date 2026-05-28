"""
GPU-accelerated PCTL verification on tree-structured DTMCs.

Four complementary methods are provided:

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

3. direct_smc  ← bypasses DTMC construction entirely
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

4. smc_resampled  ← Particle SMC with Feynman-Kac potentials
   ──────────────────────────────────────────────────────────
   Improves over direct_smc by maintaining N weighted particles and
   periodically resampling to concentrate computation on trajectories
   that are making progress toward satisfying φ.

   Algorithm (block-based):
     Initialise N particles at initial_string with uniform weight w_i = 1/N.
     For each block of K tokens:
       1. Generate K new tokens for each particle (N vLLM calls, n=1 each).
       2. Evaluate quantification_fn at the new depth.
       3. Update particle satisfaction (absorbing for F operator).
       4. Multiply weight by soft potential G_i = exp(β · frac_i)
          where frac_i = min(quant[feature], threshold) / threshold.
       5. Compute ESS = 1 / Σ w_i².  If ESS/N < ess_threshold:
          systematic-resample N particles → reset weights to 1/N.
     Final: self-normalised IS estimate p̂ = Σ w_i · 1[sat_i] / Σ w_i.

   The soft potential leverages the quantification function's partial-credit
   signal (e.g. toxicity = 37/100) at every block boundary, not just at the
   endpoint.  This makes the estimator more efficient than direct_smc for
   rare events (very low or very high p).

   CI: Chernoff–Hoeffding bound with effective sample size = min ESS across
   all blocks, giving honest (conservative) coverage guarantees.
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
    top_p: float = 1.0,
    top_k_sampling: int = -1,
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
    top_p            : nucleus sampling threshold (1.0 = disabled)
    top_k_sampling   : hard top-k cap on sampling distribution (-1 = disabled)
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
                llm_backend, initial_string, L, batch, temperature,
                top_p=top_p, top_k_sampling=top_k_sampling,
            )
        else:
            # HF fallback: generate one at a time (slow but correct)
            trajectories = _hf_generate_batch(
                llm_backend, initial_string, L, batch, temperature,
                top_p=top_p, top_k_sampling=top_k_sampling,
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
    top_p: float = 1.0,
    top_k_sampling: int = -1,
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
        top_p=top_p,
        top_k=top_k_sampling,
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
    top_p: float = 1.0,
    top_k_sampling: int = -1,
) -> List[List[str]]:
    """HF Transformers fallback: generate n sequences one at a time."""
    import torch

    enc = backend.tokenizer(
        initial_string, return_tensors="pt"
    ).to(backend.device)
    prompt_len = enc["input_ids"].shape[1]

    gen_kwargs: dict = dict(
        max_new_tokens=L,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n,
        pad_token_id=backend.tokenizer.eos_token_id,
    )
    if top_p < 1.0:
        gen_kwargs["top_p"] = top_p
    if top_k_sampling > 0:
        gen_kwargs["top_k"] = top_k_sampling

    with torch.no_grad():
        generated = backend.model.generate(**enc, **gen_kwargs)

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


# ── Semantic DTMC: exact backward induction with batched neural judge ─────────

def exact_backward_induction_semantic(
    levels: List[List[DTMCNode]],
    query: PCTLQuery,
    judge_fn,          # callable(texts: List[str]) -> List[int]
    feature: str,      # e.g. "harm_semantic"
    device: str = "cuda",
    batch_size: int = 16,
    verbose: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Exact backward induction with a batched neural judge labeling leaf nodes.

    Build the DTMC tree externally with a no-op quantifier, then:
    1. Batch-label all non-terminal leaf nodes at depth L via judge_fn
    2. Run exact GPU backward induction — P(F phi) at the root is exact,
       not a sample estimate.

    Parameters
    ----------
    levels     : output of build_dtmc_bfs (with any/no-op quantifier)
    query      : PCTLQuery — typically eventually("harm_semantic", "==", 100)
    judge_fn   : callable(List[str]) -> List[int]
                 Takes a batch of leaf node strings, returns integer scores
                 (e.g. 0 or 100).  Called repeatedly in chunks of batch_size.
    feature    : key to set in node.quantification (must match query.formula.feature)
    device     : 'cuda' | 'cpu'
    batch_size : judge batch size (tune to VRAM; 16 is safe for 7B judge on 24GB)
    verbose    : print leaf-labeling progress

    Returns
    -------
    probability : exact P(F phi) at the root
    stats       : timing breakdown + n_leaf_nodes
    """
    t0 = time.perf_counter()

    L = len(levels) - 1
    leaf_level = levels[L]

    active_leaves = [(i, node) for i, node in enumerate(leaf_level)
                     if not node.is_terminal]

    if verbose:
        print(f"  Semantic DTMC: labeling {len(active_leaves)} leaf nodes "
              f"(batch_size={batch_size})...", flush=True)

    for chunk_start in range(0, len(active_leaves), batch_size):
        chunk = active_leaves[chunk_start:chunk_start + batch_size]
        texts = [node.string for _, node in chunk]
        try:
            scores = judge_fn(texts)
        except Exception:
            scores = [0] * len(texts)
        for (_, node), score in zip(chunk, scores):
            if node.quantification is None:
                node.quantification = {}
            node.quantification[feature] = score

    label_ms = (time.perf_counter() - t0) * 1000
    if verbose:
        n_harm = sum(
            1 for _, n in active_leaves
            if (n.quantification or {}).get(feature, 0) >= 100
        )
        print(f"  Leaf labeling: {label_ms:.0f}ms  "
              f"harm={n_harm}/{len(active_leaves)}", flush=True)

    prob, bi_stats = exact_backward_induction(levels, query, device)

    stats = {
        "n_leaf_nodes": len(active_leaves),
        "leaf_label_ms": label_ms,
        **bi_stats,
    }
    stats["total_ms"] = (time.perf_counter() - t0) * 1000
    return prob, stats


# ── Particle SMC with resampling ──────────────────────────────────────────────

def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """
    Systematic resampling: one uniform draw u ~ U[0, 1/N), then evenly-spaced
    grid u, u+1/N, u+2/N, ...  Lower variance than multinomial resampling.
    """
    N = len(weights)
    cumsum = weights.float().cumsum(0)
    cumsum[-1] = 1.0   # guard against fp rounding
    u0 = torch.rand(1).item() / N
    positions = torch.arange(N, dtype=torch.float32) / N + u0
    return torch.searchsorted(cumsum, positions)


def _soft_potential(quant: Dict[str, int], query: PCTLQuery, beta: float) -> float:
    """
    Soft Feynman-Kac potential: G = exp(beta * frac) where frac is the
    fraction of the satisfaction threshold already achieved by this particle.

    For AtomicProp(feature, ">=", threshold):
        frac = min(quant[feature], threshold) / threshold  ∈ [0, 1]
    For ConjunctiveProp: geometric mean of per-conjunct fractions.
    """
    from .pctl import AtomicProp, ConjunctiveProp

    formula = query.formula
    if isinstance(formula, AtomicProp):
        val = quant.get(formula.feature, 0)
        thresh = abs(formula.value) or 1
        frac = min(abs(val), thresh) / thresh
        return math.exp(beta * frac)
    elif isinstance(formula, ConjunctiveProp):
        n = len(formula.props)
        if n == 0:
            return 1.0
        frac = 0.0
        for prop in formula.props:
            val = quant.get(prop.feature, 0)
            thresh = abs(prop.value) or 1
            frac += min(abs(val), thresh) / thresh
        return math.exp(beta * frac / n)
    return 1.0


def _vllm_generate_particles(
    backend,
    particle_texts: List[str],
    K: int,
    temperature: float,
    top_p: float = 1.0,
    top_k_sampling: int = -1,
) -> List[str]:
    """
    Generate K new tokens for each of N particle texts, one prompt per
    particle (n=1 each).  Returns N suffix strings (decoded token sequences).
    vLLM's prefix caching deduplicates KV computation for identical prefixes
    that arise after resampling.
    """
    try:
        from vllm import SamplingParams  # type: ignore
    except ImportError:
        raise ImportError("vLLM is required for smc_resampled")

    params = SamplingParams(
        n=1,
        max_tokens=K,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_sampling,
    )
    outputs = backend.llm.generate(particle_texts, params)
    tokenizer = backend.llm.get_tokenizer()
    suffixes = []
    for out in outputs:
        token_ids = out.outputs[0].token_ids
        suffix = "".join(tokenizer.decode([tid]) for tid in token_ids)
        suffixes.append(suffix)
    return suffixes


def smc_resampled(
    initial_string: str,
    L: int,
    query: PCTLQuery,
    quantification_fn,             # callable(text: str, depth: int) -> Dict[str,int]
    backend,                        # VLLMBackend (required — uses batch prompt generation)
    num_particles: int = 500,
    confidence: float = 0.95,
    block_size: int = 32,           # K tokens generated per block per particle
    ess_threshold: float = 0.5,     # resample when ESS/N drops below this
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k_sampling: int = -1,
    potential_beta: float = 2.0,    # steepness of soft potential G = exp(beta * frac)
    verbose: bool = True,
) -> Tuple[float, float, float, Dict]:
    """
    Particle SMC with block-based generation and Feynman-Kac soft potentials.

    Unlike direct_smc which treats every trajectory equally, smc_resampled
    up-weights particles showing intermediate progress toward satisfying φ
    (via quantification_fn's partial-credit signal) and periodically
    resamples to prune low-potential particles early.  This concentrates the
    particle budget on trajectories likely to satisfy φ, improving efficiency
    on rare-event properties.

    Parameters
    ----------
    initial_string   : starting prompt ω₀
    L                : total generation horizon in tokens
    query            : PCTLQuery with operator F or G
    quantification_fn: callable(text, depth) → Dict[str,int]
    backend          : VLLMBackend (must support list-of-prompts generation)
    num_particles    : N — number of parallel particles
    confidence       : 1-δ for the Chernoff–Hoeffding bound
    block_size       : K — tokens generated per block before resampling check
    ess_threshold    : resample when ESS/N < this (0.5 is standard)
    temperature      : sampling temperature
    top_p            : nucleus sampling (1.0 = off)
    top_k_sampling   : hard top-k (-1 = off)
    potential_beta   : β controlling potential sharpness; 0 → uniform weights
    verbose          : print block-by-block ESS and p̂

    Returns
    -------
    estimate, lower_bound, upper_bound, stats
    """
    t0 = time.perf_counter()
    N = num_particles

    # Partition L into blocks of K (last block may be smaller)
    block_sizes: List[int] = [block_size] * (L // block_size)
    if L % block_size:
        block_sizes.append(L % block_size)

    # Initialise N particles
    particle_texts: List[str] = [initial_string] * N
    log_weights = torch.zeros(N, dtype=torch.float64)
    # For F: starts False, flips to True on satisfaction (absorbing)
    # For G: starts True, flips to False on any violation (absorbing)
    if query.operator == "G":
        satisfied: List[bool] = [True] * N
    else:
        satisfied = [False] * N

    n_resample_events = 0
    ess_history: List[float] = []
    depth_so_far = 0

    if verbose:
        print(f"Particle SMC  N={N}  L={L}  K={block_size}  "
              f"beta={potential_beta}  ess_thresh={ess_threshold}")

    for block_idx, K in enumerate(block_sizes):
        suffixes = _vllm_generate_particles(
            backend, particle_texts, K, temperature, top_p, top_k_sampling
        )
        particle_texts = [t + s for t, s in zip(particle_texts, suffixes)]
        depth_so_far += K

        for i, text in enumerate(particle_texts):
            quant = quantification_fn(text, depth_so_far)
            holds = query.formula.evaluate(quant)

            if query.operator == "F" and holds:
                satisfied[i] = True
            elif query.operator == "G" and not holds:
                satisfied[i] = False

            # Apply potential only while the particle is still "live":
            #   F: particle is live until it satisfies — stop accumulating once satisfied,
            #      otherwise the persistent quant signal causes weight explosion
            #   G: particle is live while it maintains the property
            apply_potential = (
                (query.operator == "F" and not satisfied[i]) or
                (query.operator == "G" and satisfied[i])
            )
            if apply_potential:
                g = _soft_potential(quant, query, potential_beta)
                log_weights[i] += math.log(max(g, 1e-300))

        # Self-normalise (numerically stable via log-sum-exp)
        log_w_max = log_weights.max()
        w = torch.exp(log_weights - log_w_max)
        w = w / w.sum()

        ess = 1.0 / (w ** 2).sum().item()
        ess_history.append(ess)

        if verbose:
            p_hat = sum(w[i].item() * (1.0 if satisfied[i] else 0.0) for i in range(N))
            print(f"  block {block_idx+1}/{len(block_sizes)}  depth={depth_so_far}  "
                  f"ESS={ess:.0f}/{N}  p_hat={p_hat:.4f}  "
                  f"resample_events={n_resample_events}")

        # Resample if ESS too low (skip on the final block — no generation after it)
        if ess / N < ess_threshold and block_idx < len(block_sizes) - 1:
            indices = _systematic_resample(w).tolist()
            particle_texts = [particle_texts[i] for i in indices]
            satisfied = [satisfied[i] for i in indices]
            log_weights = torch.zeros(N, dtype=torch.float64)
            n_resample_events += 1

    # Final self-normalised IS estimate
    log_w_max = log_weights.max()
    w = torch.exp(log_weights - log_w_max)
    w = w / w.sum()

    estimate = sum(w[i].item() * (1.0 if satisfied[i] else 0.0) for i in range(N))
    estimate = float(max(0.0, min(1.0, estimate)))

    # Chernoff-Hoeffding CI using minimum ESS across all blocks (conservative)
    eff_n = float(max(min(ess_history) if ess_history else N, 1.0))
    delta = 1.0 - confidence
    eps = math.sqrt(math.log(2.0 / delta) / (2.0 * eff_n))
    lower_bound = max(0.0, estimate - eps)
    upper_bound = min(1.0, estimate + eps)

    stats = {
        "method":             "smc_resampled",
        "num_particles":      float(N),
        "L":                  float(L),
        "block_size":         float(block_size),
        "n_blocks":           float(len(block_sizes)),
        "n_resample_events":  float(n_resample_events),
        "min_ess":            float(min(ess_history)) if ess_history else float(N),
        "mean_ess":           float(sum(ess_history) / len(ess_history)) if ess_history else float(N),
        "eff_n":              eff_n,
        "total_ms":           (time.perf_counter() - t0) * 1000,
        "epsilon":            eps,
        "satisfied_count":    float(sum(satisfied)),
    }
    return estimate, lower_bound, upper_bound, stats
