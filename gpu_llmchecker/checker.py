"""
LLMCheckerGPU — the main entry point.

Mirrors the LLMCHECKER workflow from the paper but replaces:
  - DFS recursive construction  →  BFS batched construction
  - PRISM serialisation + Storm →  GPU sparse backward induction
  - (optional) exact result     →  statistical MC fallback

Typical usage
-------------
    from gpu_llmchecker import LLMCheckerGPU
    from gpu_llmchecker.pctl import eventually
    from gpu_llmchecker.quantification import GenderBias, StepCounter, MultiQuantifier
    from gpu_llmchecker.backends import VLLMBackend

    backend = VLLMBackend("meta-llama/Llama-2-7b-hf")
    quantifier = MultiQuantifier([GenderBias(), StepCounter()])
    checker = LLMCheckerGPU(backend, quantifier)

    result = checker.check(
        start_string = "The player won because",
        query        = eventually("gender", ">", 0),
        alpha        = 0.8,
        k            = 15,
        L            = 5,
    )
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .bfs_builder import build_dtmc_bfs
from .dtmc import DTMCNode
from .pctl import PCTLQuery
from .quantification import MultiQuantifier, Quantifier
from .verification import exact_backward_induction, statistical_model_check


@dataclass
class CheckResult:
    """Return value of LLMCheckerGPU.check()."""

    query: PCTLQuery
    start_string: str
    alpha: float
    k: int
    L: int

    # Core result
    probability: float

    # Tree statistics
    num_states: int
    num_transitions: int

    # Timing (seconds)
    encoding_time_s: float
    verification_time_s: float

    # SMC-only fields (None for exact)
    smc_lower_bound: Optional[float] = None
    smc_upper_bound: Optional[float] = None
    smc_samples: Optional[int] = None
    smc_epsilon: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            f"Query      : {self.query}",
            f"Start      : '{self.start_string[:60]}'",
            f"α={self.alpha}  k={self.k}  L={self.L}",
            f"States     : {self.num_states:,}   Transitions: {self.num_transitions:,}",
            f"Encode     : {self.encoding_time_s:.2f}s",
            f"Verify     : {self.verification_time_s:.4f}s",
            f"Result     : {self.probability:.6f}",
        ]
        if self.smc_lower_bound is not None:
            lines.append(
                f"95% CI     : [{self.smc_lower_bound:.4f}, {self.smc_upper_bound:.4f}]"
                f"  ε={self.smc_epsilon:.4f}  (M={self.smc_samples})"
            )
        return "\n".join(lines)


class LLMCheckerGPU:
    """
    GPU-accelerated α-k-bounded PCTL model checker for LLM text generation.

    Parameters
    ----------
    llm_backend     : VLLMBackend or HFBackend
    quantifier      : Quantifier (or MultiQuantifier) instance
    device          : 'cuda' | 'cpu' (defaults to cuda if available)
    use_smc         : if True, use statistical MC instead of exact verification
    smc_samples     : number of samples for SMC
    smc_confidence  : confidence level for Chernoff bounds
    temperature     : LLM temperature (1.0 = unmodified distribution)
    """

    def __init__(
        self,
        llm_backend: Any,
        quantifier: Quantifier,
        device: Optional[str] = None,
        use_smc: bool = False,
        smc_samples: int = 2000,
        smc_confidence: float = 0.95,
        temperature: float = 1.0,
    ) -> None:
        self.backend = llm_backend
        self.quantifier = quantifier
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_smc = use_smc
        self.smc_samples = smc_samples
        self.smc_confidence = smc_confidence
        self.temperature = temperature

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        start_string: str,
        query: PCTLQuery,
        alpha: float,
        k: int,
        L: int,
        verbose: bool = True,
    ) -> CheckResult:
        """
        Run the full check pipeline and return a CheckResult.

        1. BFS construction with batched LLM inference
        2. Exact GPU backward induction  OR  statistical MC
        """
        # ── Step 1: build DTMC ────────────────────────────────────────────
        def quant_fn(text: str, depth: int) -> Dict:
            return self.quantifier(text, depth)

        levels, build_stats = build_dtmc_bfs(
            initial_string=start_string,
            L=L,
            alpha=alpha,
            k=k,
            quantification_fn=quant_fn,
            llm_backend=self.backend,
            temperature=self.temperature,
            verbose=verbose,
        )

        # ── Step 2: verify ────────────────────────────────────────────────
        import time
        t_verify = time.perf_counter()

        smc_lower = smc_upper = smc_eps = smc_n = None

        if self.use_smc:
            prob, smc_lower, smc_upper, smc_stats = statistical_model_check(
                levels=levels,
                query=query,
                num_samples=self.smc_samples,
                confidence=self.smc_confidence,
                device=self.device,
            )
            smc_n = self.smc_samples
            smc_eps = smc_stats["epsilon"]
        else:
            prob, _ = exact_backward_induction(
                levels=levels,
                query=query,
                device=self.device,
            )

        verify_time = time.perf_counter() - t_verify

        return CheckResult(
            query=query,
            start_string=start_string,
            alpha=alpha,
            k=k,
            L=L,
            probability=prob,
            num_states=int(build_stats["total_nodes"]),
            num_transitions=int(build_stats["total_transitions"]),
            encoding_time_s=build_stats["encoding_time_s"],
            verification_time_s=verify_time,
            smc_lower_bound=smc_lower,
            smc_upper_bound=smc_upper,
            smc_samples=smc_n,
            smc_epsilon=smc_eps,
        )

    def check_batch(
        self,
        experiments: List[Dict],
        verbose: bool = True,
    ) -> List[CheckResult]:
        """
        Run multiple check() calls sharing the same backend.

        Each dict in experiments must contain the keys:
            start_string, query, alpha, k, L
        and may optionally override: verbose.
        """
        results = []
        for exp in experiments:
            r = self.check(
                start_string=exp["start_string"],
                query=exp["query"],
                alpha=exp["alpha"],
                k=exp["k"],
                L=exp["L"],
                verbose=exp.get("verbose", verbose),
            )
            results.append(r)
        return results
