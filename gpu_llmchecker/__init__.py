"""
gpu_llmchecker
==============
GPU-parallel α-k-bounded PCTL model checking for LLM text generation.

Replaces the sequential DFS + Storm pipeline from LLMCHECKER (Gross et al.,
arXiv:2509.18836) with:
  - BFS batched construction (O(L) LLM calls instead of O(k^L))
  - vLLM PagedAttention + automatic prefix caching
  - GPU sparse backward induction for exact PCTL reachability
  - Statistical MC fallback for extreme scale

Quick start
-----------
    from gpu_llmchecker import LLMCheckerGPU
    from gpu_llmchecker.pctl import eventually
    from gpu_llmchecker.quantification import GenderBias, StepCounter, MultiQuantifier
    from gpu_llmchecker.backends import VLLMBackend

    backend  = VLLMBackend("google/gemma-2b-it")
    checker  = LLMCheckerGPU(backend, MultiQuantifier([GenderBias(), StepCounter()]))
    result   = checker.check("The player won because", eventually("gender", ">", 0),
                              alpha=0.8, k=15, L=5)
    print(result)
"""

from .checker import LLMCheckerGPU, CheckResult
from .pctl import PCTLQuery, AtomicProp, ConjunctiveProp, eventually, always, eventually_conj
from .quantification import (
    GenderBias,
    SentimentScore,
    ReadingQuality,
    CopyrightSim,
    StepCounter,
    MultiQuantifier,
)
from .bfs_builder import build_dtmc_bfs
from .verification import exact_backward_induction, statistical_model_check, direct_smc

__all__ = [
    "LLMCheckerGPU",
    "CheckResult",
    "PCTLQuery",
    "AtomicProp",
    "ConjunctiveProp",
    "eventually",
    "always",
    "eventually_conj",
    "GenderBias",
    "SentimentScore",
    "ReadingQuality",
    "CopyrightSim",
    "StepCounter",
    "MultiQuantifier",
    "build_dtmc_bfs",
    "exact_backward_induction",
    "statistical_model_check",
    "direct_smc",
]
