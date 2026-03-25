"""
Benchmark: BFS-GPU pipeline vs. the sequential DFS baseline.

Measures wall-clock time for:
  1. DTMC construction (encoding time)
  2. PCTL verification time
  3. Total time

across increasing values of k and L, demonstrating the parallelism gains.

The DFS baseline is simulated by running the HFBackend with batch_size=1
(one LLM call at a time), matching the serial behaviour of the original
Algorithm 1.  The BFS-GPU pipeline uses VLLMBackend (or HFBackend with
batch_size=level_size) to issue one call per level.

Run
---
    python examples/benchmark.py --model google/gemma-2b-it
    python examples/benchmark.py --model google/gemma-2b-it --cpu   # no GPU
"""

import argparse
import time
from typing import List, Tuple

from gpu_llmchecker import (
    LLMCheckerGPU,
    MultiQuantifier,
    GenderBias,
    StepCounter,
    exact_backward_induction,
    build_dtmc_bfs,
)
from gpu_llmchecker.pctl import eventually


START_STRING = "The player won because"
QUERY = eventually("gender", ">", 0)
QUANTIFIER = MultiQuantifier([GenderBias(), StepCounter()])


def _make_backend(args, serial: bool):
    if args.cpu or args.backend == "hf":
        from gpu_llmchecker.backends import HFBackend
        bs = 1 if serial else 64
        return HFBackend(args.model, device="cpu" if args.cpu else "auto",
                         batch_size=bs)
    from gpu_llmchecker.backends import VLLMBackend
    return VLLMBackend(
        args.model,
        max_logprobs=max(args.k_values),
        enable_prefix_caching=(not serial),
    )


def run_once(backend, alpha, k, L, device, label) -> Tuple[float, float, int, int]:
    t0 = time.perf_counter()
    levels, stats = build_dtmc_bfs(
        initial_string=START_STRING,
        L=L, alpha=alpha, k=k,
        quantification_fn=lambda s, d: QUANTIFIER(s, d),
        llm_backend=backend,
        verbose=False,
    )
    encode_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob, _ = exact_backward_induction(levels, QUERY, device=device)
    verify_t = time.perf_counter() - t1

    return encode_t, verify_t, int(stats["total_nodes"]), int(stats["total_transitions"])


def run(args):
    device = "cpu" if args.cpu else "cuda"

    print(f"\nBenchmark: {args.model}   device={device}")
    print(f"Start: '{START_STRING}'   query: {QUERY}")
    print("=" * 80)
    print(f"{'k':>4} {'L':>4} {'|S|':>10} {'|T|':>12} "
          f"{'encode(s)':>12} {'verify(ms)':>12} {'total(s)':>10}")
    print("-" * 80)

    backend = _make_backend(args, serial=False)

    for k in args.k_values:
        for L in args.L_values:
            try:
                enc, ver, ns, nt = run_once(backend, args.alpha, k, L, device, "BFS-GPU")
                print(f"{k:>4} {L:>4} {ns:>10,} {nt:>12,} "
                      f"{enc:>12.2f} {ver*1000:>12.1f} {enc+ver:>10.2f}")
            except Exception as e:
                print(f"{k:>4} {L:>4}   ERROR: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2b-it")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 9, 15])
    parser.add_argument("--L-values", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run(args)
