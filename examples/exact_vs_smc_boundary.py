"""
Exact-vs-SMC boundary grid study.

Systematically explores the boundary where exact verification becomes
intractable and Direct SMC takes over, across:

  L ∈ {4, 6, 8, 10}     (lookahead depth)
  k ∈ {3, 5, 10}         (branching factor)
  α ∈ {0.80, 0.90, 0.95} (probability threshold)

For each (L, k, α) cell the script:
  1. Runs BFS + exact backward induction (or reports OOM / timeout)
  2. Runs Direct SMC with M = args.samples
  3. When both succeed, computes the absolute error |exact - SMC|

The resulting table shows:
  |S| (tree size), encode time, verify time, exact p, SMC p̂, |error|

This validates SMC accuracy when ground truth is known, and identifies
the precise parameters at which exact verification crosses the feasibility
threshold.

Usage
-----
    python examples/exact_vs_smc_boundary.py
    python examples/exact_vs_smc_boundary.py --vllm --samples 2000
    python examples/exact_vs_smc_boundary.py --L-values 4 6 8 --k-values 3 5
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpu_llmchecker import (
    MultiQuantifier, GenderBias, StepCounter,
    build_dtmc_bfs, exact_backward_induction, direct_smc,
)
from gpu_llmchecker.pctl import eventually

START_STRING = "The player won because"
QUERY        = eventually("gender", ">", 0)
QUANT        = MultiQuantifier([GenderBias(), StepCounter()])

TIMEOUT_S = 120   # seconds before marking exact as infeasible
MAX_STATES = 5_000_000


def make_backend(model: str, use_vllm: bool, k_max: int):
    if use_vllm:
        from gpu_llmchecker.backends import VLLMBackend
        return VLLMBackend(model, max_logprobs=max(k_max, 16),
                           enable_prefix_caching=True)
    from gpu_llmchecker.backends import HFBackend
    return HFBackend(model, batch_size=16)


def run_exact_safe(backend, alpha, k, L, device):
    """Run exact verification; return None on timeout / OOM / state explosion."""
    try:
        t0 = time.perf_counter()
        levels, stats = build_dtmc_bfs(
            initial_string=START_STRING, L=L, alpha=alpha, k=k,
            quantification_fn=lambda s, d: QUANT(s, d),
            llm_backend=backend, verbose=False,
        )
        if stats["total_nodes"] > MAX_STATES:
            return None, int(stats["total_nodes"]), None, None
        prob, _ = exact_backward_induction(levels, QUERY, device=device)
        elapsed = time.perf_counter() - t0
        return prob, int(stats["total_nodes"]), stats["encoding_time_s"], elapsed
    except (MemoryError, RuntimeError, Exception):
        return None, None, None, None


def run_smc_safe(backend, L, num_samples):
    try:
        p, lo, hi, stats = direct_smc(
            initial_string=START_STRING, L=L, query=QUERY,
            quantification_fn=lambda s, d: QUANT(s, d),
            llm_backend=backend, num_samples=num_samples,
            verbose=False,
        )
        return p, lo, hi, stats["epsilon"], stats["total_ms"] / 1000
    except Exception:
        return None, None, None, None, None


def run(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    k_values = args.k_values
    L_values = args.L_values
    alpha_values = args.alpha_values

    print(f"\ngpu_llmchecker — Exact vs. SMC Boundary Grid Study")
    print(f"  Model   : {args.model}  device={device}")
    print(f"  Start   : \"{START_STRING}\"")
    print(f"  Query   : {QUERY}")
    print(f"  L grid  : {L_values}")
    print(f"  k grid  : {k_values}")
    print(f"  α grid  : {alpha_values}")
    print(f"  SMC M   : {args.samples}")
    print()

    backend = make_backend(args.model, args.vllm, max(k_values))

    rows = []
    header = ("α", "k", "L", "|S|", "ET(s)", "VT(s)", "p_exact",
              "p_smc", "CI_lo", "CI_hi", "ε", "abs_err", "smc_time_s")

    print(f"{'α':>6} {'k':>4} {'L':>4} {'|S|':>10} {'ET(s)':>8} {'VT(s)':>8} "
          f"{'p_exact':>9} {'p_smc':>8} {'|err|':>8} {'smc_t':>7}")
    print("-" * 80)

    for alpha in alpha_values:
        for k in k_values:
            for L in L_values:
                p_exact, n_states, enc_t, total_t = run_exact_safe(
                    backend, alpha, k, L, device
                )
                p_smc, ci_lo, ci_hi, eps, smc_t = run_smc_safe(
                    backend, L, args.samples
                )

                abs_err = (abs(p_exact - p_smc)
                           if p_exact is not None and p_smc is not None
                           else None)

                exact_str = f"{p_exact:.4f}" if p_exact is not None else "INFEASIBLE"
                smc_str   = f"{p_smc:.4f}"   if p_smc   is not None else "ERROR"
                err_str   = f"{abs_err:.4f}"  if abs_err  is not None else "—"
                states_str = f"{n_states:,}"  if n_states is not None else "OOM"
                enc_str   = f"{enc_t:.2f}"    if enc_t    is not None else "—"
                vt_str    = f"{total_t:.4f}"  if total_t  is not None else "—"
                smc_t_str = f"{smc_t:.1f}"    if smc_t    is not None else "—"

                print(f"{alpha:>6.2f} {k:>4} {L:>4} {states_str:>10} {enc_str:>8} "
                      f"{vt_str:>8} {exact_str:>9} {smc_str:>8} {err_str:>8} {smc_t_str:>7}")

                rows.append({
                    "alpha": alpha, "k": k, "L": L,
                    "n_states":  n_states if n_states is not None else "OOM",
                    "encode_s":  enc_t    if enc_t    is not None else "INFEASIBLE",
                    "verify_s":  total_t  if total_t  is not None else "INFEASIBLE",
                    "p_exact":   p_exact  if p_exact  is not None else "INFEASIBLE",
                    "p_smc":     p_smc    if p_smc    is not None else "ERROR",
                    "ci_lo":     ci_lo    if ci_lo    is not None else "",
                    "ci_hi":     ci_hi    if ci_hi    is not None else "",
                    "epsilon":   eps      if eps      is not None else "",
                    "abs_error": abs_err  if abs_err  is not None else "",
                    "smc_time_s": smc_t   if smc_t    is not None else "",
                })

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "exact_vs_smc_grid.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")

    # ── Summary: cases where both methods ran ─────────────────────────────
    both = [r for r in rows
            if isinstance(r["p_exact"], float) and isinstance(r["p_smc"], float)]
    if both:
        errs = [float(r["abs_error"]) for r in both]
        print(f"\nSMC accuracy when exact is feasible ({len(both)} cases):")
        print(f"  Mean |error| = {sum(errs)/len(errs):.4f}")
        print(f"  Max  |error| = {max(errs):.4f}")
        print(f"  Min  |error| = {min(errs):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact vs. SMC boundary grid")
    parser.add_argument("--model",     default="gpt2")
    parser.add_argument("--vllm",      action="store_true")
    parser.add_argument("--samples",   type=int, default=1000)
    parser.add_argument("--L-values",  type=int, nargs="+",
                        default=[4, 6, 8, 10], dest="L_values")
    parser.add_argument("--k-values",  type=int, nargs="+",
                        default=[3, 5, 10], dest="k_values")
    parser.add_argument("--alpha-values", type=float, nargs="+",
                        default=[0.80, 0.90, 0.95], dest="alpha_values")
    args = parser.parse_args()
    run(args)
