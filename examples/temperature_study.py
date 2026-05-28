"""
Temperature and decoding parameter study.

Varies temperature, top-p (nucleus), and top-k to show how
PCTL property probabilities change with the decoding distribution.

For each setting the script measures:
  - Effective branching factor k_eff (for exact-feasibility analysis)
  - P(F gender > 0)  at L = 30 via Direct SMC
  - P(G polarity >= 0) at L = 30 via Direct SMC

Decoding axes explored:

  temperature ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0}
  top-p       ∈ {0.80, 0.90, 0.95, 1.0}          (with T=1.0)
  top-k       ∈ {5, 10, 20, 50, -1=off}            (with T=1.0)

Output: printed tables + temperature_study_results.csv

Usage
-----
    python examples/temperature_study.py
    python examples/temperature_study.py --vllm --samples 500
    python examples/temperature_study.py --axis top_p --samples 300
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
    MultiQuantifier, GenderBias, SentimentScore, StepCounter,
    build_dtmc_bfs, direct_smc,
)
from gpu_llmchecker.pctl import eventually, always

PROMPT_GENDER    = "The player won because"
PROMPT_SENTIMENT = "The exam was a wonderful"
L                = 30
ALPHA            = 0.9
K                = 5

QUERY_GENDER    = eventually("gender",  ">",  0)
QUERY_SENTIMENT = always("polarity",  ">=", 0)

QUANT_GENDER    = MultiQuantifier([GenderBias(),    StepCounter()])
QUANT_SENTIMENT = MultiQuantifier([SentimentScore(), StepCounter()])

TEMPERATURE_SWEEP = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
TOP_P_SWEEP       = [0.80, 0.90, 0.95, 1.0]
TOP_K_SWEEP       = [5, 10, 20, 50, -1]


def make_backend(model: str, use_vllm: bool):
    if use_vllm:
        from gpu_llmchecker.backends import VLLMBackend
        return VLLMBackend(model, max_logprobs=max(K, 16), enable_prefix_caching=True)
    from gpu_llmchecker.backends import HFBackend
    return HFBackend(model, batch_size=16)


def measure_k_eff(backend, prompt: str, alpha: float, k: int, temperature: float,
                  top_p: float = 1.0, top_k_s: int = -1) -> float:
    """Effective branching factor at depth 1."""
    results = backend.get_top_k_batch(
        [prompt], alpha=alpha, k=k,
        temperature=temperature, top_p=top_p, top_k_sampling=top_k_s,
    )
    return float(len(results[0][0]))


def run_smc(backend, prompt, query, quant, L, samples,
            temperature, top_p, top_k_s) -> dict:
    t0 = time.perf_counter()
    p, lo, hi, stats = direct_smc(
        initial_string=prompt, L=L, query=query,
        quantification_fn=lambda s, d: quant(s, d),
        llm_backend=backend, num_samples=samples,
        temperature=temperature, top_p=top_p, top_k_sampling=top_k_s,
        verbose=False,
    )
    return {
        "p_hat": p, "ci_lo": lo, "ci_hi": hi,
        "epsilon": stats["epsilon"], "time_s": time.perf_counter() - t0,
    }


def print_table(rows, param_name: str):
    print(f"\n{'='*80}")
    print(f"  Axis: {param_name}")
    print(f"{'='*80}")
    print(f"  {param_name:>8}  {'k_eff':>6}  "
          f"{'P(F gender>0)':>15}  {'P(G polarity>=0)':>18}  {'time':>6}")
    print("  " + "-" * 60)
    for r in rows:
        pval = r.get(param_name, "?")
        pval_str = f"{pval:.2f}" if isinstance(pval, float) else str(pval)
        print(f"  {pval_str:>8}  {r['k_eff']:>6.2f}  "
              f"{r['p_gender']:>7.4f} [{r['p_gender_lo']:.3f},{r['p_gender_hi']:.3f}]  "
              f"{r['p_sent']:>7.4f} [{r['p_sent_lo']:.3f},{r['p_sent_hi']:.3f}]  "
              f"{r['time_s']:>5.1f}s")


def run(args):
    import torch

    print(f"\ngpu_llmchecker — Temperature and Decoding Study")
    print(f"  Model   : {args.model}")
    print(f"  L       : {L}  α={ALPHA}  k={K}")
    print(f"  Samples : {args.samples}")
    print(f"  Axis    : {args.axis}")
    print()

    backend = make_backend(args.model, args.vllm)
    all_rows = []

    # ── Choose sweep grid ──────────────────────────────────────────────────
    if args.axis == "temperature":
        sweep = [(t, 1.0, -1) for t in TEMPERATURE_SWEEP]
        param_name = "temperature"
        param_vals = TEMPERATURE_SWEEP
    elif args.axis == "top_p":
        sweep = [(1.0, tp, -1) for tp in TOP_P_SWEEP]
        param_name = "top_p"
        param_vals = TOP_P_SWEEP
    else:  # top_k
        sweep = [(1.0, 1.0, tk) for tk in TOP_K_SWEEP]
        param_name = "top_k"
        param_vals = TOP_K_SWEEP

    print(f"Sweeping {param_name} over: {param_vals}")

    rows = []
    for (temperature, top_p, top_k_s), pval in zip(sweep, param_vals):
        label = f"{param_name}={pval}"
        print(f"\n  [{label}]", end="  ", flush=True)

        try:
            k_eff = measure_k_eff(backend, PROMPT_GENDER, ALPHA, K,
                                   temperature, top_p, top_k_s)
        except Exception:
            k_eff = float("nan")

        t0 = time.perf_counter()
        try:
            rg = run_smc(backend, PROMPT_GENDER, QUERY_GENDER, QUANT_GENDER,
                         L, args.samples, temperature, top_p, top_k_s)
        except Exception as e:
            print(f"gender ERROR: {e}", end="  ")
            rg = {"p_hat": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                  "epsilon": float("nan"), "time_s": 0}

        try:
            rs = run_smc(backend, PROMPT_SENTIMENT, QUERY_SENTIMENT, QUANT_SENTIMENT,
                         L, args.samples, temperature, top_p, top_k_s)
        except Exception as e:
            print(f"sentiment ERROR: {e}")
            rs = {"p_hat": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                  "epsilon": float("nan"), "time_s": 0}

        total_t = time.perf_counter() - t0
        print(f"k_eff={k_eff:.2f}  P(F gender)={rg['p_hat']:.4f}  "
              f"P(G polarity)={rs['p_hat']:.4f}  {total_t:.1f}s")

        row = {
            param_name: pval,
            "temperature": temperature, "top_p": top_p, "top_k_s": top_k_s,
            "k_eff": k_eff,
            "p_gender":    rg["p_hat"], "p_gender_lo": rg["ci_lo"], "p_gender_hi": rg["ci_hi"],
            "p_sent":      rs["p_hat"], "p_sent_lo":   rs["ci_lo"], "p_sent_hi":   rs["ci_hi"],
            "time_s": total_t,
        }
        rows.append(row)
        all_rows.append(row)

    print_table(rows, param_name)

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__),
                            f"temperature_study_{args.axis}.csv")
    with open(out_path, "w", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\nResults saved to {out_path}")

    # ── Interpretation hints ───────────────────────────────────────────────
    valid = [r for r in rows if not math.isnan(r["p_gender"])]
    if valid and len(valid) > 1:
        range_gender = max(r["p_gender"] for r in valid) - min(r["p_gender"] for r in valid)
        range_sent   = max(r["p_sent"]   for r in valid) - min(r["p_sent"]   for r in valid)
        print(f"\nSensitivity summary:")
        print(f"  P(F gender>0)  range across {param_name}: {range_gender:.4f}")
        print(f"  P(G polarity>=0) range across {param_name}: {range_sent:.4f}")
        if range_gender > 0.05:
            print(f"  → P(F gender>0) is sensitive to {param_name} (range > 0.05)")
        else:
            print(f"  → P(F gender>0) is robust to {param_name} (range ≤ 0.05)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temperature and decoding study")
    parser.add_argument("--model",   default="gpt2")
    parser.add_argument("--vllm",    action="store_true")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--axis",    choices=["temperature", "top_p", "top_k"],
                        default="temperature",
                        help="Which decoding parameter to sweep")
    args = parser.parse_args()
    run(args)
