"""
Compare direct_smc vs smc_resampled on a subset of HarmBench behaviors.

Runs both estimators on the same 10 behaviors with matched sample budgets
and prints a side-by-side table showing p̂, 95% CI width, and ESS stats.

Usage
-----
    python run_smc_comparison.py
    python run_smc_comparison.py --model Qwen/Qwen2.5-7B-Instruct --N 300 --L 64
"""

import argparse
import json
import sys
import time
import warnings

sys.path.insert(0, "/home/parvk/smc-llms")
warnings.filterwarnings("ignore")

import torch

from gpu_llmchecker.backends import VLLMBackend
from gpu_llmchecker.benchmarks import HARMBENCH
from gpu_llmchecker.pctl import eventually
from gpu_llmchecker.quantification import MultiQuantifier, RefusalDetector, StepCounter
from gpu_llmchecker.verification import direct_smc, smc_resampled


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--N", type=int, default=300, help="Particles / samples per method")
    p.add_argument("--L", type=int, default=64,  help="Generation horizon (tokens)")
    p.add_argument("--K", type=int, default=32,  help="Block size for smc_resampled")
    p.add_argument("--beta", type=float, default=2.0, help="Soft potential steepness")
    p.add_argument("--gpu_util", type=float, default=0.85)
    p.add_argument("--out", default="results_smc_comparison.json")
    return p.parse_args()


def apply_chat_template(prompt, tokenizer):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return prompt


def main():
    args = parse_args()

    print(f"Loading {args.model}...")
    backend = VLLMBackend(args.model, dtype="bfloat16",
                          gpu_memory_utilisation=args.gpu_util, tensor_parallel_size=1)
    tokenizer = backend.llm.get_tokenizer()
    print("Model loaded.\n")

    # Use one behavior per category (6 behaviors total — one from each HARMBENCH category)
    categories_seen = set()
    subset = []
    for behavior, category in HARMBENCH:
        if category not in categories_seen:
            subset.append((behavior, category))
            categories_seen.add(category)
        if len(subset) == 6:
            break

    query = eventually("refusal", ">=", 100)
    quant = MultiQuantifier([RefusalDetector(), StepCounter()])
    quant_fn = lambda s, d, q=quant: q(s, d)

    rows = []
    print(f"{'#':<3} {'Category':<30} {'direct_smc':^22} {'smc_resampled':^28}")
    print(f"{'':3} {'':30} {'p̂':>6} {'CI':^14} {'ms':>5}  "
          f"{'p̂':>6} {'CI':^14} {'ESS_min':>7} {'ms':>6}")
    print("-" * 100)

    for idx, (behavior, category) in enumerate(subset):
        formatted = apply_chat_template(behavior, tokenizer)

        # ── direct_smc ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        p_d, lo_d, hi_d, stats_d = direct_smc(
            formatted, args.L, query, quant_fn, backend,
            num_samples=args.N, chunk_size=args.N, verbose=False,
        )
        ms_d = (time.perf_counter() - t0) * 1000

        # ── smc_resampled ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        p_r, lo_r, hi_r, stats_r = smc_resampled(
            formatted, args.L, query, quant_fn, backend,
            num_particles=args.N, block_size=args.K, potential_beta=args.beta,
            ess_threshold=0.5, verbose=False,
        )
        ms_r = (time.perf_counter() - t0) * 1000

        ess_min = stats_r["min_ess"]
        print(f"{idx+1:<3} {category:<30} "
              f"{p_d:>6.3f} [{lo_d:.3f},{hi_d:.3f}] {ms_d:>6.0f}  "
              f"{p_r:>6.3f} [{lo_r:.3f},{hi_r:.3f}] {ess_min:>7.0f} {ms_r:>7.0f}")
        print(f"    {behavior[:95]}")

        rows.append(dict(
            behavior=behavior, category=category,
            direct=dict(p=round(p_d,4), lo=round(lo_d,4), hi=round(hi_d,4),
                        ci_width=round(hi_d-lo_d,4), ms=round(ms_d,1)),
            resampled=dict(p=round(p_r,4), lo=round(lo_r,4), hi=round(hi_r,4),
                           ci_width=round(hi_r-lo_r,4), ms=round(ms_r,1),
                           min_ess=round(ess_min,1),
                           mean_ess=round(stats_r["mean_ess"],1),
                           n_resample_events=int(stats_r["n_resample_events"])),
        ))

    print("-" * 100)
    avg_ci_d = sum(r["direct"]["ci_width"] for r in rows) / len(rows)
    avg_ci_r = sum(r["resampled"]["ci_width"] for r in rows) / len(rows)
    avg_ms_d = sum(r["direct"]["ms"] for r in rows) / len(rows)
    avg_ms_r = sum(r["resampled"]["ms"] for r in rows) / len(rows)
    print(f"{'AVERAGES':<34} CI_width={avg_ci_d:.3f}              "
          f"CI_width={avg_ci_r:.3f}           ms_ratio={avg_ms_r/avg_ms_d:.2f}x")

    result = dict(
        model=args.model, N=args.N, L=args.L, K=args.K, beta=args.beta,
        avg_ci_direct=round(avg_ci_d, 4),
        avg_ci_resampled=round(avg_ci_r, 4),
        rows=rows,
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
