"""
HarmBench compliance evaluation via cais/HarmBench-Mistral-7b-val-cls.

For each of 30 HARMBENCH behaviors this script estimates:
    P(F harm_compliant >= 100)
i.e. the probability that a model's L-token completion is judged *actually
compliant* with the harmful behavior by the HarmBench classifier — not merely
whether it avoids refusal keywords.

The classifier fires only at depth=L (fire_at_depth), so we pay exactly
M classifier calls per behavior (not L×M).

Memory layout (two RTX 5090s, 32 GB each):
  GPU 0 — target model via vLLM (gpu_memory_utilization=0.45 ≈ 14.7 GB)
  GPU 1 — HarmBench-Mistral-7b-val-cls via HF Transformers (≈ 14 GB)

Usage
-----
    python run_harmbench_cls.py
    python run_harmbench_cls.py --model Qwen/Qwen2.5-7B-Instruct
    python run_harmbench_cls.py --samples 100 --L 64
    python run_harmbench_cls.py --cls_device cuda:0 --vllm_gpu_util 0.45
"""

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, "/home/parvk/smc-llms")
warnings.filterwarnings("ignore")

from gpu_llmchecker.backends import VLLMBackend
from gpu_llmchecker.benchmarks import HARMBENCH
from gpu_llmchecker.pctl import eventually
from gpu_llmchecker.quantification import HarmBenchClassifier, MultiQuantifier, StepCounter
from gpu_llmchecker.verification import direct_smc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Target model HuggingFace id")
    p.add_argument("--samples", type=int, default=300,
                   help="Direct SMC trajectories per behavior (default: 300)")
    p.add_argument("--L", type=int, default=64,
                   help="Generation horizon in tokens (default: 64)")
    p.add_argument("--chunk", type=int, default=300,
                   help="vLLM batch size — set to --samples to batch all at once")
    p.add_argument("--vllm_gpu_util", type=float, default=0.45,
                   help="vLLM gpu_memory_utilization for target model on GPU 0 (default: 0.45)")
    p.add_argument("--cls_device", default="cuda:1",
                   help="Device for HarmBench classifier (default: cuda:1)")
    p.add_argument("--out", default="results_harmbench_cls.json",
                   help="Output JSON path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_chat_template(prompt: str, tokenizer) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / max(len(values) - 1, 1)
    return round(m, 4), round(var ** 0.5, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Device: GPU 0 (target, vLLM util={args.vllm_gpu_util})  "
          f"+ {args.cls_device} (HarmBench classifier)")
    print(f"Loading target model: {args.model}")
    backend = VLLMBackend(
        args.model,
        dtype="bfloat16",
        gpu_memory_utilisation=args.vllm_gpu_util,
        tensor_parallel_size=1,
    )
    tokenizer = backend.llm.get_tokenizer()

    print(f"Loading HarmBench classifier (cais/HarmBench-Mistral-7b-val-cls) "
          f"on {args.cls_device}...")
    from transformers import pipeline as hf_pipeline
    cls_pipe = hf_pipeline(
        "text-generation",
        model="cais/HarmBench-Mistral-7b-val-cls",
        device_map={"": args.cls_device},
        torch_dtype=torch.bfloat16,
    )
    print("Both models loaded.\n")

    query = eventually("harm_compliant", ">=", 100)

    rows = []
    by_category: Dict[str, List[float]] = defaultdict(list)

    print("=" * 72)
    print(f"HarmBench Compliance  (M={args.samples}, L={args.L})  →  "
          f"P(F harm_compliant)  [want LOW]")
    print(f"Target: {args.model}")
    print("=" * 72)

    for i, (behavior, category) in enumerate(HARMBENCH):
        formatted = apply_chat_template(behavior, tokenizer)

        clf = HarmBenchClassifier(
            behavior=behavior,
            fire_at_depth=args.L,
            pipeline=cls_pipe,
            initial_prompt=formatted,   # strip prompt prefix from generation text
        )
        quant = MultiQuantifier([clf, StepCounter()])
        quant_fn = lambda s, d, q=quant: q(s, d)

        t0 = time.perf_counter()
        p, lo, hi, _ = direct_smc(
            formatted, args.L, query, quant_fn, backend,
            num_samples=args.samples,
            chunk_size=args.chunk,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        row = dict(
            behavior=behavior,
            category=category,
            p=round(p, 4),
            lo=round(lo, 4),
            hi=round(hi, 4),
            ci_width=round(hi - lo, 4),
            time_s=round(elapsed, 1),
        )
        rows.append(row)
        by_category[category].append(p)

        print(f"  [{i+1:2d}/30] [{category:<28s}] "
              f"p={p:.3f} [{lo:.3f},{hi:.3f}]  {elapsed:.1f}s")
        print(f"         {behavior[:80]}")

    # Per-category summary
    print(f"\n  --- Category summary ---")
    cat_summary = {}
    for cat, ps in sorted(by_category.items()):
        m, s = mean_std(ps)
        print(f"    {cat:<32s}  mean={m:.3f}  std={s:.3f}  n={len(ps)}")
        cat_summary[cat] = dict(mean=m, std=s, n=len(ps))

    all_ps = [r["p"] for r in rows]
    overall_m, overall_s = mean_std(all_ps)
    print(f"\n  Overall P(harm_compliant): mean={overall_m:.3f}  std={overall_s:.3f}")

    result = dict(
        model=args.model,
        benchmark="HarmBench (P(F harm_compliant) via cais/HarmBench-Mistral-7b-val-cls)",
        n_behaviors=len(rows),
        L=args.L,
        num_samples=args.samples,
        overall_mean=overall_m,
        overall_std=overall_s,
        by_category=cat_summary,
        rows=rows,
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
