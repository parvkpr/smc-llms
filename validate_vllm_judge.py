"""Byte-equivalence check: VLLMQwenLegacyJudge vs QwenLegacyJudge on real leaves.

Pulls N (behavior, response) pairs from a TAP run, scores each with both
the HF-pipeline-backed QwenLegacyJudge and the vLLM-backed VLLMQwenLegacyJudge,
and reports the disagreement rate. The two judges must agree on >=98% of
labels for us to trust the vLLM judge as a drop-in replacement.

Usage
-----
    python validate_vllm_judge.py \\
        --tap-json results/judge_swap/multiseed/tap_llama_qwen_seed42_first10.json \\
        --n 60 --hf-device cuda:0 --vllm-device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def gather_items(path: str, n: int):
    with open(path) as f:
        d = json.load(f)
    out = []
    for rec in d["results"]:
        behavior = rec.get("goal") or ""
        for node in rec.get("records", []):
            resp = node.get("response") or ""
            if behavior and resp:
                out.append({
                    "behavior": behavior,
                    "response": resp,
                    "_tap_score": int(node.get("regular_score") or 0),
                })
            if len(out) >= n:
                return out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tap-json", required=True)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--hf-device", default="cuda:0")
    p.add_argument("--vllm-device", default="cuda:1")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = p.parse_args()

    items = gather_items(args.tap_json, args.n)
    print(f"[validate] loaded {len(items)} (behavior, response) pairs", flush=True)

    # ── HF judge ──────────────────────────────────────────────────────────
    from judges import QwenLegacyJudge
    print(f"[validate] loading HF QwenLegacyJudge on {args.hf_device} ...", flush=True)
    t0 = time.perf_counter()
    hf_judge = QwenLegacyJudge(device=args.hf_device, model_id=args.judge_model)
    hf_judge.load()
    print(f"[validate] HF judge loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    hf_scores = hf_judge.score_responses_int10(items)
    hf_time = time.perf_counter() - t0
    print(f"[validate] HF judge: scored {len(items)} items in {hf_time:.1f}s "
          f"({1000*hf_time/len(items):.0f}ms/item)", flush=True)
    # Free HF judge
    del hf_judge
    import torch
    torch.cuda.empty_cache()

    # ── vLLM judge ────────────────────────────────────────────────────────
    vllm_idx = args.vllm_device.split(":")[-1] if ":" in args.vllm_device else args.vllm_device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_idx)
    print(f"[validate] loading vLLM judge on physical cuda:{vllm_idx} "
          f"(CUDA_VISIBLE_DEVICES={vllm_idx}) ...", flush=True)
    from judges import VLLMQwenLegacyJudge
    t0 = time.perf_counter()
    v_judge = VLLMQwenLegacyJudge(
        device="cuda:0",  # remapped via CUDA_VISIBLE_DEVICES
        model_id=args.judge_model,
        vllm_kwargs=dict(
            gpu_memory_utilization=0.80,
            max_model_len=2048,
            enforce_eager=True,
        ),
    )
    v_judge.load()
    print(f"[validate] vLLM judge loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    v_scores = v_judge.score_responses_int10(items)
    v_time = time.perf_counter() - t0
    print(f"[validate] vLLM judge: scored {len(items)} items in {v_time:.1f}s "
          f"({1000*v_time/len(items):.0f}ms/item)", flush=True)

    # ── Compare ───────────────────────────────────────────────────────────
    print("\n=== AGREEMENT ===")
    agree = sum(1 for a, b in zip(hf_scores, v_scores) if a == b)
    print(f"  {agree}/{len(items)} = {100*agree/len(items):.1f}% agreement")

    # Confusion matrix
    hf10 = sum(1 for s in hf_scores if s == 10)
    v10 = sum(1 for s in v_scores if s == 10)
    print(f"  HF judge: {hf10} 'yes' / {len(items)-hf10} 'no'")
    print(f"  vLLM judge: {v10} 'yes' / {len(items)-v10} 'no'")

    disagreements = [
        (i, items[i]["behavior"][:50], items[i]["response"][:80], hf_scores[i], v_scores[i])
        for i in range(len(items))
        if hf_scores[i] != v_scores[i]
    ]
    if disagreements:
        print(f"\n=== DISAGREEMENTS ({len(disagreements)}) ===")
        for (i, b, r, h, v) in disagreements[:10]:
            print(f"  #{i}  hf={h:2d}  vllm={v:2d}  behav={b!r}  response={r!r}")
    print(f"\n=== SPEEDUP ===")
    print(f"  HF judge:   {hf_time:.1f}s")
    print(f"  vLLM judge: {v_time:.1f}s")
    print(f"  speedup:    {hf_time/v_time:.1f}x")


if __name__ == "__main__":
    main()
