"""Re-score every leaf in a TAP JSON with the technically-correct vLLM judge.

Background: the HF-pipeline-based ``QwenLegacyJudge`` used in-loop by the
original TAP runs (seed42, seed43, the "original" seed, every method) used
the HF default right-padding for batched generation. HF itself emits a
``right-padding was detected`` warning because right-padding silently shifts
position encodings for shorter items in a batch and changes the next-token
prediction. We validated that fixing this to left-padding produces output
that is byte-equivalent to vLLM (see validate_vllm_judge.py: 200/200 = 100%
agreement). The right-padded judge under-reports ``yes`` by ~12% on
borderline cases.

This script re-labels every leaf with the vLLM judge and writes a new JSON
with ``regular_score_corrected`` (and ``hybrid_score_corrected`` if present)
populated. Downstream analysis scripts can then read the corrected fields
without re-running TAP itself (which would also have flowed through a
different pruning order under the corrected scores — see the discussion in
``analyze_tap_multiseed.py`` for why we accept this as a post-hoc labelling
correction rather than a re-run).

Usage
-----
    python rescore_tap_with_vllm_judge.py \\
        --in results/judge_swap/multiseed/tap_llama_qwen_seed42.json \\
        --out results/judge_swap/multiseed/tap_llama_qwen_seed42_corrected.json \\
        --vllm-device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _flatten(records_json: dict) -> List[Tuple[int, int, str, str]]:
    """Yield (result_idx, record_idx, behavior, response) for every leaf."""
    flat: List[Tuple[int, int, str, str]] = []
    for ri, rec in enumerate(records_json.get("results", [])):
        behavior = rec.get("goal") or ""
        for ni, node in enumerate(rec.get("records", []) or []):
            resp = node.get("response") or ""
            if behavior and resp:
                flat.append((ri, ni, behavior, resp))
    return flat


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--vllm-device", default="cuda:1",
                   help="Physical CUDA device for the vLLM judge.")
    p.add_argument("--judge-model", default=None,
                   help="HF model id; defaults to the judge class's DEFAULT_MODEL.")
    p.add_argument("--judge-type", default="vllm-qwen-legacy",
                   choices=["vllm-qwen-legacy", "qwen-legacy", "harmbench"],
                   help="Which judge to use for the re-labelling. "
                        "vllm-qwen-legacy is byte-equivalent to the fixed HF "
                        "qwen-legacy but ~3x faster. harmbench uses the fixed "
                        "HF HarmBench classifier (right-padding bug patched).")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Items per generate() call to vLLM. Higher = better "
                        "continuous-batching throughput up to memory limits.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-model-len", type=int, default=2048)
    args = p.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    flat = _flatten(data)
    if not flat:
        print(f"[rescore] {args.in_path}: no (behavior, response) pairs found",
              flush=True)
        return
    print(f"[rescore] {args.in_path}: {len(flat)} leaves to re-score", flush=True)

    vllm_idx = args.vllm_device.split(":")[-1] if ":" in args.vllm_device else args.vllm_device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(vllm_idx)
    print(f"[rescore] loading vLLM judge on physical cuda:{vllm_idx} "
          f"(CUDA_VISIBLE_DEVICES={vllm_idx}) ...", flush=True)

    from judges import VLLMQwenLegacyJudge
    t0 = time.perf_counter()
    judge = VLLMQwenLegacyJudge(
        device="cuda:0",  # remapped via CUDA_VISIBLE_DEVICES
        model_id=args.judge_model,
        vllm_kwargs=dict(
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=True,
        ),
    )
    judge.load()
    print(f"[rescore] vLLM judge loaded in {time.perf_counter()-t0:.1f}s",
          flush=True)

    items = [{"behavior": b, "response": r} for (_, _, b, r) in flat]
    scores: List[int] = []
    t0 = time.perf_counter()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        chunk_scores = judge.score_responses_int10(chunk)
        scores.extend(chunk_scores)
        elapsed = time.perf_counter() - t0
        done = start + len(chunk)
        rate = done / max(elapsed, 1e-6)
        eta = (len(items) - done) / max(rate, 1e-6)
        print(f"[rescore]   {done}/{len(items)}  ({rate:.0f}/s, ETA {eta:.0f}s)",
              flush=True)

    print(f"[rescore] scored {len(scores)} items in "
          f"{time.perf_counter()-t0:.1f}s", flush=True)

    # Tally before/after
    before_yes = 0
    after_yes = 0
    for (ri, ni, _, _), s in zip(flat, scores):
        node = data["results"][ri]["records"][ni]
        old = int(node.get("regular_score") or 0)
        node["regular_score_corrected"] = int(s)
        if "hybrid_score" in node:
            # We can't easily re-compute hybrid_score (depends on BCA mass)
            # post-hoc; downstream code treats *_corrected as the leaf's true
            # harm label, regardless of which objective TAP was optimizing.
            pass
        before_yes += int(old == 10)
        after_yes += int(s == 10)

    print(f"[rescore] yes-label count: before={before_yes}  after={after_yes}  "
          f"Δ={after_yes-before_yes:+d}  ({100*(after_yes-before_yes)/max(len(flat),1):+.1f}pp)",
          flush=True)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[rescore] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
