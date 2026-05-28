"""
generate_completions.py
-----------------------
For each model × behavior, generates:
  - the direct (baseline) greedy completion
  - the best-template greedy completion

Output: results/completions.json
  {
    "ModelShortName": {
      "behavior_id": {
        "direct": "<completion text>",
        "best":   "<completion text>",
        "best_template_id": "suffix_force"
      }, ...
    }, ...
  }
"""

from __future__ import annotations

import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from template_search import TEMPLATES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_PATH    = os.path.join(RESULTS_DIR, "completions.json")

MODEL_FILES = {
    "Qwen2.5-1.5B": "template_search_semantic_dtmc_qwen_qwen2.5_1.5b_instruct.json",
    "Qwen2.5-7B":   "template_search_semantic_dtmc_qwen_qwen2.5_7b_instruct.json",
    "Mistral-7B":   "template_search_semantic_dtmc_mistralai_mistral_7b_instruct_v0.2.json",
    "Llama-3.1-8B": "template_search_semantic_dtmc_meta_llama_llama_3.1_8b_instruct.json",
}
MODEL_IDS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-7B":   "Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7B":   "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
}

TEMPLATE_MAP = {t["id"]: t for t in TEMPLATES}

MAX_NEW_TOKENS = 300
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}", flush=True)


def build_prompt(model_id: str, tok, text: str) -> str:
    """Wrap text in the model's chat template and return the full prompt string."""
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return text


def generate_one(model, tok, prompt: str) -> str:
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def load_results(short_name: str):
    path = os.path.join(RESULTS_DIR, MODEL_FILES[short_name])
    with open(path) as f:
        return json.load(f)["results"]


def run_model(short_name: str, existing: dict) -> dict:
    model_id = MODEL_IDS[short_name]
    results_data = load_results(short_name)

    # Figure out what still needs generating
    done = existing.get(short_name, {})
    pending = [r for r in results_data if r["behavior_id"] not in done]

    if not pending:
        print(f"[{short_name}] All {len(results_data)} behaviors already done, skipping.")
        return done

    print(f"\n{'='*70}")
    print(f"[{short_name}] Loading model: {model_id}")
    tok   = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = {"torch_dtype": torch.bfloat16}
    if DEVICE.startswith("cuda"):
        load_kwargs["device_map"] = {"": DEVICE}
    else:
        load_kwargs["device_map"] = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print(f"[{short_name}] Loaded. Generating {len(pending)} behavior(s)...")

    out = dict(done)

    for i, r in enumerate(pending, 1):
        bid   = r["behavior_id"]
        goal  = r["goal"]
        best_id = r["best_template_id"]
        best_tmpl = TEMPLATE_MAP.get(best_id, TEMPLATE_MAP["direct"])

        direct_text  = goal
        best_text    = best_tmpl["wrap"](goal)

        t0 = time.perf_counter()
        direct_prompt = build_prompt(model_id, tok, direct_text)
        direct_comp   = generate_one(model, tok, direct_prompt)

        best_prompt   = build_prompt(model_id, tok, best_text)
        best_comp     = generate_one(model, tok, best_prompt)
        elapsed = time.perf_counter() - t0

        out[bid] = {
            "direct":           direct_comp,
            "best":             best_comp,
            "best_template_id": best_id,
        }

        print(f"  [{i:3d}/{len(pending)}] {bid[:40]:<40}  ({elapsed:.1f}s)")

        # Checkpoint every 10
        if i % 10 == 0:
            save(existing | {short_name: out})

    # Unload
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def save(data: dict):
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, OUT_PATH)
    print(f"  Saved → {OUT_PATH}")


def main():
    existing = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"Resuming from {OUT_PATH} ({sum(len(v) for v in existing.values())} completions already done)")

    for short_name in MODEL_IDS:
        model_out = run_model(short_name, existing)
        existing[short_name] = model_out
        save(existing)

    total = sum(len(v) for v in existing.values())
    print(f"\nDone. {total} completions saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
