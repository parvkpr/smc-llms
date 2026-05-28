"""
budget_comparison.py
--------------------
Under a fixed attack-search budget, compare three template-selection strategies:

  1. Random     — pick one template uniformly at random (zero search cost)
  2. Empirical  — allocate B_search samples per template for ranking, pick the winner
  3. DTMC       — pick top-1 by Pr[F harm] (zero empirical search cost)

All three strategies are then evaluated on the same held-out pool of K completions
from the chosen template.  The question: for a given B_search, which strategy
finds the highest-ASR template?

Data requirements:
  - results/mass_vs_asr.json   with verdicts: List[bool] per DataPoint
  - results/template_search.json  with pr_harm per (behavior, template) pair

The simulation is Monte Carlo: for each B_search we repeat M=2000 trials.
Each trial independently sub-samples B_search verdicts per template for empirical
selection and K verdicts for evaluation, then reports the winner's eval ASR.

Usage
-----
    python budget_comparison.py
    python budget_comparison.py --K 50 --M 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ── Contaminated behaviors (same as mass_vs_asr.py) ───────────────────────────
CONTAMINATED = {
    "defamation", "ageism", "disable_antivirus", "ransomware", "keylogger_script",
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data(asr_path: str, ts_path: str):
    with open(asr_path) as f:
        asr_data = json.load(f)
    with open(ts_path) as f:
        ts_data = json.load(f)

    # Build pr_harm lookup: behavior_id → template_id → pr_harm
    pr_harm_map: Dict[str, Dict[str, float]] = {}
    for r in ts_data["results"]:
        bid = r["behavior_id"]
        if bid in CONTAMINATED:
            continue
        pr_harm_map[bid] = {t["template_id"]: t["pr_harm"] for t in r["template_results"]}

    # Group data points by behavior
    by_behavior: Dict[str, List[dict]] = defaultdict(list)
    for dp in asr_data["data_points"]:
        bid = dp["behavior_id"]
        if bid in CONTAMINATED:
            continue
        verdicts = dp.get("verdicts", [])
        if not verdicts:
            # Fall back to synthesising from n_harmful / n_samples (no simulation possible)
            continue
        pr_harm = pr_harm_map.get(bid, {}).get(dp["template_id"], 0.0)
        by_behavior[bid].append({
            "template_id":    dp["template_id"],
            "template_label": dp["template_label"],
            "pr_harm":        pr_harm,
            "verdicts":       verdicts,
            "true_asr":       dp["empirical_asr"],
        })

    return by_behavior


# ── Monte Carlo simulation ─────────────────────────────────────────────────────

def simulate(
    templates: List[dict],
    B_search: int,
    K: int,
    M: int,
    rng: random.Random,
) -> Tuple[float, float, float]:
    """
    Returns (mean_asr_random, mean_asr_empirical, mean_asr_dtmc) over M trials.

    templates  : list of dicts with keys "pr_harm" and "verdicts" (List[bool])
    B_search   : empirical search budget (samples per template for ranking)
    K          : evaluation budget (samples from chosen template to measure ASR)
    M          : number of Monte Carlo trials
    """
    n_templates = len(templates)
    dtmc_winner_idx = max(range(n_templates), key=lambda i: templates[i]["pr_harm"])

    random_asrs   = []
    empirical_asrs = []
    dtmc_asrs     = []

    for _ in range(M):
        # ── Random ────────────────────────────────────────────────────────────
        rand_idx = rng.randrange(n_templates)
        rand_pool = templates[rand_idx]["verdicts"]
        rand_sample = rng.choices(rand_pool, k=K)
        random_asrs.append(sum(rand_sample) / K)

        # ── Empirical search ──────────────────────────────────────────────────
        # For each template, draw B_search verdicts to estimate ASR; pick winner.
        search_asrs = []
        for t in templates:
            s = rng.choices(t["verdicts"], k=B_search)
            search_asrs.append(sum(s) / B_search)
        emp_winner_idx = max(range(n_templates), key=lambda i: search_asrs[i])
        emp_pool = templates[emp_winner_idx]["verdicts"]
        emp_sample = rng.choices(emp_pool, k=K)
        empirical_asrs.append(sum(emp_sample) / K)

        # ── DTMC (mass-guided) ────────────────────────────────────────────────
        dtmc_pool = templates[dtmc_winner_idx]["verdicts"]
        dtmc_sample = rng.choices(dtmc_pool, k=K)
        dtmc_asrs.append(sum(dtmc_sample) / K)

    return np.mean(random_asrs), np.mean(empirical_asrs), np.mean(dtmc_asrs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    base = os.path.dirname(os.path.abspath(__file__))
    asr_path = os.path.join(base, "results", "mass_vs_asr.json")
    ts_fname = "template_search_semantic.json" if args.semantic else "template_search.json"
    ts_path  = os.path.join(base, "results", ts_fname)

    by_behavior = load_data(asr_path, ts_path)
    n_behaviors = len(by_behavior)
    print(f"Loaded {n_behaviors} behaviors with verdict data")

    rng = random.Random(args.seed)
    budgets = [1, 2, 5, 10, 25, 50]

    print(f"\nBudget comparison  (K={args.K} eval samples, M={args.M} trials per cell)")
    print(f"{'B_search':>10}  {'Random':>10}  {'Empirical':>10}  {'DTMC':>10}  {'DTMC/Emp':>10}")
    print("-" * 58)

    rows = []
    for B in budgets:
        rand_all, emp_all, dtmc_all = [], [], []
        for bid, templates in by_behavior.items():
            if len(templates) < 2:
                continue
            r, e, d = simulate(templates, B_search=B, K=args.K, M=args.M, rng=rng)
            rand_all.append(r)
            emp_all.append(e)
            dtmc_all.append(d)

        mr = np.mean(rand_all)
        me = np.mean(emp_all)
        md = np.mean(dtmc_all)
        ratio = md / max(me, 1e-9)

        print(f"{B:>10}  {mr:>10.4f}  {me:>10.4f}  {md:>10.4f}  {ratio:>9.2f}×")
        rows.append({"B_search": B, "random": round(mr, 4), "empirical": round(me, 4),
                     "dtmc": round(md, 4), "dtmc_over_empirical": round(ratio, 3)})

    # ── Per-behavior table at B=10 ────────────────────────────────────────────
    B_show = 10
    print(f"\nPer-behavior at B_search={B_show}:")
    print(f"{'Behavior':<32}  {'Random':>8}  {'Empirical':>10}  {'DTMC':>8}  {'Lift':>8}")
    print("-" * 74)
    for bid, templates in sorted(by_behavior.items()):
        if len(templates) < 2:
            continue
        r, e, d = simulate(templates, B_search=B_show, K=args.K, M=args.M, rng=rng)
        lift = d / max(e, 1e-9)
        print(f"{bid[:32]:<32}  {r:>8.4f}  {e:>10.4f}  {d:>8.4f}  {lift:>7.2f}×")

    # ── Crossover point ───────────────────────────────────────────────────────
    print("\nCrossover: how many search samples/template does empirical need to match DTMC?")
    for row in rows:
        flag = " ← empirical beats DTMC" if row["empirical"] >= row["dtmc"] else ""
        print(f"  B={row['B_search']:>3}: empirical={row['empirical']:.4f}  dtmc={row['dtmc']:.4f}{flag}")

    out_fname = "budget_comparison_semantic.json" if args.semantic else "budget_comparison.json"
    out = os.path.join(base, "results", out_fname)
    with open(out, "w") as f:
        json.dump({"K": args.K, "M": args.M, "seed": args.seed, "rows": rows}, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget comparison: DTMC vs empirical vs random")
    parser.add_argument("--K",       type=int, default=50,   help="Eval samples from chosen template")
    parser.add_argument("--M",       type=int, default=2000, help="Monte Carlo trials per cell")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--semantic", action="store_true",
                        help="Use semantic Pr[F harm] from template_search_semantic.json")
    main(parser.parse_args())
