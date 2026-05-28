# Recovery notes — 2026-05-27 file loss

## What happened

Between `22:14:58 UTC` and `22:59 UTC` on 2026-05-27, a bulk file operation
rewrote mtimes on the tracked-or-was-tracked sources in `/home/parvk/smc-llms/`
in three sub-50 ms bursts (signature of a `tar -x`, `cp -r`, `git restore`, or
`rsync`). At the same time, several **untracked attack-pipeline scripts were
deleted** from disk. We don't know which process did it (only `parvk` was
logged in, no `cron`, no reboot, `git stash list` empty), but the timing
coincides with activity in the Cursor server log directory
`~/.cursor-server/data/logs/20260527T155204/`.

## What was lost (irrecoverable from this machine)

| File                          | .pyc cache? | Status |
|-------------------------------|-------------|--------|
| `tap_bca_batch.py`            | no          | **gone** — was run as a script (never imported), so Python never cached its bytecode |
| `pair_bca_batch.py`           | yes (3.10)  | `__pycache__/pair_bca_batch.cpython-310.pyc` present |
| `pair_bca_llama.py`           | yes (3.10, 3.13) | `__pycache__/pair_bca_llama.cpython-310.pyc` present |
| `judges.py`                   | yes (3.10)  | `__pycache__/judges.cpython-310.pyc` present, **but rewritten fresh** |
| `judge_audit.py`              | no          | gone |
| `mass_vs_asr.py`              | no          | gone |
| `analyze_metric_validity.py`  | no          | gone |

`decompyle3` and `uncompyle6` do **not** support Python 3.10 bytecode, so the
`.cpython-310.pyc` caches above cannot be auto-decompiled in this environment.
The only practical decompilers for Python 3.10 are `pycdc` (build from source)
and `pylingual.io` (web). We have **not** attempted those because the user
opted to rewrite from scratch.

## What was saved

| Category | Files / paths |
|---|---|
| Cross-judge in-loop TAP runs | `results/judge_swap/tap_llama_{qwen,harmbench,llamaguard}.json` |
| Multi-seed TAP Qwen-pruning | `results/judge_swap/multiseed/tap_llama_qwen_seed{42,43}.json` (seeds 44/45/46 never ran) |
| Original 19-behavior TAP run (Figure 7 source) | `results/tap_bca_llama.json` |
| All PAIR-q10 / PAIR-q3 / template-search outputs | `results/pair_bca_*.json`, `results/template_search_*.json` |
| Per-target metric-validity numbers | `results/mass_vs_asr_*.json` |
| The main `gpu_llmchecker` library and verification kernels | tracked in git; untouched |

## What was rewritten in this session

| File | Purpose |
|---|---|
| `judges.py` | Unified harm-judge interface (`QwenJudge`, `QwenLegacyJudge`, `HarmBenchJudge`, `LlamaGuard3Judge`) plus `build_judge(name, device)` factory. Importable from downstream batch scripts. |
| `analyze_tap_multiseed.py` | Reads any number of TAP-batch JSONs, computes per-method judge-ASR (any-leaf-success semantics, the standard ASR convention), and reports pooled mean ± 95% CI for Δ(hybrid − regular). |

## Reproducing the multi-seed bonus-result analysis

```bash
cd /home/parvk/smc-llms

# Recompute the table that's pasted into the HTML "Update" section:
python analyze_tap_multiseed.py \
  --inputs results/judge_swap/tap_llama_qwen.json \
           results/judge_swap/tap_llama_harmbench.json \
           results/judge_swap/tap_llama_llamaguard.json \
           results/judge_swap/multiseed/tap_llama_qwen_seed42.json \
           results/judge_swap/multiseed/tap_llama_qwen_seed43.json \
  --labels "original (Qwen, 30 beh)" \
           "Axis1 HarmBench-pruning" \
           "Axis1 LlamaGuard-pruning" \
           "multiseed seed=42 (Qwen)" \
           "multiseed seed=43 (Qwen)" \
  --pool \
  --json-out results/judge_swap/multiseed/summary.json
```

Expected last rows:

    pooled[regular]  mean= 69.3%  stdev=16.19pp  se= 7.24pp  n_runs=5
    pooled[    bca]  mean= 67.3%  stdev=17.64pp  se= 7.89pp  n_runs=5
    pooled[ hybrid]  mean= 71.9%  stdev=15.63pp  se= 6.99pp  n_runs=5
    pooled[Δ(h-r)]   mean=+2.59pp  stdev= 9.46pp  se= 4.23pp  95%CI=± 8.29pp  n_runs=5

The pooled 95% CI **contains zero**, so the original +13.3 pp / +15.0 pp
single-seed observation in Figure 7 is not statistically robust on the data
that survived. The HTML "Update" section in `gpu_llmchecker/index.html`
records this honestly.

## To re-run seeds 44/45/46

We would need to either:

1. Restore `tap_bca_batch.py` from a backup outside this machine (the
   bash-history-mentioned commands suggest the user has run it many times,
   so it might live in another workspace clone, or on the upstream remote).
2. Decompile one of the cached `.pyc` files of a *related* attack pipeline
   (`pair_bca_batch.cpython-310.pyc`) using `pycdc` and adapt the TAP control
   flow from the JSON outputs we still have. This is feasible but a few
   hundred lines of careful work.
3. Re-implement `tap_bca_batch.py` from scratch, reusing `judges.py` (above)
   and the `gpu_llmchecker` library's `build_dtmc_bfs` /
   `exact_backward_induction_semantic` for the BCA computation.

Given that the 3 Qwen seeds we already have show Δ ∈ {−3.7, 0, +21.2} pp with
a pooled 95% CI of ±12.4 pp (zero is well inside), three more seeds are
unlikely to flip the qualitative conclusion. The honest reading is already in
the HTML; the rerun is a nice-to-have rather than a load-bearing experiment.
