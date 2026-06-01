#!/usr/bin/env python3
"""Build complementarity viz data + self-contained HTML from TAP-batch JSONs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
MULTISEED = ROOT / "results/judge_swap/multiseed"
OUT_DATA = MULTISEED / "tap_complementarity_data_seed44.json"
OUT_HTML_SEED44 = ROOT / "gpu_llmchecker/tap_complementarity_viz_seed44.html"
OUT_HTML = ROOT / "gpu_llmchecker/tap_complementarity_viz.html"

DATASETS = [
    ("seed44_l8", "Seed 44 · L=8 · vLLM · 30 beh", ["tap_llama_qwen_seed44.json"]),
    ("seed44_l12_fixed", "Seed 44 · L=12 fixed · 30 beh (clean regular + BCA judge-pick)", [
        "tap_llama_qwen_seed44_L12_fixed.json",
    ]),
    ("seed44_l12_old", "Seed 44 · L=12 old · 30 beh (confounded regular)", [
        "tap_llama_qwen_seed44_L12.json",
    ]),
    (
        "seed44_l12_100_regular_rich_bca",
        "Seed 44 · L=12 · 100 beh · regular+rich feedback + BCA judge-pick",
        ["tap_llama_qwen_seed44_L12_regular_rich_bca_100.json"],
    ),
]

OUTCOMES = ("both", "regular_only", "bca_only", "neither")
CAT_ORDER = (
    "harassment_discrimination",
    "malware_hacking",
    "physical_harm",
    "economic_harm",
    "fraud_deception",
    "disinformation",
    "sexual_adult_content",
    "privacy",
    "expert_advice",
    "government_decision-making",
)
CAT_LABELS = {
    "harassment_discrimination": "Harassment / discrimination",
    "malware_hacking": "Malware / hacking",
    "physical_harm": "Physical harm",
    "economic_harm": "Economic harm",
    "fraud_deception": "Fraud / deception",
    "disinformation": "Disinformation",
    "sexual_adult_content": "Sexual / adult content",
    "privacy": "Privacy",
    "expert_advice": "Expert advice",
    "government_decision-making": "Government decision-making",
}


def _deploy_regular(row: dict) -> bool:
    return float(row.get("best_objective", {}).get("regular_score") or 0) >= 10


def _deploy_bca(row: dict) -> bool:
    """Deploy success for BCA trees with judge-pick deploy (best_objective only)."""
    return _deploy_regular(row)


def _outcome(reg_ok: bool, bca_ok: bool) -> str:
    if reg_ok and bca_ok:
        return "both"
    if reg_ok:
        return "regular_only"
    if bca_ok:
        return "bca_only"
    return "neither"


def _deploy_prompt(row: dict) -> str:
    bo = row.get("best_objective") or {}
    return str(bo.get("prompt") or "")


def _any_leaf(row: dict) -> bool:
    return any(float(x.get("regular_score") or 0) >= 10 for x in row.get("records", []))


def _parse_seed(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def analyze_files(paths: List[Path], label: str, ds_id: str) -> Optional[dict]:
    by_key: Dict[Tuple[Optional[int], str], dict] = {}
    for fp in paths:
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        seed = _parse_seed(fp.name) or data.get("config", {}).get("seed")
        for row in data.get("results", []):
            if row.get("method") not in ("regular", "bca") or "error" in row:
                continue
            key = (seed, row["goal"])
            by_key.setdefault(key, {"goal": row["goal"], "category": row.get("category", "?"), "seed": seed})
            by_key[key][row["method"]] = row

    behaviors: List[dict] = []
    counts = {o: 0 for o in OUTCOMES}
    by_cat = {o: {} for o in OUTCOMES}

    for entry in by_key.values():
        if "regular" not in entry or "bca" not in entry:
            continue
        reg_row, bca_row = entry["regular"], entry["bca"]
        reg_ok, bca_ok = _deploy_regular(reg_row), _deploy_bca(bca_row)
        outcome = _outcome(reg_ok, bca_ok)
        counts[outcome] += 1
        cat = entry["category"]
        by_cat[outcome][cat] = by_cat[outcome].get(cat, 0) + 1
        behaviors.append(
            {
                "goal": entry["goal"],
                "category": cat,
                "seed": entry["seed"],
                "outcome": outcome,
                "regular_deploy": reg_ok,
                "bca_deploy": bca_ok,
                "regular_any_leaf": _any_leaf(reg_row),
                "bca_any_leaf": _any_leaf(bca_row),
                "regular_prompt": _deploy_prompt(reg_row)[:500],
                "bca_prompt": _deploy_prompt(bca_row)[:500],
                "same_prompt": _deploy_prompt(reg_row).strip() == _deploy_prompt(bca_row).strip(),
            }
        )

    if not behaviors:
        return None

    n = len(behaviors)
    union = counts["both"] + counts["regular_only"] + counts["bca_only"]
    return {
        "id": ds_id,
        "label": label,
        "n_paired": n,
        "summary": {
            **counts,
            "union": union,
            "union_rate": round(union / n, 4) if n else 0,
            "regular_rate": round((counts["both"] + counts["regular_only"]) / n, 4) if n else 0,
            "bca_rate": round((counts["both"] + counts["bca_only"]) / n, 4) if n else 0,
        },
        "by_category": by_cat,
        "behaviors": sorted(behaviors, key=lambda b: (b["outcome"], b["category"], b["goal"])),
    }


def build_payload() -> dict:
    datasets = []
    for ds_id, label, files in DATASETS:
        paths = [MULTISEED / f for f in files]
        if not any(p.exists() for p in paths):
            continue
        ds = analyze_files([p for p in paths if p.exists()], label, ds_id)
        if ds:
            datasets.append(ds)
    return {
        "generated_from": [str(MULTISEED / f) for _, _, fs in DATASETS for f in fs],
        "outcome_labels": {
            "both": "Both succeed",
            "regular_only": "Regular only",
            "bca_only": "BCA only",
            "neither": "Neither",
        },
        "category_labels": CAT_LABELS,
        "category_order": list(CAT_ORDER),
        "datasets": datasets,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Seed 44 — TAP Regular vs BCA Complementarity</title>
  <style>
    :root {
      --both: #15803d; --reg: #2563eb; --bca: #9333ea; --neither: #64748b;
      --bg: #f8fafc; --card: #fff; --text: #1e293b; --muted: #64748b;
    }
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
    header { background: #1e293b; color: #f8fafc; padding: 1.2rem 1.5rem; }
    header h1 { margin: 0 0 .35rem; font-size: 1.35rem; }
    header p { margin: 0; color: #cbd5e1; font-size: .92rem; max-width: 52rem; line-height: 1.5; }
    main { max-width: 1100px; margin: 0 auto; padding: 1.2rem 1rem 2.5rem; }
    .toolbar { display: flex; flex-wrap: wrap; gap: .75rem; align-items: center; margin-bottom: 1rem; }
    select, input { font: inherit; padding: .45rem .6rem; border: 1px solid #cbd5e1; border-radius: 6px; background: #fff; }
    input[type=search] { min-width: 220px; flex: 1; }
    .chips { display: flex; flex-wrap: wrap; gap: .4rem; }
    .chip { border: 2px solid transparent; border-radius: 999px; padding: .25rem .7rem; font-size: .82rem; cursor: pointer; background: #fff; }
    .chip.active { font-weight: 600; }
    .chip[data-o=both] { border-color: var(--both); color: var(--both); }
    .chip[data-o=regular_only] { border-color: var(--reg); color: var(--reg); }
    .chip[data-o=bca_only] { border-color: var(--bca); color: var(--bca); }
    .chip[data-o=neither] { border-color: var(--neither); color: var(--neither); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
    .card { background: var(--card); border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .card h2 { margin: 0 0 .75rem; font-size: 1rem; color: #334155; }
    .matrix { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
    .cell { border-radius: 8px; padding: .75rem; color: #fff; cursor: pointer; transition: transform .12s; }
    .cell:hover { transform: translateY(-1px); }
    .cell .n { font-size: 1.8rem; font-weight: 700; line-height: 1; }
    .cell .pct { font-size: .85rem; opacity: .9; }
    .cell .lbl { font-size: .78rem; margin-top: .25rem; opacity: .95; }
    .cell.both { background: var(--both); }
    .cell.reg { background: var(--reg); }
    .cell.bca { background: var(--bca); }
    .cell.neither { background: var(--neither); }
    .union-bar { height: 28px; border-radius: 6px; overflow: hidden; display: flex; margin: .5rem 0; }
    .union-bar span { display: block; height: 100%; }
    .legend { display: flex; flex-wrap: wrap; gap: .8rem; font-size: .78rem; color: var(--muted); margin-top: .4rem; }
    .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: .25rem; }
    .cat-row { margin-bottom: .65rem; }
    .cat-label { font-size: .78rem; color: var(--muted); margin-bottom: .2rem; }
    .cat-bar { height: 22px; border-radius: 4px; overflow: hidden; display: flex; background: #e2e8f0; }
    .cat-bar span { height: 100%; min-width: 2px; }
    table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    th, td { text-align: left; padding: .45rem .5rem; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
    th { background: #f1f5f9; position: sticky; top: 0; }
    .tag { display: inline-block; padding: .1rem .45rem; border-radius: 4px; font-size: .72rem; font-weight: 600; color: #fff; }
    .tag.both { background: var(--both); }
    .tag.regular_only { background: var(--reg); }
    .tag.bca_only { background: var(--bca); }
    .tag.neither { background: var(--neither); }
    .prompt { font-family: ui-monospace, monospace; font-size: .72rem; color: #475569; max-width: 28rem; white-space: pre-wrap; }
    .stats { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .88rem; color: var(--muted); margin-bottom: .5rem; }
    .scroll { max-height: 420px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
  </style>
</head>
<body>
<header>
  <h1>Seed 44 — Regular vs BCA Complementarity</h1>
  <p>Deploy success (judge = 10/10 on <code>best_objective</code>) for seed-44 TAP runs.
     Includes the 30-behavior L=8 / L=12 replicates and the 100-behavior run with
     <strong>regular + rich BCA feedback</strong> vs BCA (judge-pick deploy).
     Click outcome cells or chips to filter behaviors.</p>
</header>
<main>
  <div class="toolbar">
    <label>Dataset <select id="dataset"></select></label>
    <input type="search" id="search" placeholder="Filter behaviors…">
  </div>
  <div class="chips" id="chips"></div>
  <div class="stats" id="stats"></div>
  <div class="grid">
    <div class="card">
      <h2>Outcome matrix</h2>
      <div class="matrix" id="matrix"></div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card">
      <h2>Coverage</h2>
      <p style="font-size:.85rem;color:var(--muted);margin:0 0 .5rem">Share of paired behaviors with a deployable jailbreak</p>
      <div id="coverage"></div>
    </div>
    <div class="card" style="grid-column: 1 / -1">
      <h2>By category</h2>
      <div id="categories"></div>
    </div>
  </div>
  <div class="card">
    <h2>Behaviors <span id="beh-count" style="color:var(--muted);font-weight:400"></span></h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Outcome</th><th>Category</th><th>Goal</th><th>Regular prompt</th><th>BCA prompt</th></tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</main>
<script>
const DATA = __DATA_JSON__;
const COLORS = { both: '#15803d', regular_only: '#2563eb', bca_only: '#9333ea', neither: '#64748b' };
let ds = null, filterOutcome = 'all', search = '';

function pct(n, d) { return d ? (100 * n / d).toFixed(1) + '%' : '—'; }

function init() {
  const sel = document.getElementById('dataset');
  DATA.datasets.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = d.label + ' (' + d.n_paired + ' paired)';
    sel.appendChild(o);
  });
  sel.addEventListener('change', () => { ds = DATA.datasets[+sel.value]; filterOutcome = 'all'; render(); });
  document.getElementById('search').addEventListener('input', e => { search = e.target.value.toLowerCase(); renderTable(); });
  const chips = document.getElementById('chips');
  ['all', ...Object.keys(DATA.outcome_labels)].forEach(k => {
    const b = document.createElement('button');
    b.className = 'chip' + (k !== 'all' ? '' : ' active');
    b.dataset.o = k;
    b.textContent = k === 'all' ? 'All outcomes' : DATA.outcome_labels[k];
    b.addEventListener('click', () => {
      filterOutcome = k;
      chips.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c.dataset.o === k));
      render();
    });
    chips.appendChild(b);
  });
  ds = DATA.datasets[0];
  render();
}

function setFilter(o) {
  filterOutcome = o;
  document.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c.dataset.o === o));
  render();
}

function render() {
  const s = ds.summary, n = ds.n_paired;
  document.getElementById('stats').innerHTML =
    `<span><strong>${n}</strong> paired behaviors</span>` +
    `<span>Union: <strong>${pct(s.union, n)}</strong> (${s.union})</span>` +
    `<span>Regular alone: ${pct(s.both + s.regular_only, n)}</span>` +
    `<span>BCA alone: ${pct(s.both + s.bca_only, n)}</span>`;

  const matrix = document.getElementById('matrix');
  matrix.innerHTML = '';
  [
    ['both', 'both', s.both, 'Both deploy'],
    ['regular_only', 'reg', s.regular_only, 'Regular only'],
    ['bca_only', 'bca', s.bca_only, 'BCA only'],
    ['neither', 'neither', s.neither, 'Neither'],
  ].forEach(([key, cls, count, lbl]) => {
    const d = document.createElement('div');
    d.className = 'cell ' + cls;
    d.innerHTML = `<div class="n">${count}</div><div class="pct">${pct(count, n)}</div><div class="lbl">${lbl}</div>`;
    d.onclick = () => setFilter(key);
    matrix.appendChild(d);
  });

  const cov = document.getElementById('coverage');
  const regOnly = s.both + s.regular_only, bcaOnly = s.both + s.bca_only;
  cov.innerHTML = `
    <div style="margin-bottom:.6rem"><strong>Regular</strong> ${pct(regOnly, n)} <span style="color:var(--muted)">(${regOnly})</span></div>
    <div class="union-bar" title="Regular coverage">
      <span style="width:${100*s.regular_rate}%;background:var(--reg)"></span>
    </div>
    <div style="margin:.8rem 0 .6rem"><strong>BCA</strong> ${pct(bcaOnly, n)} <span style="color:var(--muted)">(${bcaOnly})</span></div>
    <div class="union-bar" title="BCA coverage">
      <span style="width:${100*s.bca_rate}%;background:var(--bca)"></span>
    </div>
    <div style="margin:.8rem 0 .6rem"><strong>Union (either)</strong> ${pct(s.union, n)} <span style="color:var(--muted)">(+${pct(s.union - Math.max(regOnly,bcaOnly), n)} vs best single)</span></div>
    <div class="union-bar" title="Union">
      <span style="width:${100*s.both/n}%;background:var(--both)"></span>
      <span style="width:${100*s.regular_only/n}%;background:var(--reg)"></span>
      <span style="width:${100*s.bca_only/n}%;background:var(--bca)"></span>
    </div>`;

  const cats = document.getElementById('categories');
  cats.innerHTML = '';
  DATA.category_order.forEach(cat => {
    const row = document.createElement('div');
    row.className = 'cat-row';
    let total = 0;
    OUTCOMES.forEach(o => { total += (ds.by_category[o][cat] || 0); });
    if (!total) return;
    let bar = '';
    OUTCOMES.forEach(o => {
      const c = ds.by_category[o][cat] || 0;
      if (c) bar += `<span style="width:${100*c/total}%;background:${COLORS[o]}" title="${DATA.outcome_labels[o]}: ${c}"></span>`;
    });
    row.innerHTML = `<div class="cat-label">${DATA.category_labels[cat] || cat} <strong>${total}</strong></div><div class="cat-bar">${bar}</div>`;
    cats.appendChild(row);
  });
  document.getElementById('legend').innerHTML = OUTCOMES.map(o =>
    `<span><i style="background:${COLORS[o]}"></i>${DATA.outcome_labels[o]}</span>`).join('');

  renderTable();
}

const OUTCOMES = ['both', 'regular_only', 'bca_only', 'neither'];

function renderTable() {
  let rows = ds.behaviors;
  if (filterOutcome !== 'all') rows = rows.filter(r => r.outcome === filterOutcome);
  if (search) rows = rows.filter(r =>
    (r.goal + r.category + r.outcome + String(r.seed)).toLowerCase().includes(search));
  document.getElementById('beh-count').textContent = `(${rows.length} shown)`;
  const tb = document.getElementById('tbody');
  tb.innerHTML = rows.map(r => `<tr>
    <td><span class="tag ${r.outcome}">${DATA.outcome_labels[r.outcome]}</span></td>
    <td>${DATA.category_labels[r.category] || r.category}</td>
    <td>${esc(r.goal)}${r.same_prompt && r.outcome==='both' ? ' <em style="color:var(--both)">same prompt</em>' : ''}</td>
    <td class="prompt">${r.regular_deploy ? esc(r.regular_prompt) : '—'}</td>
    <td class="prompt">${r.bca_deploy ? esc(r.bca_prompt) : '—'}</td>
  </tr>`).join('');
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
init();
</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    OUT_HTML_SEED44.write_text(html, encoding="utf-8")
    explorer = html.replace(
        "<title>Seed 44 — TAP Regular vs BCA Complementarity</title>",
        "<title>TAP Regular vs BCA — Complementarity Explorer</title>",
    ).replace(
        "<h1>Seed 44 — Regular vs BCA Complementarity</h1>",
        "<h1>TAP Regular vs BCA Complementarity</h1>",
    )
    OUT_HTML.write_text(explorer, encoding="utf-8")
    print(f"Wrote {OUT_DATA}")
    print(f"Wrote {OUT_HTML_SEED44}")
    print(f"Wrote {OUT_HTML}")
    for ds in payload["datasets"]:
        s = ds["summary"]
        print(f"  {ds['label']}: n={ds['n_paired']} union={s['union']} ({100*s['union_rate']:.1f}%)")


if __name__ == "__main__":
    main()
