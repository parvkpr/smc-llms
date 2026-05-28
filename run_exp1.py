"""Experiments 1-3: Exact, Direct SMC long, Convergence"""
import sys, os, time, math, json
sys.path.insert(0, '/home/parvk/smc-llms')
import warnings; warnings.filterwarnings("ignore")

from gpu_llmchecker.backends import HFBackend
from gpu_llmchecker.quantification import (
    GenderBias, SentimentScore, ReadingQuality, StepCounter, MultiQuantifier
)
from gpu_llmchecker.pctl import eventually, always
from gpu_llmchecker import build_dtmc_bfs, exact_backward_induction, direct_smc
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}", flush=True)
print("Loading GPT-2...", flush=True)
backend = HFBackend('gpt2', device='auto', batch_size=64)
print("OK\n", flush=True)

res = {}

# ── 1. Exact verification ─────────────────────────────────────────────────────
print("=== EXACT ===", flush=True)
exps = [
    ("The player won because",  eventually("gender",">",0),  MultiQuantifier([GenderBias(),StepCounter()]),      0.9,5,4,"gender"),
    ("The exam was",            eventually("polarity",">=",10),MultiQuantifier([SentimentScore(),StepCounter()]), 0.9,5,4,"polarity"),
    ("Our story",               always("readability",">",1000),MultiQuantifier([ReadingQuality(),StepCounter()]), 0.8,3,5,"readability"),
]
rows=[]
for start,q,quant,alpha,k,L,label in exps:
    t0=time.perf_counter()
    lvls,stats=build_dtmc_bfs(start,L,alpha,k,lambda s,d,qq=quant:qq(s,d),backend,verbose=False)
    tenc=time.perf_counter()-t0
    t1=time.perf_counter()
    prob,_=exact_backward_induction(lvls,q,device=device)
    tver=(time.perf_counter()-t1)*1000
    row=dict(label=label,alpha=alpha,k=k,L=L,states=int(stats["total_nodes"]),
             encode_s=round(tenc,2),verify_ms=round(tver,1),prob=round(float(prob),4))
    rows.append(row)
    print(f"  {label}  |S|={row['states']}  ET={row['encode_s']:.2f}s  VT={row['verify_ms']:.1f}ms  p={row['prob']:.4f}", flush=True)
res["exact"]=rows

# ── 2. Direct SMC long lookahead ──────────────────────────────────────────────
print("\n=== SMC LONG ===", flush=True)
long_exps=[
    ("The player won because",  eventually("gender",">",2),   MultiQuantifier([GenderBias(),StepCounter()]),     50),
    ("The exam was a wonderful",always("polarity",">=",0),    MultiQuantifier([SentimentScore(),StepCounter()]), 30),
    ("Our story begins",        always("readability",">",100),MultiQuantifier([ReadingQuality(),StepCounter()]), 40),
]
long_rows=[]
for start,q,quant,L in long_exps:
    t0=time.perf_counter()
    p,lo,hi,stats=direct_smc(start,L,q,lambda s,d,qq=quant:qq(s,d),backend,num_samples=1000,chunk_size=64,verbose=False)
    elapsed=time.perf_counter()-t0
    row=dict(L=L,p=round(p,3),lo=round(lo,3),hi=round(hi,3),time_s=round(elapsed,1))
    long_rows.append(row)
    print(f"  L={L}  p={p:.3f}  [{lo:.3f},{hi:.3f}]  {elapsed:.1f}s", flush=True)
res["dsmc_long"]=long_rows

# ── 3. Convergence ────────────────────────────────────────────────────────────
print("\n=== CONVERGENCE ===", flush=True)
cq=MultiQuantifier([GenderBias(),StepCounter()])
conv_rows=[]
for m in [100,300,500,1000,2000,3000]:
    t0=time.perf_counter()
    p,lo,hi,stats=direct_smc("The player won because",50,eventually("gender",">",0),
        lambda s,d:cq(s,d),backend,num_samples=m,chunk_size=64,verbose=False)
    elapsed=time.perf_counter()-t0
    row=dict(M=m,p=round(p,3),lo=round(lo,3),hi=round(hi,3),
             ci_width=round(hi-lo,3),eps=round(stats["epsilon"],3),time_s=round(elapsed,1))
    conv_rows.append(row)
    print(f"  M={m}  p={p:.3f}  CI=[{lo:.3f},{hi:.3f}]  w={hi-lo:.3f}  eps={stats['epsilon']:.3f}  {elapsed:.1f}s", flush=True)
res["convergence"]=conv_rows

print("\nJSON:", json.dumps(res, indent=2), flush=True)
