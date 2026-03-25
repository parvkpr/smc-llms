# gpu_llmchecker

GPU-parallel α-k-bounded PCTL model checking for LLM text generation.

Reimplements and accelerates [LLMCHECKER (Gross et al., 2025)](https://arxiv.org/abs/2509.18836) by replacing the sequential DFS + Storm pipeline with:

| Original | This codebase |
|---|---|
| Recursive DFS (Algorithm 1) | **BFS with one batched LLM call per depth level** |
| Serial LLM forward passes | **vLLM PagedAttention + automatic prefix caching** |
| PRISM serialisation → Storm | **GPU sparse backward induction (PyTorch)** |
| O(k^L) sequential calls | **O(L) batched calls, O(k^L) parallel GPU work** |

## Install

```bash
pip install -r requirements.txt
# vLLM requires CUDA; for CPU-only use, the HFBackend works without it
```

## Quick start

```python
from gpu_llmchecker import LLMCheckerGPU, MultiQuantifier, GenderBias, StepCounter
from gpu_llmchecker.pctl import eventually
from gpu_llmchecker.backends import VLLMBackend

backend  = VLLMBackend("google/gemma-2b-it", enable_prefix_caching=True)
checker  = LLMCheckerGPU(backend, MultiQuantifier([GenderBias(), StepCounter()]))

result = checker.check(
    start_string = "The player won because",
    query        = eventually("gender", ">", 0),
    alpha        = 0.8,
    k            = 15,
    L            = 5,
)
print(result)
```

## Architecture

```
gpu_llmchecker/
├── dtmc.py           DTMCNode dataclass + sparse transition matrix builder
├── pctl.py           PCTLQuery, AtomicProp, ConjunctiveProp, convenience constructors
├── quantification.py GenderBias, SentimentScore, ReadingQuality, CopyrightSim, MultiQuantifier
├── bfs_builder.py    BFS DTMC construction — one batched LLM call per depth level
├── verification.py   GPU backward induction (exact) + statistical MC (approximate)
├── checker.py        LLMCheckerGPU main class
└── backends/
    ├── vllm_backend.py   vLLM (PagedAttention, prefix caching, continuous batching)
    └── hf_backend.py     HuggingFace Transformers fallback
```

## Key ideas

### BFS replaces DFS

The original Algorithm 1 expands the DTMC tree in DFS order, issuing one
LLM forward pass per node.  With k=15 and L=5 that is up to 15^5 ≈ 750k
*serial* LLM calls.  BFS collects all nodes at depth d and submits them as
a single batch, reducing serial LLM calls from O(k^L) to O(L).

### vLLM prefix caching

Siblings in the BFS tree share a common string prefix (the parent's string).
vLLM's automatic KV-cache prefix sharing means the transformer's attention
for the shared prefix is computed only once and reused across all siblings —
exactly the implicit reuse in the DFS recursion, but now across a whole GPU
batch.

### GPU sparse backward induction

For tree-structured DTMCs, PCTL reachability reduces to level-by-level
backward induction (no iterative linear solver needed).  Each level step is
one sparse matrix–vector product:

```
values[d] = T[d] @ values[d+1]       # torch.sparse_coo @ dense vector
```

followed by a vectorised mask for absorbing states.  This replaces Storm's
CPU-based value iteration with O(L) GPU kernel launches.

### Statistical MC fallback

For extreme scale (large k, L), the `use_smc=True` option samples M
independent paths through the already-built DTMC tree in parallel on GPU.
Confidence bounds use the Chernoff–Hoeffding inequality.

## Examples

```bash
# Gender bias (replicates Table II of the paper)
python examples/gender_bias.py --model google/gemma-2b-it --alpha 0.8 --k 15 --L 5

# Statistical MC (faster, approximate)
python examples/gender_bias.py --model google/gemma-2b-it --smc --smc-samples 5000

# Benchmark across k and L values
python examples/benchmark.py --model google/gemma-2b-it --k-values 3 5 9 --L-values 3 5 7
```

## PCTL queries

```python
from gpu_llmchecker.pctl import eventually, always, eventually_conj

eventually("gender", ">", 0)                       # P(F gender > 0)
always("readability", ">", 5997)                   # P(G readability > 5997)
eventually_conj([("step","==",5),("similarity",">",90)])  # P(F step==5 ∧ similarity>90)
```

## Quantifiers

```python
from gpu_llmchecker.quantification import MultiQuantifier, GenderBias, SentimentScore, ReadingQuality, CopyrightSim, StepCounter

q = MultiQuantifier([
    GenderBias(),
    SentimentScore(),
    ReadingQuality(),
    CopyrightSim(reference="Lolita, light of my life, fire of my loins."),
    StepCounter(),
])
q("The player won because he scored", depth=5)
# → {'gender': 1, 'polarity': 20, 'readability': 7500, 'similarity': 12, 'step': 5}
```
# smc-llms
