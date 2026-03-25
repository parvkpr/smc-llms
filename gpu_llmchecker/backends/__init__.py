"""
Inference backends for batched top-k token probability retrieval.

Both backends expose the same interface:
    get_top_k_batch(strings, alpha, k) -> List[Tuple[List[str], List[float]]]

VLLMBackend  : preferred; uses PagedAttention + automatic prefix caching
               (enables KV-cache sharing across tree siblings sharing a prefix)
HFBackend    : fallback for environments where vLLM is not installed;
               uses HuggingFace transformers with manual padding-aware batching
"""

from .vllm_backend import VLLMBackend
from .hf_backend import HFBackend

__all__ = ["VLLMBackend", "HFBackend"]
