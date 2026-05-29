"""
vLLM-based inference backend.

Key vLLM optimisations exploited here:
  1. PagedAttention         — non-contiguous KV cache paging removes the
                              quadratic memory cost of naïve KV allocation.
  2. Automatic prefix caching (enable_prefix_caching=True)
                            — KV blocks for shared string prefixes are reused
                              across the siblings in the BFS tree.  This is
                              the direct GPU equivalent of the DFS recursion's
                              implicit prefix reuse and is the single largest
                              constant-factor speedup over the HF backend.
  3. Continuous batching    — requests at different sequence lengths are
                              served in the same forward pass; the BFS level
                              naturally produces requests of similar length,
                              giving near-perfect batch utilisation.
  4. Chunked prefill        — long shared prefixes are processed in chunks,
                              keeping VRAM usage flat even for deep trees.

Interface
---------
    backend = VLLMBackend("meta-llama/Llama-2-7b-hf")
    results = backend.get_top_k_batch(["string1", "string2"], alpha=0.9, k=15)
    # results[i] = (token_strings, probabilities)  for string i
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

try:
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.outputs import RequestOutput  # type: ignore
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


class VLLMBackend:
    """
    Wraps vLLM's LLM engine to provide batched α-k top-token retrieval.

    Parameters
    ----------
    model_name        : HuggingFace model id or local path
    max_logprobs      : maximum k vLLM will return per token position;
                        must be >= the k you pass to get_top_k_batch
    gpu_memory_utilisation : fraction of GPU VRAM to reserve for KV cache
    tensor_parallel_size   : number of GPUs for tensor parallelism
    dtype             : weight dtype ('auto', 'float16', 'bfloat16')
    enable_prefix_caching  : share KV cache across requests with common prefixes
    sampling_top_p    : nucleus sampling threshold applied to the LLM
                        distribution before α-k bounding (1.0 = off)
    sampling_top_k    : hard-cap on top-k sampling before α-k bounding (-1 = off)
    extra_llm_kwargs  : forwarded verbatim to vllm.LLM(...)
    """

    def __init__(
        self,
        model_name: str,
        max_logprobs: int = 64,
        gpu_memory_utilisation: float = 0.90,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        enable_prefix_caching: bool = True,
        max_batch_size: int = 32_768,   # chunk large BFS levels into this many strings
        sampling_top_p: float = 1.0,
        sampling_top_k: int = -1,
        **extra_llm_kwargs,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vllm is not installed.  Run: pip install vllm\n"
                "Or use HFBackend as a fallback."
            )

        self.max_logprobs = max_logprobs
        self.model_name = model_name
        self.sampling_top_p = sampling_top_p
        self.sampling_top_k = sampling_top_k

        self.max_batch_size = max_batch_size
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilisation,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            # torch.compile in vLLM ≥0.18 has a FakeTensorMode incompatibility
            # with torch 2.10; eager mode is slightly slower but always correct.
            enforce_eager=True,
            **extra_llm_kwargs,
        )

    # ── Core method ───────────────────────────────────────────────────────────

    def get_top_k_batch(
        self,
        strings: List[str],
        alpha: float,
        k: int,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k_sampling: Optional[int] = None,
    ) -> List[Tuple[List[str], List[float]]]:
        """
        For each string in the batch return (token_strings, probabilities)
        for the α-k-bounded top tokens at the next generation step.

        The requests are submitted to vLLM in one call; with prefix caching
        enabled, siblings that share a common prefix only pay the KV cost
        for their diverging suffix.

        alpha          : cumulative probability threshold  ∈ (0, 1]
        k              : hard cap on number of tokens returned
        temperature    : softmax temperature applied to logits
        top_p          : nucleus sampling threshold (overrides instance default)
        top_k_sampling : hard top-k sampling cap (overrides instance default)
        """
        if k > self.max_logprobs:
            raise ValueError(
                f"k={k} exceeds max_logprobs={self.max_logprobs}.  "
                f"Re-initialise VLLMBackend with max_logprobs >= {k}."
            )

        effective_top_p = top_p if top_p is not None else self.sampling_top_p
        effective_top_k = top_k_sampling if top_k_sampling is not None else self.sampling_top_k

        sampling_params = SamplingParams(
            n=1,
            max_tokens=1,        # one-step lookahead only
            logprobs=k,          # return top-k log-probabilities
            temperature=temperature,
            top_p=effective_top_p,
            top_k=effective_top_k,
        )

        # Chunk large BFS levels so GPU VRAM is never oversubscribed.
        # With prefix caching enabled, the first chunk pays the full KV cost
        # for the shared prefix; subsequent chunks hit the cache for free.
        results: List[Tuple[List[str], List[float]]] = []
        for i in range(0, len(strings), self.max_batch_size):
            chunk = strings[i : i + self.max_batch_size]
            outputs: List[RequestOutput] = self.llm.generate(chunk, sampling_params)
            results.extend(self._parse_output(out, alpha, k) for out in outputs)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_output(
        output: RequestOutput,
        alpha: float,
        k: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Convert a vLLM RequestOutput into an α-k-bounded (tokens, probs) pair.

        vLLM returns logprobs as a list (one dict per generated token).
        We request max_tokens=1, so logprobs[0] is a dict:
            {token_id: Logprob(logprob=float, decoded_token=str, ...)}
        sorted descending by logprob.
        """
        logprobs_at_step: Dict = output.outputs[0].logprobs[0]

        # Sort descending by log-probability
        sorted_items = sorted(
            logprobs_at_step.items(),
            key=lambda kv: kv[1].logprob,
            reverse=True,
        )

        tokens: List[str] = []
        probs: List[float] = []
        cumulative = 0.0

        for _token_id, lp_obj in sorted_items[:k]:
            prob = math.exp(lp_obj.logprob)
            token_str = lp_obj.decoded_token or ""
            tokens.append(token_str)
            probs.append(prob)
            cumulative += prob
            if cumulative >= alpha:
                break

        return tokens, probs

    @property
    def tokenizer(self):
        """Chat-template tokenizer (for BCA initial-string rendering)."""
        return self.llm.get_tokenizer()

    @property
    def device(self) -> str:
        return "cuda:0"

    def generate_chat_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        """Multi-token chat completion for TAP/PAIR attack and target queries."""
        from pair_bca_llama import chat_prompt  # noqa: WPS433 — shared template helper

        prompts = [chat_prompt(self.tokenizer, conv) for conv in conversations]
        do_sample = temperature > 0
        sp = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
        )
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs]
