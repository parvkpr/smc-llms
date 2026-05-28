"""
HuggingFace Transformers fallback backend.

Uses manual padding-aware batching: for each string in the batch we find
the last non-padding token position and read the logits there, so all
strings in the BFS level are processed in a single forward pass.

This is considerably slower than VLLMBackend (no prefix caching, no paged
attention) but requires only transformers + torch and works on any hardware.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch


class HFBackend:
    """
    Parameters
    ----------
    model_name     : HuggingFace model id or local path
    device         : 'cuda', 'cpu', or 'auto' (uses device_map='auto')
    dtype          : torch dtype for model weights (default: float16 on CUDA)
    batch_size     : max strings per forward pass (tune to fit VRAM)
    sampling_top_p : nucleus sampling threshold applied before α-k bounding
                     (1.0 = disabled; carried as default for generate calls)
    sampling_top_k : hard-cap for top-k sampling before α-k bounding
                     (-1 = disabled)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 8,
        sampling_top_p: float = 1.0,
        sampling_top_k: int = -1,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.batch_size = batch_size
        self.model_name = model_name
        self.sampling_top_p = sampling_top_p
        self.sampling_top_k = sampling_top_k

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device if device == "auto" else None,
        )
        if device != "auto":
            self.model = self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.model.eval()

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
        effective_top_p = top_p if top_p is not None else self.sampling_top_p
        effective_top_k = top_k_sampling if top_k_sampling is not None else self.sampling_top_k
        results: List[Tuple[List[str], List[float]]] = []
        for i in range(0, len(strings), self.batch_size):
            chunk = strings[i : i + self.batch_size]
            results.extend(
                self._process_chunk(chunk, alpha, k, temperature, effective_top_p, effective_top_k)
            )
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _process_chunk(
        self,
        strings: List[str],
        alpha: float,
        k: int,
        temperature: float,
        top_p: float = 1.0,
        top_k_sampling: int = -1,
    ) -> List[Tuple[List[str], List[float]]]:
        enc = self.tokenizer(
            strings,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)

        # For each sequence, identify the last *real* token position
        # (left-padding means the last column is always the last real token)
        last_real_pos = enc["attention_mask"].sum(dim=1) - 1  # [B]

        results: List[Tuple[List[str], List[float]]] = []
        vocab_size = out.logits.shape[-1]
        effective_k = min(k, vocab_size)

        for i in range(len(strings)):
            pos = last_real_pos[i].item()
            logits = out.logits[i, pos].clone()  # [vocab_size]

            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # Apply top-k truncation to logits before softmax
            if top_k_sampling > 0:
                top_k_cap = min(top_k_sampling, vocab_size)
                kth_val = torch.topk(logits, top_k_cap).values[-1]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            probs_full = torch.softmax(logits, dim=-1)

            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs_full, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=0)
                # Remove tokens with cumulative prob above threshold (keep first token)
                remove_mask = cum_probs - sorted_probs > top_p
                sorted_probs[remove_mask] = 0.0
                probs_full = torch.zeros_like(probs_full)
                probs_full.scatter_(0, sorted_idx, sorted_probs)
                s = probs_full.sum()
                if s > 0:
                    probs_full = probs_full / s

            log_probs = torch.log(probs_full.clamp(min=1e-45))
            top_log_probs, top_ids = torch.topk(log_probs, effective_k)

            tokens: List[str] = []
            probs: List[float] = []
            cumulative = 0.0

            for tid, lp in zip(top_ids.tolist(), top_log_probs.tolist()):
                if lp == float("-inf"):
                    break
                prob = math.exp(lp)
                token_str = self.tokenizer.decode([tid])
                tokens.append(token_str)
                probs.append(prob)
                cumulative += prob
                if cumulative >= alpha:
                    break

            results.append((tokens, probs))

        return results
