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
    model_name  : HuggingFace model id or local path
    device      : 'cuda', 'cpu', or 'auto' (uses device_map='auto')
    dtype       : torch dtype for model weights (default: float16 on CUDA)
    batch_size  : max strings per forward pass (tune to fit VRAM)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 8,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.batch_size = batch_size
        self.model_name = model_name

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
    ) -> List[Tuple[List[str], List[float]]]:
        results: List[Tuple[List[str], List[float]]] = []
        for i in range(0, len(strings), self.batch_size):
            chunk = strings[i : i + self.batch_size]
            results.extend(self._process_chunk(chunk, alpha, k, temperature))
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _process_chunk(
        self,
        strings: List[str],
        alpha: float,
        k: int,
        temperature: float,
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
            logits = out.logits[i, pos]  # [vocab_size]

            if temperature != 1.0:
                logits = logits / temperature

            log_probs = torch.log_softmax(logits, dim=-1)
            top_log_probs, top_ids = torch.topk(log_probs, effective_k)

            tokens: List[str] = []
            probs: List[float] = []
            cumulative = 0.0

            for tid, lp in zip(top_ids.tolist(), top_log_probs.tolist()):
                prob = math.exp(lp)
                token_str = self.tokenizer.decode([tid])
                tokens.append(token_str)
                probs.append(prob)
                cumulative += prob
                if cumulative >= alpha:
                    break

            results.append((tokens, probs))

        return results
