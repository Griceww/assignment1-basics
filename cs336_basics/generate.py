"""Autoregressive decoding for TransformerLM (temperature, top-p, EOS)."""

from __future__ import annotations

import torch

from cs336_basics.tokenizer import Tokenizer


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive for sampling")
    return logits / temperature


def _sample_next_token(
    probs: torch.Tensor,
    top_p: float,
) -> int:
    """Sample one token from 1D probability vector (V,) summing to 1."""
    if top_p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1).item())

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    mask[0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    total = sorted_probs.sum()
    if total <= 0:
        sorted_probs = torch.zeros_like(sorted_probs)
        sorted_probs[0] = 1.0
    else:
        sorted_probs = sorted_probs / total
    j = int(torch.multinomial(sorted_probs, num_samples=1).item())
    return int(sorted_idx[j].item())


def generate_completion(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device,
    eos_token: str = "<|endoftext|>",
) -> str:
    """
    Generate text continuation from ``prompt`` until ``eos_token`` or ``max_new_tokens``.

    At each step the last position logits are temperature-scaled, softmaxed, optionally
    nucleus-filtered (top-p), then sampled with replacement.

    Args:
        model: ``TransformerLM`` (or compatible) with ``context_length`` and forward
            returning logits ``(batch, seq, vocab)``.
        tokenizer: BPE tokenizer with ``encode`` / ``decode`` and optional ``<|endoftext|>``.
        prompt: User text including any special tokens to start from.
        max_new_tokens: Upper bound on *new* tokens (not including prompt).
        temperature: If ``<= 0``, take argmax (greedy). Otherwise divide logits by
            ``temperature`` before softmax.
        top_p: Nucleus mass in [0, 1]; 1.0 disables top-p.
        device: Where to run the model.
        eos_token: Stop when this token ID is sampled (default GPT-2 style).

    Returns:
        Decoded string for prompt + generated tokens (EOS token id is included before stop;
        decode may show the special sequence per tokenizer).
    """
    model.eval()
    device = torch.device(device)
    ctx = model.context_length

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = []

    eos_bytes = eos_token.encode("utf-8")
    eos_id = tokenizer.vocab_to_id.get(eos_bytes)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = token_ids[-ctx:] if len(token_ids) > ctx else token_ids
            x = torch.as_tensor([window], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1, :]

            if temperature <= 0:
                next_id = int(next_logits.argmax(dim=-1).item())
            else:
                next_logits = _apply_temperature(next_logits, temperature)
                probs = torch.softmax(next_logits, dim=-1)
                if top_p < 1.0:
                    if top_p <= 0:
                        raise ValueError("top_p must be positive when < 1.0")
                    next_id = _sample_next_token(probs, top_p)
                else:
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

            token_ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

    return tokenizer.decode(token_ids)
