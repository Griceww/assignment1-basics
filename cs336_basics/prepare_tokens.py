"""Prepare uint16 token binaries from raw text files.

This script trains a byte-level BPE tokenizer, encodes train/valid corpora, and
writes token IDs as uint16 .bin files consumable by train.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BPE and write uint16 token bins.")

    p.add_argument("--train_text", type=str, required=True, help="Path to raw training text file.")
    p.add_argument("--valid_text", type=str, required=True, help="Path to raw validation text file.")
    p.add_argument("--train_bin_out", type=str, required=True, help="Output path for training tokens (.bin).")
    p.add_argument("--valid_bin_out", type=str, required=True, help="Output path for validation tokens (.bin).")

    p.add_argument("--vocab_size", type=int, required=True, help="Final vocabulary size (including special tokens).")
    p.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help='Special tokens list, e.g. --special_tokens "<|endoftext|>"',
    )
    p.add_argument("--num_processes", type=int, default=50, help="Worker processes for BPE pre-tokenization.")
    p.add_argument(
        "--imap_chunksize",
        type=int,
        default=1,
        help="imap chunksize for BPE pre-tokenization multiprocessing.",
    )

    p.add_argument(
        "--tokenizer_prefix",
        type=str,
        default=None,
        help="If set, save tokenizer assets to <prefix>_vocab.json and <prefix>_merges.json.",
    )
    p.add_argument(
        "--flush_size",
        type=int,
        default=1_000_000,
        help="How many token IDs to buffer before writing to disk.",
    )
    return p.parse_args()


def _ensure_parent(path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _check_uint16(ids: list[int]) -> None:
    if not ids:
        return
    max_id = max(ids)
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Found token id {max_id}, which exceeds uint16 range. "
            "Use a smaller vocab_size or switch training data format."
        )


def _write_chunk(file_obj, ids: list[int]) -> None:
    if not ids:
        return
    _check_uint16(ids)
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(file_obj)


def encode_text_to_bin(
    tokenizer: Tokenizer,
    input_txt: str | os.PathLike,
    output_bin: str | os.PathLike,
    flush_size: int,
) -> int:
    _ensure_parent(output_bin)
    total = 0
    buffer: list[int] = []

    with open(input_txt, "r", encoding="utf-8") as f_in, open(output_bin, "wb") as f_out:
        for token_id in tokenizer.encode_iterable(f_in):
            buffer.append(token_id)
            if len(buffer) >= flush_size:
                _write_chunk(f_out, buffer)
                total += len(buffer)
                buffer.clear()
                if total % (10 * flush_size) == 0:
                    print(f"[encode] {input_txt}: wrote {total} tokens", flush=True)
        if buffer:
            _write_chunk(f_out, buffer)
            total += len(buffer)

    return total


def save_tokenizer_assets(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    prefix: str,
) -> tuple[str, str]:
    vocab_path = f"{prefix}_vocab.json"
    merges_path = f"{prefix}_merges.json"
    _ensure_parent(vocab_path)
    _ensure_parent(merges_path)

    serializable_vocab = {str(k): list(v) for k, v in vocab.items()}
    serializable_merges = [[list(a), list(b)] for a, b in merges]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f)
    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f)

    return vocab_path, merges_path


def main() -> None:
    args = parse_args()

    if args.vocab_size <= 0:
        raise ValueError("--vocab_size must be positive")
    if args.flush_size <= 0:
        raise ValueError("--flush_size must be positive")
    if args.num_processes <= 0:
        raise ValueError("--num_processes must be positive")
    if args.imap_chunksize <= 0:
        raise ValueError("--imap_chunksize must be positive")

    t0 = time.time()
    print("[stage] train_bpe start", flush=True)
    vocab, merges = train_bpe(
        input_path=args.train_text,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
        imap_chunksize=args.imap_chunksize,
    )
    print(f"[stage] train_bpe done in {time.time() - t0:.1f}s", flush=True)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=args.special_tokens)

    print("[stage] encode train start", flush=True)
    t1 = time.time()
    train_tokens = encode_text_to_bin(
        tokenizer=tokenizer,
        input_txt=args.train_text,
        output_bin=args.train_bin_out,
        flush_size=args.flush_size,
    )
    print(f"[stage] encode train done in {time.time() - t1:.1f}s", flush=True)

    # Encourage release of temporary objects before validation encoding.
    gc.collect()

    print("[stage] encode valid start", flush=True)
    t2 = time.time()
    valid_tokens = encode_text_to_bin(
        tokenizer=tokenizer,
        input_txt=args.valid_text,
        output_bin=args.valid_bin_out,
        flush_size=args.flush_size,
    )
    print(f"[stage] encode valid done in {time.time() - t2:.1f}s", flush=True)

    print(f"train tokens: {train_tokens} -> {args.train_bin_out}")
    print(f"valid tokens: {valid_tokens} -> {args.valid_bin_out}")
    print(f"final vocab size: {len(vocab)}")

    if args.tokenizer_prefix:
        vocab_path, merges_path = save_tokenizer_assets(vocab=vocab, merges=merges, prefix=args.tokenizer_prefix)
        print(f"saved tokenizer vocab: {vocab_path}")
        print(f"saved tokenizer merges: {merges_path}")


if __name__ == "__main__":
    main()
