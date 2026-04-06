"""CLI for autoregressive text generation from a trained TransformerLM checkpoint."""

from __future__ import annotations

import argparse
import sys

import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generate import generate_completion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a TransformerLM checkpoint.")

    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--vocab", type=str, required=True, help="Path to vocab JSON (from prepare_tokens.py).")
    p.add_argument("--merges", type=str, required=True, help="Path to merges JSON (from prepare_tokens.py).")

    p.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to start generation.")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy).")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold in (0,1]; 1.0 disables.")
    p.add_argument("--device", type=str, default=None, help="Device (default: cuda if available, else cpu).")
    p.add_argument("--output", type=str, default=None, help="Optional file path to save the generated text.")

    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    p.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help='Special tokens list, e.g. --special_tokens "<|endoftext|>"',
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=args.special_tokens,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
    )

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    text = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[saved to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    main()
