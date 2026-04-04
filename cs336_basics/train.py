"""Training script for TransformerLM. Ties together all cs336_basics components."""

import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.lr_scheduler import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import clip_gradients
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.get_batch import get_batch
from cs336_basics.cross_entropy import cross_entropy


def parse_args():
    p = argparse.ArgumentParser(description="Train a TransformerLM")

    # --- model ---
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # --- optimizer ---
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # --- lr schedule ---
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--cosine_cycle_iters", type=int, default=None,
                    help="Defaults to max_iters if not set")
    p.add_argument("--min_lr_ratio", type=float, default=0.1)

    # --- data & training ---
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=10000)
    p.add_argument("--device", type=str, default=None,
                    help="Defaults to cuda if available, else cpu")

    # --- logging & checkpoint ---
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    p.add_argument("--wandb_project", type=str, default=None,
                    help="If set, enable wandb logging under this project name")

    return p.parse_args()


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, device, num_batches):
    # 推理模式
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        total_loss += cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total_loss / num_batches


def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters

    min_lr = args.lr * args.min_lr_ratio

    # --- wandb ---
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # --- data ---
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    # --- model ---
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
    # 开启训练模式
    model.train()

    # --- optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # --- resume ---
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iter}")

    # --- checkpoint dir ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- training loop ---
    for it in range(start_iter, args.max_iters):
        t0 = time.time()

        new_lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr,
            min_learning_rate=min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        clip_gradients(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0

        # --- logging ---
        if it % args.log_interval == 0:
            print(f"iter {it:>6d} | loss {loss.item():.4f} | lr {new_lr:.6f} | {dt*1000:.0f}ms")
            if args.wandb_project:
                import wandb
                wandb.log({"train_loss": loss.item(), "lr": new_lr, "iteration": it})

        # --- eval ---
        if it > 0 and it % args.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length,
                args.device, args.eval_batches,
            )
            print(f"iter {it:>6d} | val_loss {val_loss:.4f}")
            if args.wandb_project:
                import wandb
                wandb.log({"val_loss": val_loss, "iteration": it})

        # --- checkpoint ---
        if it > 0 and it % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"iter {it:>6d} | saved checkpoint to {ckpt_path}")

    # --- final save ---
    ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{args.max_iters}.pt")
    save_checkpoint(model, optimizer, args.max_iters, ckpt_path)
    print(f"Training complete. Final checkpoint saved to {ckpt_path}")

    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
